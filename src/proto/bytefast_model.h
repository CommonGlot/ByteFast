#pragma once
// ByteFast CPU-only supervised classifier prototype.
// A standalone, self-contained executable so experiments cannot break legacy fastText.
// MIT License. Copyright 2026 CommonGlot contributors.
#include "byte_ngrams.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
struct Options {
  uint32_t bucket = 65536, dim = 64, minN = 2, maxN = 5;
  uint32_t maxBytes = 512, epochs = 5, asciiFold = 0;
  float lr = 0.15f;
};
struct Record { std::string label, text; };
bool parseExample(const std::string& line, Record& r) {
  constexpr const char* prefix = "__label__";
  if (line.compare(0, 9, prefix) != 0) return false;
  auto delim = line.find_first_of(" \t", 9);
  if (delim == std::string::npos || delim == 9) return false;
  r.label = line.substr(9, delim-9);
  r.text = line.substr(delim+1);
  return !r.text.empty();
}
std::vector<Record> readRecords(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open: " + path);
  std::vector<Record> lines;
  std::string s;
  while (std::getline(f,s)) {
    if (!s.empty() && s.back()=='\r') s.pop_back();
    Record r;
    if (parseExample(s,r)) lines.push_back(std::move(r));
  }
  if (lines.empty()) throw std::runtime_error("No valid labeled lines: " + path);
  return lines;
}
uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU;
  x ^= x >> 15; x *= 0x846ca68bU;
  x ^= x >> 16; return x;
}
class Model {
 public:
  Options opt;
  std::vector<std::string> labels;
  std::vector<float> input, output;
  std::vector<float> bias;

  std::vector<uint32_t> features(const std::string& raw) const {
    const size_t length = std::min(raw.size(), static_cast<size_t>(opt.maxBytes));
    const std::string_view s(raw.data(), length);
    std::vector<uint32_t> ids;
    if (length < opt.minN) return ids;
    ids.reserve(length * (opt.maxN - opt.minN + 1));
    // Hash the actual bytes first; independently mix each hash before modulo.
    // Spaces and punctuation are never split or discarded.
    for (size_t pos=0; pos<length; ++pos) {
      uint32_t rolling=2166136261u;
      for (uint32_t n=1; n<=opt.maxN && n<=length-pos; ++n) {
        uint8_t byte=static_cast<uint8_t>(s[pos+n-1]);
        if (opt.asciiFold && byte >= 'A' && byte <= 'Z') byte += 32;
        rolling ^= byte;
        rolling *= 16777619u;
        if (n<opt.minN) continue;
        // Match hashByteNgram: finalize by incorporating the n-gram length.
        const uint32_t base=(rolling ^ n)*16777619u;
        ids.push_back(mix(base) % opt.bucket);
      }
    }
    return ids;
  }
  void initialize(const std::vector<Record>& records) {
    std::unordered_map<std::string,uint32_t> seen;
    for (const auto& r: records) if (!seen.count(r.label)) {
      seen[r.label]=static_cast<uint32_t>(labels.size());
      labels.push_back(r.label);
    }
    const uint64_t cells=uint64_t(opt.bucket)*opt.dim;
    const uint64_t outputCells=uint64_t(labels.size())*opt.dim;
    uint64_t labelBytes=0;
    for (const auto& s:labels) labelBytes+=s.size()+sizeof(uint32_t);
    const uint64_t estimatedFileBytes=64+labelBytes+sizeof(float)*
        (cells+outputCells+labels.size());
    if (estimatedFileBytes>=1000000000ULL)
      throw std::runtime_error("ByteFast saved model would exceed 1 GB limit");
    input.resize(static_cast<size_t>(cells));
    output.resize(labels.size()*opt.dim);
    bias.assign(labels.size(),0);
    std::mt19937 gen(17);
    std::uniform_real_distribution<float> d(-0.1f,0.1f);
    for (float& w:input) w=d(gen);
    for (float& w:output) w=d(gen);
  }
  std::vector<float> embedding(const std::vector<uint32_t>& ids) const {
    std::vector<float> h(opt.dim,0.0f);
    if (ids.empty()) return h;
    for (uint32_t id:ids) {
      const float* w=input.data()+size_t(id)*opt.dim;
      for (uint32_t j=0;j<opt.dim;++j) h[j]+=w[j];
    }
    const float inv=1.0f/ids.size();
    for (float& v:h) v*=inv;
    return h;
  }
  std::vector<float> probabilities(const std::vector<float>& h) const {
    std::vector<float> p(labels.size());
    float highest=-std::numeric_limits<float>::infinity();
    for (size_t c=0;c<labels.size();++c) {
      float v=bias[c];
      const float* w=output.data()+c*opt.dim;
      for (uint32_t j=0;j<opt.dim;++j) v+=w[j]*h[j];
      p[c]=v;
      highest=std::max(highest,v);
    }
    float sum=0;
    for (float& x:p) { x=std::exp(x-highest); sum+=x; }
    for (float& x:p) x/=sum;
    return p;
  }
  uint32_t predict(const std::string& s, float* confidence=nullptr,
                   float* margin=nullptr) const {
    const auto ids=features(s);
    if (ids.empty()) {
      if (confidence) *confidence=0.0f;
      if (margin) *margin=0.0f;
      return 0;
    }
    const auto p=probabilities(embedding(ids));
    uint32_t best=0;
    float largest=-1.0f, second=-1.0f;
    for (uint32_t i=0;i<p.size();++i) {
      if (p[i]>largest) {
        second=largest; largest=p[i]; best=i;
      } else if (p[i]>second) second=p[i];
    }
    if (confidence) *confidence=largest;
    if (margin) *margin=largest-std::max(second,0.0f);
    return best;
  }
  void train(const std::vector<Record>& records) {
    initialize(records);
    std::unordered_map<std::string,uint32_t> idx;
    for (uint32_t i=0;i<labels.size();++i) idx[labels[i]]=i;
    std::vector<size_t> order(records.size());
    std::iota(order.begin(),order.end(),0);
    std::mt19937 rng(1337);
    for (uint32_t epoch=0;epoch<opt.epochs;++epoch) {
      std::shuffle(order.begin(),order.end(),rng);
      double loss=0;
      for (size_t i:order) {
        const auto ids=features(records[i].text);
        if (ids.empty()) continue;
        const auto h=embedding(ids);
        const auto p=probabilities(h);
        const uint32_t gold=idx.at(records[i].label);
        loss-=std::log(std::max(p[gold],1e-20f));
        std::vector<float> grad(opt.dim,0.0f);
        for (uint32_t c=0;c<labels.size();++c) {
          const float g=p[c]-(c==gold?1.f:0.f);
          float* w=output.data()+size_t(c)*opt.dim;
          for (uint32_t j=0;j<opt.dim;++j) {
            grad[j]+=g*w[j];
            w[j]-=opt.lr*g*h[j];
          }
          bias[c]-=opt.lr*g;
        }
        const float scale=opt.lr/ids.size();
        for (uint32_t id:ids) {
          float* w=input.data()+size_t(id)*opt.dim;
          for (uint32_t j=0;j<opt.dim;++j) w[j]-=scale*grad[j];
        }
      }
      std::cerr << "epoch=" << epoch+1 << " loss=" << loss/records.size() << '\n';
    }
  }
  template<class T> static void write(std::ostream& f, const T& value) {
    f.write(reinterpret_cast<const char*>(&value),sizeof(T));
  }
  template<class T> static void read(std::istream& f, T& value) {
    if (!f.read(reinterpret_cast<char*>(&value),sizeof(T)))
      throw std::runtime_error("Corrupt model header");
  }
  void save(const std::string& path) const {
    std::ofstream f(path,std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write model");
    const uint32_t magic=0x42594654U, version=2;
    write(f,magic);write(f,version);
    write(f,opt.bucket);write(f,opt.dim);write(f,opt.minN);write(f,opt.maxN);
    // Former hash-count and position fields, kept so the layout is unchanged.
    const uint32_t hashes=1, positions=0;
    write(f,hashes);write(f,positions);write(f,opt.maxBytes);
    write(f,opt.asciiFold);
    uint32_t count=static_cast<uint32_t>(labels.size());
    write(f,count);
    for (const auto& s:labels) {
      const uint32_t n=static_cast<uint32_t>(s.size());
      write(f,n);f.write(s.data(),n);
    }
    f.write(reinterpret_cast<const char*>(input.data()),input.size()*sizeof(float));
    f.write(reinterpret_cast<const char*>(output.data()),output.size()*sizeof(float));
    f.write(reinterpret_cast<const char*>(bias.data()),bias.size()*sizeof(float));
    if (!f) throw std::runtime_error("Model write failed");
  }
  void load(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read model");
    uint32_t magic,version,count,hashes,positions;
    read(f,magic);read(f,version);
    if (magic!=0x42594654U || (version!=1 && version!=2)) throw std::runtime_error("Invalid ByteFast model");
    read(f,opt.bucket);read(f,opt.dim);read(f,opt.minN);read(f,opt.maxN);
    read(f,hashes);read(f,positions);read(f,opt.maxBytes);
    if (hashes!=1 || positions!=0)
      throw std::runtime_error("This model uses -hashCount or -positionBins, "
                               "which ByteFast no longer supports");
    if (version>=2) read(f,opt.asciiFold);
    read(f,count);
    if (!opt.bucket || !opt.dim || !opt.minN || opt.minN>opt.maxN ||
        opt.maxN>16 || !opt.maxBytes || opt.asciiFold>1 || !count || count>10000 ||
        64ULL+4ULL*(uint64_t(opt.bucket)*opt.dim+
          uint64_t(count)*opt.dim+count)>=1000000000ULL)
      throw std::runtime_error("Invalid model dimensions");
    labels.clear();
    for (uint32_t i=0;i<count;++i) {
      uint32_t n;read(f,n);
      if (n>256) throw std::runtime_error("Invalid label length");
      std::string s(n,'\0');
      if (!f.read(&s[0],n)) throw std::runtime_error("Truncated labels");
      labels.push_back(s);
    }
    input.resize(size_t(opt.bucket)*opt.dim);
    output.resize(size_t(count)*opt.dim);
    bias.resize(count);
    if (!f.read(reinterpret_cast<char*>(input.data()),input.size()*sizeof(float)) ||
        !f.read(reinterpret_cast<char*>(output.data()),output.size()*sizeof(float)) ||
        !f.read(reinterpret_cast<char*>(bias.data()),bias.size()*sizeof(float)))
      throw std::runtime_error("Truncated weights");
  }
};
uint32_t numeric(const char* s) { return static_cast<uint32_t>(std::stoul(s)); }
void usage() {
  std::cerr << "Usage:\n"
            << "  bytefast train INPUT MODEL [-bucket N] [-dim N] [-epoch N]"
               " [-lr FLOAT] [-minn N] [-maxn N] [-maxBytes N]"
               " [-asciiFold 0|1]\n"
            << "  bytefast test MODEL LABELED_FILE\n"
            << "  bytefast predict MODEL TEXT_FILE [-rejectThreshold 0..1] [-marginThreshold 0..1]\\n"
            << "  bytefast bench MODEL TEXT_FILE\n";
}
}
