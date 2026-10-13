/**
 * Copyright (c) 2026 CommonGlot contributors. MIT License.
 *
 * Experimental optional whole-word lexical sidecar for ByteFast.
 * It supplements raw-byte features; it never replaces the byte classifier.
 */
#include "bytefast_model.h"
#include <cctype>
#include <unordered_set>

namespace {

using Counts = std::unordered_map<uint32_t, uint32_t>;
struct WordCounts {
  uint32_t total=0;
  Counts languages;
};
struct WordWeight {
  uint32_t language=0;
  float logOdds=0;
};
using WordMap = std::unordered_map<std::string,std::vector<WordWeight>>;

std::string asciiLower(std::string s) {
  for (char& c:s) {
    auto b=static_cast<unsigned char>(c);
    if (b>='A' && b<='Z') c=static_cast<char>(b+32);
  }
  return s;
}

// Only a weak optional lexical segmentation, *not* a tokenizer used by the
// main ByteFast encoder. Non-ASCII UTF-8 bytes stay together; ASCII punctuation
// and ASCII whitespace delimit lexical pieces.
std::vector<std::string> words(std::string_view text, size_t maxBytes) {
  std::vector<std::string> out;
  const size_t n=std::min(text.size(),maxBytes);
  size_t start=0;
  bool inside=false;
  for (size_t i=0;i<=n;++i) {
    const uint8_t b=(i<n)?static_cast<uint8_t>(text[i]):0;
    const bool isLetter=(b>='A'&&b<='Z') || (b>='a'&&b<='z') ||
                        (b>='0'&&b<='9') || b>=128;
    if (isLetter && !inside) {
      inside=true;start=i;
    } else if (!isLetter && inside) {
      if (i-start>=2 && i-start<=48) {
        out.push_back(asciiLower(std::string(text.substr(start,i-start))));
      }
      inside=false;
    }
  }
  return out;
}

class Lexicon {
 public:
  std::vector<std::string> labels;
  WordMap entries;
  uint32_t inputMaxBytes=512;

  void train(const Model& base, const std::vector<Record>& docs,
             size_t maxWords, unsigned minCount) {
    labels=base.labels;inputMaxBytes=base.opt.maxBytes;
    std::unordered_map<std::string,uint32_t> labelIds;
    for (uint32_t i=0;i<labels.size();++i) labelIds[labels[i]]=i;
    std::unordered_map<std::string,WordCounts> counts;
    Counts totals;
    for (const auto& r:docs) {
      auto lid=labelIds.find(r.label);
      if (lid==labelIds.end()) continue;
      ++totals[lid->second];
      const auto tokens=words(r.text,inputMaxBytes);
      std::unordered_set<std::string> unique(tokens.begin(),tokens.end());
      for (const auto& token:unique) {
        auto& wc=counts[token];
        ++wc.total;
        ++wc.languages[lid->second];
      }
    }
    uint64_t documents=0;
    for (const auto& x:totals) documents+=x.second;
    struct Candidate {
      std::string word;
      float priority;
      std::vector<WordWeight> scores;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(counts.size());
    constexpr double beta=0.5;
    for (const auto& e:counts) {
      if (e.second.total<minCount) continue;
      std::vector<WordWeight> scores;
      for (const auto& lc:e.second.languages) {
        const double posterior=(lc.second+beta)/
            (e.second.total+beta*labels.size());
        const double prior=(totals[lc.first]+beta)/
            (documents+beta*labels.size());
        const float evidence=static_cast<float>(std::log(posterior/prior));
        if (evidence>0.25f) scores.push_back({lc.first,std::min(evidence,6.0f)});
      }
      if (scores.empty()) continue;
      std::sort(scores.begin(),scores.end(),
                [](const WordWeight& a,const WordWeight& b) {
                  return a.logOdds>b.logOdds;
                });
      if (scores.size()>4) scores.resize(4);
      const float priority=std::log1p(static_cast<float>(e.second.total))*
                           scores.front().logOdds;
      candidates.push_back({e.first,priority,std::move(scores)});
    }
    if (candidates.size()>maxWords) {
      std::nth_element(candidates.begin(),candidates.begin()+maxWords,
          candidates.end(),[](const Candidate& a,const Candidate& b) {
            return a.priority>b.priority;
          });
      candidates.resize(maxWords);
    }
    for (auto& c:candidates) entries.emplace(std::move(c.word),std::move(c.scores));
    std::cerr << "lexicon_words=" << entries.size()
              << " documents=" << documents << '\n';
  }

  uint32_t predict(const Model& model,const std::string& txt,float weight) const {
    const auto base=model.probabilities(model.embedding(model.features(txt)));
    std::vector<float> combined(base.size());
    for (size_t c=0;c<base.size();++c) combined[c]=
        std::log(std::max(base[c],1e-20f));
    const auto pieces=words(txt,model.opt.maxBytes);
    // Repetitions in boilerplate should not swamp the classifier.
    std::unordered_set<std::string> unique(pieces.begin(),pieces.end());
    size_t matched=0;
    for (const auto& w:unique) {
      const auto it=entries.find(w);
      if (it==entries.end()) continue;
      ++matched;
      for (const auto& s:it->second) {
        if (s.language<combined.size()) combined[s.language]+=weight*s.logOdds;
      }
    }
    if (matched>1) {
      // Centre lexical scores to be an average, not dependent on document length.
      // Base log probabilities were added separately, so adjust the total
      // lexical contribution while preserving relative score differences.
      // Implemented directly below for clarity using separate lexical values.
      std::vector<float> lexical(base.size(),0.0f);
      for (const auto& w:unique) {
        const auto it=entries.find(w);
        if (it==entries.end()) continue;
        for (const auto& s:it->second) {
          if (s.language<lexical.size()) lexical[s.language]+=s.logOdds;
        }
      }
      const float denom=std::sqrt(static_cast<float>(matched));
      for (size_t c=0;c<base.size();++c) combined[c]=
          std::log(std::max(base[c],1e-20f))+weight*lexical[c]/denom;
    }
    return static_cast<uint32_t>(
      std::max_element(combined.begin(),combined.end())-combined.begin());
  }

  void save(const std::string& path) const {
    std::ofstream f(path,std::ios::binary);
    if (!f) throw std::runtime_error("Could not save lexicon");
    const uint32_t magic=0x424c4558, version=1;
    Model::write(f,magic);Model::write(f,version);
    const uint32_t labelCount=static_cast<uint32_t>(labels.size());
    Model::write(f,labelCount);Model::write(f,inputMaxBytes);
    for (const auto& s:labels) {
      uint32_t n=static_cast<uint32_t>(s.size());
      Model::write(f,n);f.write(s.data(),n);
    }
    const uint32_t count=static_cast<uint32_t>(entries.size());
    Model::write(f,count);
    for (const auto& kv:entries) {
      const uint32_t n=static_cast<uint32_t>(kv.first.size());
      Model::write(f,n);f.write(kv.first.data(),n);
      const uint32_t k=static_cast<uint32_t>(kv.second.size());
      Model::write(f,k);
      for(const auto& score:kv.second) {
        Model::write(f,score.language);Model::write(f,score.logOdds);
      }
    }
    if (!f) throw std::runtime_error("Lexicon write failed");
  }
  void load(const Model& base,const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if (!f) throw std::runtime_error("Could not open lexicon");
    uint32_t magic,version,nlabels,nwords;
    Model::read(f,magic);Model::read(f,version);
    if (magic!=0x424c4558 || version!=1)
      throw std::runtime_error("Invalid lexicon format");
    Model::read(f,nlabels);Model::read(f,inputMaxBytes);
    if(nlabels!=base.labels.size() || inputMaxBytes!=base.opt.maxBytes)
      throw std::runtime_error("Lexicon and base model incompatible");
    auto readString=[&f](uint32_t cap) {
      uint32_t n;Model::read(f,n);
      if(n>cap) throw std::runtime_error("Corrupt lexicon string");
      std::string s(n,'\0');
      if(n && !f.read(&s[0],n)) throw std::runtime_error("Truncated lexicon");
      return s;
    };
    labels.clear();
    for(uint32_t i=0;i<nlabels;++i) labels.push_back(readString(256));
    if(labels!=base.labels) throw std::runtime_error("Mismatched labels");
    Model::read(f,nwords);
    if(nwords>1000000) throw std::runtime_error("Excessive lexicon entries");
    for(uint32_t i=0;i<nwords;++i) {
      auto s=readString(48);
      uint32_t k;Model::read(f,k);
      if(k>4) throw std::runtime_error("Corrupt lexical association count");
      std::vector<WordWeight> weight(k);
      for (auto& v:weight) {
        Model::read(f,v.language);Model::read(f,v.logOdds);
        if(v.language>=nlabels || !std::isfinite(v.logOdds))
          throw std::runtime_error("Invalid lexicon weights");
      }
      entries.emplace(std::move(s),std::move(weight));
    }
  }
};
void help() {
  std::cerr<<"Usage:\n"
    <<" bytefast-lexical build BASE_MODEL TRAIN_DATA OUTPUT.lex [maxWords] [minCount]\n"
    <<" bytefast-lexical predict BASE_MODEL LEXICON INPUT [-weight FLOAT]\n"
    <<" bytefast-lexical bench BASE_MODEL LEXICON INPUT [-weight FLOAT]\n";
}
} // namespace

int main(int argc,char** argv) {
  try {
    if (argc<5) {help();return 2;}
    const std::string cmd=argv[1];
    Model model;model.load(argv[2]);
    Lexicon lex;
    if(cmd=="build") {
      const size_t cap=argc>5?std::stoul(argv[5]):50000;
      const uint32_t count=argc>6?std::stoul(argv[6]):3;
      if(cap>1000000 || !count) throw std::runtime_error("Invalid lexical budget");
      lex.train(model,readRecords(argv[3]),cap,count);
      lex.save(argv[4]);
    } else if(cmd=="predict" || cmd=="bench") {
      if(argc<5) {help();return 2;}
      lex.load(model,argv[3]);
      float weight=1.0f;
      if(argc>5) {
        if(argc!=7 || std::string(argv[5])!="-weight")
          throw std::runtime_error("Expected -weight FLOAT");
        weight=std::stof(argv[6]);
      }
      if(!std::isfinite(weight) || weight<0 || weight>10)
        throw std::runtime_error("Invalid lexical weight");
      std::ifstream f(argv[4]);
      if(!f) throw std::runtime_error("Missing input file");
      std::string line;size_t total=0;
      const auto start=std::chrono::steady_clock::now();
      while(std::getline(f,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        const auto prediction=lex.predict(model,line,weight);
        if(cmd=="predict") std::cout<<"__label__"<<model.labels[prediction]<<'\n';
        ++total;
      }
      if(cmd=="bench") {
        const auto sec=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-start).count();
        std::cout<<"documents="<<total<<" seconds="<<sec
                 <<" docs_per_second="<<total/sec<<'\n';
      }
    } else {help();return 2;}
    return 0;
  } catch (const std::exception& e) {
    std::cerr<<"Lexical error: "<<e.what()<<'\n';return 1;
  }
}
