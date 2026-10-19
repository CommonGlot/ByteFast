// Copyright (c) 2026 CommonGlot contributors. MIT License.

#include "lexicon.h"
#include "dictionary.h"
#include "safetensors.h"
#include "textnorm.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

namespace bytefast {

namespace {
uint64_t mix64(uint64_t h) {
  h ^= h >> 33;
  h *= 0xFF51AFD7ED558CCDull;
  h ^= h >> 33;
  h *= 0xC4CEB9FE1A85EC53ull;
  h ^= h >> 33;
  return h;
}
} // namespace

// The pair's key picks a block; kHashes 9-bit fields of a second hash pick
// the bits inside it.
void Lexicon::add(int32_t label, uint64_t word) {
  const uint64_t key = mix64(word ^ (uint64_t(label) + 1) * 0x9E3779B97F4A7C15ull);
  uint64_t* block = &bloom_[key % (bloom_.size() / kBlockWords) * kBlockWords];
  uint64_t bits = mix64(key);
  for (int32_t i = 0; i < kHashes; i++, bits >>= 9) {
    block[(bits >> 6) & 7] |= uint64_t(1) << (bits & 63);
  }
}

bool Lexicon::has(int32_t label, uint64_t word) const {
  const uint64_t key = mix64(word ^ (uint64_t(label) + 1) * 0x9E3779B97F4A7C15ull);
  const uint64_t* block = &bloom_[key % (bloom_.size() / kBlockWords) * kBlockWords];
  uint64_t bits = mix64(key);
  for (int32_t i = 0; i < kHashes; i++, bits >>= 9) {
    if (!(block[(bits >> 6) & 7] >> (bits & 63) & 1)) {
      return false;
    }
  }
  return true;
}

void Lexicon::build(const std::string& path, const Dictionary& dict,
    const std::string& prefix, int verbose) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::invalid_argument(path + " cannot be opened!");
  }
  std::vector<std::pair<int32_t, uint64_t>> entries;
  std::unordered_set<uint64_t> seen;
  int64_t skipped = 0;
  std::string line;
  while (std::getline(ifs, line)) {
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string label = line.substr(0, tab);
    std::string word = line.substr(tab + 1, line.find('\t', tab + 1) - tab - 1);
    int32_t id = dict.getId(label);
    if (id < 0 || dict.getType(id) != entry_type::label) {
      id = dict.getId(prefix + label);
    }
    if (id < 0 || dict.getType(id) != entry_type::label) {
      skipped++;
      continue;
    }
    uint64_t h;
    if (!text::normalizedHash(word, h)) {
      continue;
    }
    id -= dict.nwords();
    if (seen.insert(h * 0x9E3779B97F4A7C15ull + uint64_t(id)).second) {
      entries.emplace_back(id, h);
    }
  }
  const uint64_t blocks =
      std::max<uint64_t>(1, (entries.size() * kBitsPerEntry + 511) / 512);
  bloom_.assign(blocks * kBlockWords, 0);
  hasLexicon_.assign(dict.nlabels(), 0);
  for (const auto& e : entries) {
    add(e.first, e.second);
    hasLexicon_[e.first] = 1;
  }
  if (verbose > 0) {
    int32_t labels = 0;
    for (uint8_t h : hasLexicon_) {
      labels += h;
    }
    std::cerr << "Lexicon: " << entries.size() << " words for " << labels
              << " labels";
    if (skipped > 0) {
      std::cerr << " (" << skipped << " lines with unknown labels skipped)";
    }
    std::cerr << std::endl;
  }
}

bool Lexicon::empty() const {
  return bloom_.empty();
}

void Lexicon::save(SafeTensorsWriter& out) const {
  if (empty()) {
    return;
  }
  out.tensor("lexicon.bloom", "U64", {int64_t(bloom_.size())}, bloom_.data(),
      bloom_.size() * sizeof(uint64_t));
  out.tensor("lexicon.labels", "U8", {int64_t(hasLexicon_.size())},
      hasLexicon_.data(), hasLexicon_.size());
}

void Lexicon::load(SafeTensorsReader& in) {
  bloom_.clear();
  hasLexicon_.clear();
  if (!in.has("lexicon.bloom")) {
    return;
  }
  bloom_.resize(in.shape("lexicon.bloom")[0]);
  in.read("lexicon.bloom", "U64", bloom_.data(), bloom_.size() * sizeof(uint64_t));
  hasLexicon_.resize(in.shape("lexicon.labels")[0]);
  in.read("lexicon.labels", "U8", hasLexicon_.data(), hasLexicon_.size());
}

std::vector<uint64_t> Lexicon::words(const std::vector<std::string>& tokens) {
  std::vector<uint64_t> out;
  uint64_t h;
  for (const auto& token : tokens) {
    if (token != Dictionary::EOS && text::normalizedHash(token, h)) {
      out.push_back(h);
    }
  }
  return out;
}

bool Lexicon::passes(int32_t label, const std::vector<uint64_t>& words,
    double threshold, int32_t minWords) const {
  if (empty() || label < 0 || label >= int32_t(hasLexicon_.size()) ||
      !hasLexicon_[label] || int32_t(words.size()) < minWords) {
    return true;
  }
  // Stop as soon as the answer is known.
  const double need = threshold * words.size();
  int32_t known = 0;
  for (size_t i = 0; i < words.size(); i++) {
    known += has(label, words[i]);
    if (known >= need) {
      return true;
    }
    if (known + double(words.size() - i - 1) < need) {
      return false;
    }
  }
  return known >= need;
}

} // namespace bytefast
