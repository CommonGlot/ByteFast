/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dictionary.h"
#include "safetensors.h"

#include <assert.h>

#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <unordered_set>

#include "textnorm.h"

namespace bytefast {

namespace {
bool readWordNoNewline(std::string_view& in, std::string_view& word) {
  const std::string_view spaces(" \n\r\t\v\f\0");
  std::string_view::size_type begin = in.find_first_not_of(spaces);
  if (begin == std::string_view::npos) {
    in.remove_prefix(in.size());
    return false;
  }
  in.remove_prefix(begin);
  word = in.substr(0, in.find_first_of(spaces));
  in.remove_prefix(word.size());
  return true;
}
} // namespace

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary(std::shared_ptr<Args> args)
    : args_(args),
      word2int_(MAX_VOCAB_SIZE, -1),
      size_(0),
      nwords_(0),
      nlabels_(0),
      ntokens_(0) {}

Dictionary::Dictionary(std::shared_ptr<Args> args, SafeTensorsReader& in)
    : args_(args),
      size_(0),
      nwords_(0),
      nlabels_(0),
      ntokens_(0) {
  load(in);
}

int32_t Dictionary::find(const std::string_view w) const {
  return find(w, hash(w));
}

int32_t Dictionary::find(const std::string_view w, uint32_t h) const {
  int32_t word2intsize = word2int_.size();
  int32_t id = h % word2intsize;
  while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
    id = (id + 1) % word2intsize;
  }
  return id;
}

void Dictionary::add(const std::string& w) {
  int32_t h = find(w);
  ntokens_++;
  if (word2int_[h] == -1) {
    entry e;
    e.word = w;
    e.count = 1;
    e.type = getType(w);
    words_.push_back(e);
    word2int_[h] = size_++;
  } else {
    words_[word2int_[h]].count++;
  }
}

int32_t Dictionary::nwords() const {
  return nwords_;
}

int32_t Dictionary::nlabels() const {
  return nlabels_;
}

int64_t Dictionary::ntokens() const {
  return ntokens_;
}

const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

int32_t Dictionary::getId(const std::string_view w, uint32_t h) const {
  int32_t id = find(w, h);
  return word2int_[id];
}

int32_t Dictionary::getId(const std::string_view w) const {
  int32_t h = find(w);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

entry_type Dictionary::getType(const std::string_view w) const {
  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

// The correct implementation of fnv should be:
// h = h ^ uint32_t(uint8_t(str[i]));
// Unfortunately, earlier version of fasttext used
// h = h ^ uint32_t(str[i]);
// which is undefined behavior (as char can be signed or unsigned).
// Since all fasttext models that were already released were trained
// using signed char, we fixed the hash function to make models
// compatible whatever compiler is used.
uint32_t Dictionary::hash(const std::string_view str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(int8_t(str[i]));
    h = h * 16777619;
  }
  return h;
}

void Dictionary::computeSubwords(
    const std::string& word,
    std::vector<int32_t>& ngrams) const {
  for (size_t i = 0; i < word.size(); i++) {
    if ((word[i] & 0xC0) == 0x80) {
      continue;
    }
    // The FNV hash is extended byte by byte, giving hash(ngram) for every
    // prefix without building each n-gram string.
    uint32_t h = 2166136261;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      do {
        h = (h ^ uint32_t(int8_t(word[j++]))) * 16777619;
      } while (j < word.size() && (word[j] & 0xC0) == 0x80);
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        pushHash(ngrams, h % args_->bucket);
      }
    }
  }
}

void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    words_[i].subwords.push_back(i);
    if (words_[i].word != EOS) {
      computeSubwords(word, words_[i].subwords);
    }
  }
}

bool Dictionary::readWord(std::istream& in, std::string& word) const {
  int c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  int64_t minThreshold = 1;
  while (readWord(in, word)) {
    add(word);
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
    }
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      threshold(minThreshold);
    }
  }
  applyWordlist();
  threshold(args_->minCount);
  initNgrams();
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}

// Keep only words listed in -wordlist (first field of each line, folded like
// the input). Labels and the end-of-sentence token are always kept.
void Dictionary::applyWordlist() {
  if (args_->wordlist.empty()) {
    return;
  }
  std::ifstream ifs(args_->wordlist);
  if (!ifs.is_open()) {
    throw std::invalid_argument(args_->wordlist + " cannot be opened!");
  }
  std::unordered_set<std::string> allowed;
  std::string line;
  while (std::getline(ifs, line)) {
    std::string_view rest(line), word;
    if (readWordNoNewline(rest, word)) {
      allowed.emplace(word);
    }
  }
  words_.erase(
      remove_if(
          words_.begin(),
          words_.end(),
          [&](const entry& e) {
            return e.type == entry_type::word && e.word != EOS &&
                allowed.count(e.word) == 0;
          }),
      words_.end());
  if (args_->verbose > 0) {
    std::cerr << "Wordlist entries: " << allowed.size() << std::endl;
  }
}

// Drops words seen fewer than t times; labels are always kept.
void Dictionary::threshold(int64_t t) {
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
    if (e1.type != e2.type) {
      return e1.type < e2.type;
    }
    return e1.count > e2.count;
  });
  words_.erase(
      remove_if(
          words_.begin(),
          words_.end(),
          [&](const entry& e) {
            return e.type == entry_type::word && e.count < t;
          }),
      words_.end());
  words_.shrink_to_fit();
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  std::fill(word2int_.begin(), word2int_.end(), -1);
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) {
      nwords_++;
    }
    if (it->type == entry_type::label) {
      nlabels_++;
    }
  }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  for (auto& w : words_) {
    if (w.type == type) {
      counts.push_back(w.count);
    }
  }
  return counts;
}

void Dictionary::addSubwords(
    std::vector<int32_t>& line,
    const std::string_view token,
    int32_t wid) const {
  if (wid < 0) { // out of vocab
    if (token != EOS) {
      std::string concat;
      concat.reserve(BOW.size() + token.size() + EOW.size());
      concat += BOW;
      concat.append(token.data(), token.size());
      concat += EOW;
      computeSubwords(concat, line);
    }
  } else {
    if (args_->maxn <= 0) { // in vocab w/o subwords
      line.push_back(wid);
    } else { // in vocab w/ subwords
      const std::vector<int32_t>& ngrams = getSubwords(wid);
      line.insert(line.end(), ngrams.cbegin(), ngrams.cend());
    }
  }
}

// Byte n-grams over the words of the line joined by single
// spaces, so n-grams cross word boundaries. The text is padded with a space
// on each side, which makes the first and last words look like the others.
void Dictionary::addTextNgrams(
    std::vector<int32_t>& line,
    const std::vector<std::string>& tokens) const {
  if (args_->maxb <= 0) {
    return;
  }
  const uint32_t buckets = static_cast<uint32_t>(args_->bucket);
  std::string joined;
  for (const auto& token : tokens) {
    if (token == EOS) {
      continue;
    }
    if (!joined.empty()) {
      joined.push_back(' ');
    }
    joined += token;
  }
  if (joined.empty()) {
    return;
  }
  std::string padded = " ";
  padded += text::truncateUtf8(joined, args_->maxBytes);
  padded.push_back(' ');
  text::byteNgrams(padded, args_->minb, args_->maxb,
      [&](uint32_t h) { pushHash(line, h % buckets); });
}

void Dictionary::reset(std::istream& in) const {
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
}

int32_t Dictionary::readLine(
    std::istream& in,
    std::vector<std::string>& tokens,
    std::vector<int32_t>& labels) const {
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  tokens.clear();
  labels.clear();
  while (readWord(in, token)) {
    ntokens++;
    addToken(token, tokens, labels);
    if (token == EOS) {
      break;
    }
  }
  return ntokens;
}

void Dictionary::addToken(
    const std::string& token,
    std::vector<std::string>& tokens,
    std::vector<int32_t>& labels) const {
  if (getType(token) == entry_type::label) {
    int32_t wid = getId(token);
    if (wid >= 0) {
      labels.push_back(wid - nwords_);
    }
  } else {
    tokens.push_back(token);
  }
}

void Dictionary::addFeatures(
    const std::vector<std::string>& tokens,
    std::vector<int32_t>& words) const {
  words.clear();
  for (const auto& token : tokens) {
    addSubwords(words, token, getId(token));
  }
  addTextNgrams(words, tokens);
}

int32_t Dictionary::getLine(
    std::istream& in,
    std::vector<int32_t>& words,
    std::vector<int32_t>& labels) const {
  std::vector<std::string> tokens;
  int32_t ntokens = readLine(in, tokens, labels);
  addFeatures(tokens, words);
  return ntokens;
}

// Training variant: with probability -cropProb, keep only a random
// contiguous span of -cropMin..-cropMax characters of the line.
int32_t Dictionary::getLine(
    std::istream& in,
    std::vector<int32_t>& words,
    std::vector<int32_t>& labels,
    std::minstd_rand& rng) const {
  std::vector<std::string> tokens;
  int32_t ntokens = readLine(in, tokens, labels);
  if (args_->cropProb > 0 &&
      std::uniform_real_distribution<>(0, 1)(rng) < args_->cropProb) {
    cropTokens(tokens, rng);
  }
  // Normalize after cropping, as when a short input is cleaned before
  // prediction.
  if (args_->normProb > 0 &&
      std::uniform_real_distribution<>(0, 1)(rng) < args_->normProb) {
    std::vector<std::string> kept;
    for (auto& token : tokens) {
      if (token != EOS) {
        text::normalizeToken(token);
      }
      if (!token.empty()) {
        kept.push_back(std::move(token));
      }
    }
    tokens.swap(kept);
  }
  // Optional diacritics (tones, accents, Arabic and Hebrew vowel marks) are
  // removed last, so a line can be both normalized and stripped.
  if (args_->stripProb > 0 &&
      std::uniform_real_distribution<>(0, 1)(rng) < args_->stripProb) {
    for (auto& token : tokens) {
      if (token != EOS) {
        text::stripDiacritics(token);
      }
    }
  }
  addFeatures(tokens, words);
  return ntokens;
}

void Dictionary::cropTokens(
    std::vector<std::string>& tokens,
    std::minstd_rand& rng) const {
  const bool eos = !tokens.empty() && tokens.back() == EOS;
  if (eos) {
    tokens.pop_back();
  }
  const int32_t length =
      std::uniform_int_distribution<int32_t>(args_->cropMin, args_->cropMax)(rng);
  std::string joined;
  for (const auto& token : tokens) {
    if (!joined.empty()) {
      joined.push_back(' ');
    }
    joined += token;
  }
  std::vector<size_t> starts;
  for (size_t i = 0; i < joined.size(); i++) {
    if (!text::isContinuation(static_cast<uint8_t>(joined[i]))) {
      starts.push_back(i);
    }
  }
  if (length < static_cast<int32_t>(starts.size())) {
    const size_t first =
        std::uniform_int_distribution<size_t>(0, starts.size() - length)(rng);
    const size_t begin = starts[first];
    const size_t end = first + length < starts.size() ? starts[first + length]
                                                      : joined.size();
    std::string_view span(joined.data() + begin, end - begin), word;
    tokens.clear();
    while (readWordNoNewline(span, word)) {
      tokens.emplace_back(word);
    }
  }
  if (eos) {
    tokens.push_back(EOS);
  }
}

void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  hashes.push_back(nwords_ + id);
}

std::string Dictionary::getLabel(int32_t lid) const {
  if (lid < 0 || lid >= nlabels_) {
    throw std::invalid_argument(
        "Label id is out of range [0, " + std::to_string(nlabels_) + "]");
  }
  return words_[lid + nwords_].word;
}

// The vocabulary is stored as three tensors: the entries' UTF-8 strings,
// each ended by a NUL byte, their counts and their types (0 word, 1 label).
void Dictionary::save(SafeTensorsWriter& out) const {
  std::string strings;
  std::string counts(size_ * sizeof(int64_t), '\0');
  std::string types(size_, '\0');
  for (int32_t i = 0; i < size_; i++) {
    strings += words_[i].word;
    strings.push_back('\0');
    std::memcpy(&counts[i * sizeof(int64_t)], &words_[i].count, sizeof(int64_t));
    types[i] = static_cast<char>(words_[i].type);
  }
  out.metadata("vocab.nwords", std::to_string(nwords_));
  out.metadata("vocab.nlabels", std::to_string(nlabels_));
  out.metadata("vocab.ntokens", std::to_string(ntokens_));
  const int64_t nbytes = strings.size();
  out.tensor("vocab.strings", "U8", {nbytes}, std::move(strings));
  out.tensor("vocab.counts", "I64", {size_}, std::move(counts));
  out.tensor("vocab.types", "I8", {size_}, std::move(types));

}

void Dictionary::load(SafeTensorsReader& in) {
  nwords_ = in.metadataInt("vocab.nwords");
  nlabels_ = in.metadataInt("vocab.nlabels");
  ntokens_ = in.metadataInt("vocab.ntokens");
  size_ = nwords_ + nlabels_;
  std::string strings(in.bytes("vocab.strings"), '\0');
  in.read("vocab.strings", "U8", &strings[0], strings.size());
  std::vector<int64_t> counts(size_);
  in.read("vocab.counts", "I64", counts.data(), size_ * sizeof(int64_t));
  std::vector<int8_t> types(size_);
  in.read("vocab.types", "I8", types.data(), size_);
  words_.clear();
  size_t start = 0;
  for (int32_t i = 0; i < size_; i++) {
    size_t end = strings.find('\0', start);
    if (end == std::string::npos) {
      throw std::invalid_argument("Corrupt vocabulary");
    }
    entry e;
    e.word = strings.substr(start, end - start);
    e.count = counts[i];
    e.type = static_cast<entry_type>(types[i]);
    words_.push_back(e);
    start = end + 1;
  }
  initLookup();
}

void Dictionary::initLookup() {
  initNgrams();
  int32_t word2intsize = std::ceil(size_ / 0.7);
  word2int_.assign(word2intsize, -1);
  for (int32_t i = 0; i < size_; i++) {
    word2int_[find(words_[i].word)] = i;
  }
}

void Dictionary::init() {
  initNgrams();
}

} // namespace bytefast
