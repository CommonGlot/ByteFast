// Copyright (c) 2026 CommonGlot contributors. MIT License.
//
// Percent-threshold lexicon check (Caswell et al., COLING 2020): a label is
// kept for a line only if enough of the line's words are known words of that
// label. The (label, word) pairs are kept in a blocked Bloom filter (all bits
// of a pair in one 64-byte block, so a lookup reads one cache line), so a few
// words outside a lexicon are counted as known (about 1-2%).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bytefast {

class Dictionary;
class SafeTensorsReader;
class SafeTensorsWriter;

class Lexicon {
 public:
  // Reads LABEL<TAB>WORD lines (other columns are ignored). LABEL may omit
  // the label prefix; words are normalized like the CommonLID input. Lines
  // whose label is not in the dictionary are skipped.
  void build(const std::string& path, const Dictionary& dict,
      const std::string& prefix, int verbose);
  bool empty() const;
  void save(SafeTensorsWriter&) const;
  void load(SafeTensorsReader&);

  // Hashes of the normalized words of a line's tokens (the end-of-line
  // token excluded).
  static std::vector<uint64_t> words(const std::vector<std::string>& tokens);
  // Whether the check applies: the label has a lexicon and the line has at
  // least minWords words.
  bool checks(int32_t label, size_t nwords, int32_t minWords) const;
  // Whether at least `threshold` of the words are in the lexicon of the
  // label (true when the check does not apply).
  bool passes(int32_t label, const std::vector<uint64_t>& words,
      double threshold, int32_t minWords) const;

 private:
  static constexpr int32_t kHashes = 7;
  static constexpr int32_t kBitsPerEntry = 10;
  static constexpr uint64_t kBlockWords = 8; // 512 bits
  void add(int32_t label, uint64_t word);
  bool has(int32_t label, uint64_t word) const;

  std::vector<uint64_t> bloom_;
  std::vector<uint8_t> hasLexicon_;
};

} // namespace bytefast
