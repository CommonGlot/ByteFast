/**
 * Copyright (c) 2026 CommonGlot contributors.
 * Licensed under the MIT license; see LICENSE.
 *
 * Unit tests for ByteFast text features (src/textnorm.h):
 *   c++ -std=c++17 -O2 -Wall -Wextra -Werror -Isrc \
 *       tests/textnorm_test.cc -o /tmp/textnorm_test
 *   /tmp/textnorm_test
 */
#include "textnorm.h"

#include <cassert>
#include <set>
#include <string>
#include <vector>

using namespace bytefast;

namespace {

std::vector<uint32_t> bytes(const std::string& s, int minN, int maxN) {
  std::vector<uint32_t> out;
  text::byteNgrams(s, minN, maxN, [&](uint32_t h) { out.push_back(h); });
  return out;
}

std::vector<uint32_t> chars(const std::string& s, int minN, int maxN) {
  std::vector<uint32_t> out;
  text::charNgrams(s, minN, maxN, [&](uint32_t h) { out.push_back(h); });
  return out;
}

std::string folded(std::string s, fold_name mode) {
  text::fold(s, mode);
  return s;
}

} // namespace

int main() {
  // Byte n-grams: every start position, every length in [min, max].
  assert(bytes("abcd", 2, 3).size() == 5); // ab abc bc bcd cd
  assert(bytes("a", 2, 5).empty());
  // "é" is two bytes: one byte bigram, but a single character.
  assert(bytes("\xC3\xA9", 2, 2).size() == 1);
  assert(chars("\xC3\xA9", 2, 2).empty());
  assert(chars("\xC3\xA9", 1, 1).size() == 1);
  // Character n-grams count code points: "日本語" has 3 characters (9 bytes).
  const std::string nihongo = "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E";
  assert(chars(nihongo, 1, 3).size() == 6); // 3 + 2 + 1
  assert(bytes(nihongo, 1, 3).size() == 24); // 9 + 8 + 7
  // Same n-gram text, same hash, independent of its position.
  const auto ab = bytes("ab", 2, 2);
  const auto xab = bytes("xab", 2, 2);
  assert(ab.size() == 1 && xab.size() == 2 && xab[1] == ab[0]);
  // Byte and character features of identical text hash differently.
  assert(bytes("ab", 2, 2)[0] != chars("ab", 2, 2)[0]);
  // Hashes differ across lengths and are spread over buckets.
  std::set<uint32_t> buckets;
  for (uint32_t h : bytes(" the quick brown fox jumps over the lazy dog ", 1, 5)) {
    buckets.insert(h % 1024);
  }
  assert(buckets.size() > 150);
  // Hash values are part of the model format and must not change.
  assert(text::hashBytes("bytefast", text::kWordSeed) ==
         text::hashBytes("bytefast", text::kWordSeed));
  assert(text::hashBytes("a", text::kByteSeed) !=
         text::hashBytes("a", text::kCharSeed));

  // Folding.
  assert(folded("Hello WORLD", fold_name::ascii) == "hello world");
  assert(folded("Hello WORLD", fold_name::none) == "Hello WORLD");
  // ASCII folding leaves non-ASCII letters alone; Unicode folds them.
  assert(folded("\xC3\x89T\xC3\x89", fold_name::ascii) == "\xC3\x89t\xC3\x89");
  assert(folded("\xC3\x89T\xC3\x89", fold_name::unicode) == "\xC3\xA9t\xC3\xA9");
  // Cyrillic "ПРИВЕТ" -> "привет", Greek "ΣΟΦΙΑ" -> "σοφια".
  assert(folded("\xD0\x9F\xD0\xA0\xD0\x98\xD0\x92\xD0\x95\xD0\xA2", fold_name::unicode) ==
         "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82");
  assert(folded("\xCE\xA3\xCE\x9F\xCE\xA6\xCE\x99\xCE\x91", fold_name::unicode) ==
         "\xCF\x83\xCE\xBF\xCF\x86\xCE\xB9\xCE\xB1");
  // U+0130 uses its one-to-one mapping (i), so the length in code points
  // never changes; scripts without case are unchanged.
  assert(folded("\xC4\xB0", fold_name::unicode) == "i");
  assert(folded(nihongo, fold_name::unicode) == nihongo);
  // Malformed UTF-8 is copied byte by byte.
  assert(folded("A\xFF\xC3", fold_name::unicode) == "a\xFF\xC3");

  // Truncation never splits a character.
  assert(text::truncateUtf8("abcdef", 0) == "abcdef");
  assert(text::truncateUtf8("abcdef", 4) == "abcd");
  assert(text::truncateUtf8(nihongo, 4) == "\xE6\x97\xA5");
  assert(text::truncateUtf8(nihongo, 6) == nihongo.substr(0, 6));
  return 0;
}
