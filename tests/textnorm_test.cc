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
  text::byteNgrams(s, minN, maxN, [&](uint32_t h, size_t) { out.push_back(h); });
  return out;
}

std::vector<uint32_t> chars(const std::string& s, int minN, int maxN) {
  std::vector<uint32_t> out;
  text::charNgrams(s, minN, maxN, [&](uint32_t h, size_t) { out.push_back(h); });
  return out;
}

std::string lowered(std::string s) {
  text::lowercaseUtf8(s);
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
  // Byte and character n-grams use different seeds.
  assert(text::hashBytes("a", text::kByteSeed) !=
         text::hashBytes("a", text::kCharSeed));

  // Positions: start offsets are reported, and position hashes differ by bin.
  std::vector<size_t> starts;
  text::byteNgrams("abc", 2, 2, [&](uint32_t, size_t i) { starts.push_back(i); });
  assert((starts == std::vector<size_t>{0, 1}));
  starts.clear();
  text::charNgrams(nihongo, 1, 1, [&](uint32_t, size_t i) { starts.push_back(i); });
  assert((starts == std::vector<size_t>{0, 3, 6}));

  // Lowercasing (used by the CommonLID-style normalization).
  assert(lowered("Hello WORLD") == "hello world");
  assert(lowered("\xC3\x89T\xC3\x89") == "\xC3\xA9t\xC3\xA9");
  // Cyrillic "ПРИВЕТ" -> "привет", Greek "ΣΟΦΙΑ" -> "σοφια".
  assert(lowered("\xD0\x9F\xD0\xA0\xD0\x98\xD0\x92\xD0\x95\xD0\xA2") ==
         "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82");
  assert(lowered("\xCE\xA3\xCE\x9F\xCE\xA6\xCE\x99\xCE\x91") ==
         "\xCF\x83\xCE\xBF\xCF\x86\xCE\xB9\xCE\xB1");
  // U+0130 uses its one-to-one mapping (i), so the length in code points
  // never changes; scripts without case are unchanged.
  assert(lowered("\xC4\xB0") == "i");
  assert(lowered(nihongo) == nihongo);
  // Malformed UTF-8 is copied byte by byte.
  assert(lowered("A\xFF\xC3") == "a\xFF\xC3");

  // CommonLID-style normalization (expected values from the evaluator's
  // regex cleaner, applied per whitespace token).
  auto norm = [](std::string s) {
    text::normalizeToken(s);
    return s;
  };
  assert(norm("Hello\xE2\x80\x94World,") == "helloworld");
  assert(norm("Caf\xC3\xA9!") == "caf\xC3\xA9");
  assert(norm("123").empty());
  assert(norm("2024!!").empty());
  assert(norm("\xD0\x9C\xD0\x98\xD0\xA0") == "\xD0\xBC\xD0\xB8\xD1\x80");
  assert(norm("snake_case") == "snake_case");
  assert(norm("\xE2\x80\x98quotes\xE2\x80\x99") == "quotes");
  assert(norm("&").empty());
  assert(norm("\xD9\xA1\xD9\xA2\xD9\xA3").empty());  // Arabic-Indic digits
  assert(norm("\xE6\x97\xA5\xE6\x9C\xAC\xE3\x80\x82") == "\xE6\x97\xA5\xE6\x9C\xAC");
  assert(norm("a\xE2\x80\x8D" "b") == "a\xE2\x80\x8D" "b");  // zero-width joiner is kept
  assert(norm("x\xCC\x81y") == "x\xCC\x81y");  // combining marks are kept
  assert(norm("\xC4\xB0zmir") == "i\xCC\x87zmir");  // as Python's str.lower()

  // Truncation never splits a character.
  assert(text::truncateUtf8("abcdef", 0) == "abcdef");
  assert(text::truncateUtf8("abcdef", 4) == "abcd");
  assert(text::truncateUtf8(nihongo, 4) == "\xE6\x97\xA5");
  assert(text::truncateUtf8(nihongo, 6) == nihongo.substr(0, 6));
  // -stripProb: tones, accents and Arabic/Hebrew vowel marks are removed;
  // marks that are letters of other scripts (Devanagari vowel signs) stay.
  auto strip = [](std::string t) {
    text::stripDiacritics(t);
    return t;
  };
  assert(strip("K\xC3\xA0 \xC9\xB2\xC9\x94g\xC9\x94n l\xC3\xA1j\xC9\x9Bn") ==
         "Ka \xC9\xB2\xC9\x94g\xC9\x94n laj\xC9\x9Bn");  // Dyula: Kà ɲɔgɔn lájɛn
  assert(strip("Ti\xE1\xBA\xBFng Vi\xE1\xBB\x87t") == "Tieng Viet");  // Tiếng Việt
  assert(strip("\xC9\x9B\xCC\x81") == "\xC9\x9B");  // ɛ + U+0301
  assert(strip("\xD9\x85\xD9\x8E\xD8\xB1\xD9\x92\xD8\xAD\xD9\x8E\xD8\xA8\xD9\x8B\xD8\xA7") ==
         "\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7");  // Arabic with harakat
  assert(strip("\xD7\xA9\xD7\x81\xD6\xB8\xD7\x9C\xD7\x95\xD6\xB9\xD7\x9D") ==
         "\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D");  // Hebrew with niqqud
  const std::string hindi = "\xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xA6\xE0\xA5\x80";
  assert(strip(hindi) == hindi);
  assert(strip("plain ascii 123") == "plain ascii 123");
  assert(strip("\xFF bad") == "\xFF bad");

  return 0;
}
