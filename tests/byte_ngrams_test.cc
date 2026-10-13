/**
 * Copyright (c) 2026 CommonGlot contributors.
 * Licensed under the MIT license; see LICENSE.
 *
 * Standalone smoke test:
 *   c++ -std=c++17 -O2 -Wall -Wextra -Werror \
 *       -Isrc tests/byte_ngrams_test.cc -o /tmp/byte_ngrams_test
 *   /tmp/byte_ngrams_test
 */
#include "byte_ngrams.h"

#include <cassert>
#include <string>

using fasttext::bytefast::byteNgramHashes;
using fasttext::bytefast::hashByteNgram;

int main() {
  const std::string withSpace = "a b";
  const auto features = byteNgramHashes(withSpace, 2, 3, 100000);
  assert(features.size() == 3); // "a ", "a b", " b"
  assert(features[0] == hashByteNgram(withSpace, 0, 2) % 100000);
  assert(features[1] == hashByteNgram(withSpace, 0, 3) % 100000);
  assert(features[2] == hashByteNgram(withSpace, 1, 2) % 100000);

  // UTF-8 codepoint boundaries do not affect raw byte features.
  const std::string utf8 = "\xC3\xA9";
  assert(byteNgramHashes(utf8, 2, 2, 100000).size() == 1);

  assert(byteNgramHashes("x", 2, 5, 100000).empty());
  assert(byteNgramHashes("hi", 2, 5, 0).empty());
  assert(byteNgramHashes("hi", 0, 5, 100000).empty());
  const auto one = fasttext::bytefast::byteNgramMultiHashes(
      "a b", 2, 3, 100000, 1);
  const auto two = fasttext::bytefast::byteNgramMultiHashes(
      "a b", 2, 3, 100000, 2);
  assert(one == features);
  assert(two.size() == 2 * features.size());
  for (size_t i = 0; i < features.size(); ++i) {
    assert(two[2 * i] == features[i]);
    assert(two[2 * i + 1] < 100000);
  }
  bool rejected = false;
  try {
    (void)fasttext::bytefast::byteNgramMultiHashes(
        "abc", 2, 3, 100, 0);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
