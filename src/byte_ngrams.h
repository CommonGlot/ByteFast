/**
 * Copyright (c) 2026 CommonGlot contributors.
 * Licensed under the MIT license; see LICENSE.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace fasttext {
namespace bytefast {

// FNV-1a over *unsigned bytes*, independent of char signedness.
// This intentionally differs from the legacy fastText hash.
inline uint32_t hashByteNgram(
    std::string_view bytes, size_t start, size_t length) {
  uint32_t hash = 2166136261u;
  for (size_t j = 0; j < length; ++j) {
    hash ^= static_cast<uint8_t>(bytes[start + j]);
    hash *= 16777619u;
  }
  // Domain separation by n-gram length.
  hash ^= static_cast<uint32_t>(length);
  hash *= 16777619u;
  return hash;
}

// Extract contiguous byte n-grams, *including* spaces and UTF-8 continuation
// bytes. No Unicode decoder, vocabulary, or word tokenizer is involved.
// Caller must strip supervised labels and the example delimiter first.
inline std::vector<uint32_t> byteNgramHashes(
    std::string_view text, size_t minN, size_t maxN, uint32_t buckets) {
  std::vector<uint32_t> result;
  if (minN == 0 || minN > maxN || buckets == 0) {
    return result;
  }
  for (size_t i = 0; i < text.size(); ++i) {
    for (size_t n = minN; n <= maxN && n <= text.size() - i; ++n) {
      result.push_back(hashByteNgram(text, i, n) % buckets);
    }
  }
  return result;
}

} // namespace bytefast
} // namespace fasttext
