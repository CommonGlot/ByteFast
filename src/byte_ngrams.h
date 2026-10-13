/**
 * Copyright (c) 2026 CommonGlot contributors.
 * Licensed under the MIT license; see LICENSE.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <stdexcept>

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
  const size_t orders = maxN - minN + 1;
  if (text.size() <= result.max_size() / orders) {
    result.reserve(text.size() * orders);
  }
  for (size_t i = 0; i < text.size(); ++i) {
    for (size_t n = minN; n <= maxN && n <= text.size() - i; ++n) {
      result.push_back(hashByteNgram(text, i, n) % buckets);
    }
  }
  return result;
}

// Bloom-style multiple independent hashes into one shared table.
// 1 hash is the fast baseline; 2 hashes provide more distinct combinations
// but roughly double the embedding lookups per byte n-gram.
inline std::vector<uint32_t> byteNgramMultiHashes(
    std::string_view text, size_t minN, size_t maxN,
    uint32_t buckets, uint32_t hashCount) {
  if (hashCount == 0 || hashCount > 4) {
    throw std::invalid_argument("hashCount must be between 1 and 4");
  }
  auto features = byteNgramHashes(text, minN, maxN, buckets);
  if (hashCount == 1 || buckets == 0) {
    return features;
  }
  std::vector<uint32_t> result;
  result.reserve(features.size() * hashCount);
  for (const uint32_t primary : features) {
    // Distinct deterministic mixes. This mixes an already reduced hash;
    // its independence is limited when buckets are small, so evaluate quality.
    uint32_t h = primary;
    for (uint32_t k = 0; k < hashCount; ++k) {
      if (k) {
        h ^= 0x9e3779b9u + k * 0x85ebca6bu;
        h ^= h >> 16;
        h *= 0x7feb352du;
        h ^= h >> 15;
        h *= 0x846ca68bu;
        h ^= h >> 16;
      }
      result.push_back(h % buckets);
    }
  }
  return result;
}

} // namespace bytefast
} // namespace fasttext
