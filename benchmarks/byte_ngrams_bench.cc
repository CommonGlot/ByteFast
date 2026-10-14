// CPU microbenchmark: byte feature extraction ONLY (not full model inference).
// c++ -std=c++17 -O3 -Isrc benchmarks/byte_ngrams_bench.cc -o byte_ngrams_bench
// ./byte_ngrams_bench
#include "byte_ngrams.h"
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

int main() {
  std::string sample = "The quick brown fox jumps over the lazy dog. "
                       "Bonjour le monde! Привет мир! こんにちは世界 ";
  while (sample.size() < 512) sample += sample;
  sample.resize(512);
  constexpr int repeats = 20000;
  constexpr uint32_t buckets = 524288;

  for (uint32_t hashes : {1u, 2u}) {
    volatile uint64_t checksum = 0;
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
      auto features = fasttext::bytefast::byteNgramMultiHashes(
          sample, 2, 5, buckets, hashes);
      checksum += features[static_cast<size_t>(i) % features.size()];
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    const size_t featureCount = fasttext::bytefast::byteNgramMultiHashes(
        sample, 2, 5, buckets, hashes).size();
    std::cout << "hashes=" << hashes << " docs=" << repeats
              << " bytes/doc=" << sample.size()
              << " features/doc=" << featureCount
              << " seconds=" << std::fixed << std::setprecision(4) << seconds
              << " docs/sec=" << repeats / seconds
              << " checksum=" << checksum << '\n';
  }
}
