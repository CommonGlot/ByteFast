// Copyright (c) 2026 CommonGlot contributors. MIT License.
//
// Minimal reader and writer for the safetensors format
// (https://github.com/huggingface/safetensors): an 8-byte little-endian
// header length, a JSON header that maps tensor names to dtype, shape and
// byte offsets (plus a "__metadata__" map of strings), then the tensor bytes.

#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace bytefast {

class SafeTensorsWriter {
 public:
  void metadata(const std::string& key, const std::string& value);
  // The data must stay valid until write().
  void tensor(
      const std::string& name,
      const std::string& dtype,
      const std::vector<int64_t>& shape,
      const void* data,
      size_t bytes);
  // Same, for data the writer keeps.
  void tensor(
      const std::string& name,
      const std::string& dtype,
      const std::vector<int64_t>& shape,
      std::string data);
  void write(const std::string& path) const;

 private:
  struct Entry {
    std::string name, dtype;
    std::vector<int64_t> shape;
    const void* data;
    size_t bytes;
  };
  std::map<std::string, std::string> metadata_;
  std::vector<Entry> tensors_;
  std::vector<std::unique_ptr<std::string>> owned_;
};

class SafeTensorsReader {
 public:
  explicit SafeTensorsReader(const std::string& path);
  static bool isSafeTensors(const std::string& path);

  bool has(const std::string& name) const;
  const std::string& metadata(const std::string& key) const;
  int64_t metadataInt(const std::string& key) const;
  const std::vector<int64_t>& shape(const std::string& name) const;
  size_t bytes(const std::string& name) const;
  // Reads a whole tensor into dst; dtype and size must match.
  void read(
      const std::string& name,
      const std::string& dtype,
      void* dst,
      size_t bytes);

 private:
  struct Entry {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin, end;
  };
  const Entry& entry(const std::string& name) const;
  std::string path_;
  std::ifstream in_;
  uint64_t dataStart_;
  std::map<std::string, std::string> metadata_;
  std::map<std::string, Entry> tensors_;
};

} // namespace bytefast
