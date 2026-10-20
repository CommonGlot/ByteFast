// Copyright (c) 2026 CommonGlot contributors. MIT License.

#include "safetensors.h"

#include <cstring>
#include <stdexcept>

namespace bytefast {

namespace {

void writeJsonString(std::string& out, const std::string& s) {
  static const char* hex = "0123456789abcdef";
  out.push_back('"');
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(c);
    } else if (c < 0x20) {
      out += "\\u00";
      out.push_back(hex[c >> 4]);
      out.push_back(hex[c & 15]);
    } else {
      out.push_back(c);
    }
  }
  out.push_back('"');
}

// Parser for the subset of JSON a safetensors header uses: objects, strings,
// integers and arrays of integers.
class HeaderParser {
 public:
  explicit HeaderParser(const std::string& s) : s_(s) {}

  template <typename OnKey>
  void object(OnKey onKey) {
    expect('{');
    if (peek() == '}') {
      i_++;
      return;
    }
    while (true) {
      std::string key = string();
      expect(':');
      onKey(key);
      if (peek() == ',') {
        i_++;
        continue;
      }
      expect('}');
      return;
    }
  }

  std::string string() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (i_ >= s_.size()) {
        fail();
      }
      char e = s_[i_++];
      switch (e) {
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case 'r': out.push_back('\r'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'u': {
          if (i_ + 4 > s_.size()) {
            fail();
          }
          uint32_t cp = std::stoul(s_.substr(i_, 4), nullptr, 16);
          i_ += 4;
          if (cp >= 0xD800 && cp < 0xDC00 && s_.compare(i_, 2, "\\u") == 0) {
            uint32_t lo = std::stoul(s_.substr(i_ + 2, 4), nullptr, 16);
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            i_ += 6;
          }
          appendUtf8(out, cp);
          break;
        }
        default: out.push_back(e);
      }
    }
    expect('"');
    return out;
  }

  int64_t integer() {
    skipSpace();
    size_t start = i_;
    if (i_ < s_.size() && s_[i_] == '-') {
      i_++;
    }
    while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') {
      i_++;
    }
    if (start == i_) {
      fail();
    }
    return std::stoll(s_.substr(start, i_ - start));
  }

  std::vector<int64_t> integers() {
    std::vector<int64_t> out;
    expect('[');
    if (peek() == ']') {
      i_++;
      return out;
    }
    while (true) {
      out.push_back(integer());
      if (peek() == ',') {
        i_++;
        continue;
      }
      expect(']');
      return out;
    }
  }

  void end() {
    skipSpace();
    if (i_ != s_.size()) {
      fail();
    }
  }

 private:
  static void appendUtf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out.push_back(cp);
    } else if (cp < 0x800) {
      out.push_back(0xC0 | (cp >> 6));
      out.push_back(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out.push_back(0xE0 | (cp >> 12));
      out.push_back(0x80 | ((cp >> 6) & 0x3F));
      out.push_back(0x80 | (cp & 0x3F));
    } else {
      out.push_back(0xF0 | (cp >> 18));
      out.push_back(0x80 | ((cp >> 12) & 0x3F));
      out.push_back(0x80 | ((cp >> 6) & 0x3F));
      out.push_back(0x80 | (cp & 0x3F));
    }
  }
  void skipSpace() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' ||
                              s_[i_] == '\r' || s_[i_] == '\t')) {
      i_++;
    }
  }
  char peek() {
    skipSpace();
    return i_ < s_.size() ? s_[i_] : '\0';
  }
  void expect(char c) {
    if (peek() != c) {
      fail();
    }
    i_++;
  }
  [[noreturn]] void fail() const {
    throw std::invalid_argument(
        "Invalid safetensors header near byte " + std::to_string(i_));
  }

  const std::string& s_;
  size_t i_ = 0;
};

size_t dtypeSize(const std::string& dtype) {
  if (dtype == "F32" || dtype == "I32") {
    return 4;
  }
  if (dtype == "I64" || dtype == "U64") {
    return 8;
  }
  if (dtype == "U8" || dtype == "I8") {
    return 1;
  }
  throw std::invalid_argument("Unsupported safetensors dtype " + dtype);
}

} // namespace

void SafeTensorsWriter::metadata(
    const std::string& key,
    const std::string& value) {
  metadata_[key] = value;
}

void SafeTensorsWriter::tensor(
    const std::string& name,
    const std::string& dtype,
    const std::vector<int64_t>& shape,
    const void* data,
    size_t bytes) {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d;
  }
  if (size_t(count) * dtypeSize(dtype) != bytes) {
    throw std::invalid_argument("Tensor " + name + " has the wrong size");
  }
  tensors_.push_back({name, dtype, shape, data, bytes});
}

void SafeTensorsWriter::tensor(
    const std::string& name,
    const std::string& dtype,
    const std::vector<int64_t>& shape,
    std::string data) {
  owned_.push_back(std::make_unique<std::string>(std::move(data)));
  tensor(name, dtype, shape, owned_.back()->data(), owned_.back()->size());
}

void SafeTensorsWriter::write(const std::string& path) const {
  std::string header = "{\"__metadata__\":{";
  bool first = true;
  for (const auto& kv : metadata_) {
    if (!first) {
      header.push_back(',');
    }
    first = false;
    writeJsonString(header, kv.first);
    header.push_back(':');
    writeJsonString(header, kv.second);
  }
  header.push_back('}');
  uint64_t offset = 0;
  for (const auto& t : tensors_) {
    header.push_back(',');
    writeJsonString(header, t.name);
    header += ":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); i++) {
      header += (i ? "," : "") + std::to_string(t.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
        std::to_string(offset + t.bytes) + "]}";
    offset += t.bytes;
  }
  header.push_back('}');
  // Pad so that the tensor data starts at a multiple of 8 bytes.
  while ((8 + header.size()) % 8 != 0) {
    header.push_back(' ');
  }
  std::ofstream out(path, std::ofstream::binary);
  if (!out.is_open()) {
    throw std::invalid_argument(path + " cannot be opened for saving!");
  }
  uint64_t n = header.size();
  unsigned char len[8];
  for (int i = 0; i < 8; i++) {
    len[i] = (n >> (8 * i)) & 0xFF;
  }
  out.write(reinterpret_cast<const char*>(len), 8);
  out.write(header.data(), header.size());
  for (const auto& t : tensors_) {
    out.write(static_cast<const char*>(t.data), t.bytes);
  }
  if (!out) {
    throw std::runtime_error("Error while writing " + path);
  }
}

bool SafeTensorsReader::isSafeTensors(const std::string& path) {
  std::ifstream in(path, std::ifstream::binary);
  unsigned char head[9];
  if (!in.read(reinterpret_cast<char*>(head), 9)) {
    return false;
  }
  uint64_t n = 0;
  for (int i = 0; i < 8; i++) {
    n |= uint64_t(head[i]) << (8 * i);
  }
  return n > 1 && n < (uint64_t(1) << 30) && head[8] == '{';
}

SafeTensorsReader::SafeTensorsReader(const std::string& path)
    : path_(path), in_(path, std::ifstream::binary) {
  if (!in_.is_open()) {
    throw std::invalid_argument(path + " cannot be opened for loading!");
  }
  unsigned char len[8];
  in_.read(reinterpret_cast<char*>(len), 8);
  uint64_t n = 0;
  for (int i = 0; i < 8; i++) {
    n |= uint64_t(len[i]) << (8 * i);
  }
  if (!in_ || n > (uint64_t(1) << 30)) {
    throw std::invalid_argument(path + " is not a safetensors file");
  }
  std::string header(n, '\0');
  if (!in_.read(&header[0], n)) {
    throw std::invalid_argument(path + " has a truncated header");
  }
  dataStart_ = 8 + n;
  HeaderParser p(header);
  p.object([&](const std::string& name) {
    if (name == "__metadata__") {
      p.object([&](const std::string& key) { metadata_[key] = p.string(); });
      return;
    }
    Entry e;
    p.object([&](const std::string& field) {
      if (field == "dtype") {
        e.dtype = p.string();
      } else if (field == "shape") {
        e.shape = p.integers();
      } else if (field == "data_offsets") {
        auto offsets = p.integers();
        if (offsets.size() != 2 || offsets[0] < 0 || offsets[1] < offsets[0]) {
          throw std::invalid_argument(path_ + ": bad offsets for " + name);
        }
        e.begin = offsets[0];
        e.end = offsets[1];
      } else {
        throw std::invalid_argument(path_ + ": unknown field " + field);
      }
    });
    tensors_[name] = e;
  });
  p.end();
}

bool SafeTensorsReader::has(const std::string& name) const {
  return tensors_.count(name) != 0;
}

bool SafeTensorsReader::hasMetadata(const std::string& key) const {
  return metadata_.count(key) != 0;
}

const std::string& SafeTensorsReader::metadata(const std::string& key) const {
  auto it = metadata_.find(key);
  if (it == metadata_.end()) {
    throw std::invalid_argument(path_ + ": missing metadata " + key);
  }
  return it->second;
}

int64_t SafeTensorsReader::metadataInt(const std::string& key) const {
  return std::stoll(metadata(key));
}

const SafeTensorsReader::Entry& SafeTensorsReader::entry(
    const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::invalid_argument(path_ + ": missing tensor " + name);
  }
  return it->second;
}

const std::vector<int64_t>& SafeTensorsReader::shape(
    const std::string& name) const {
  return entry(name).shape;
}

size_t SafeTensorsReader::bytes(const std::string& name) const {
  const Entry& e = entry(name);
  return e.end - e.begin;
}

void SafeTensorsReader::read(
    const std::string& name,
    const std::string& dtype,
    void* dst,
    size_t bytes) {
  const Entry& e = entry(name);
  if (e.dtype != dtype || e.end - e.begin != bytes) {
    throw std::invalid_argument(
        path_ + ": tensor " + name + " is not " + dtype + " of " +
        std::to_string(bytes) + " bytes");
  }
  in_.seekg(dataStart_ + e.begin);
  if (bytes > 0 && !in_.read(static_cast<char*>(dst), bytes)) {
    throw std::invalid_argument(path_ + ": truncated tensor " + name);
  }
}

} // namespace bytefast
