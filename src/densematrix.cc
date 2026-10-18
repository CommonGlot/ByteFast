/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "densematrix.h"
#include "safetensors.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include "utils.h"
#include "vector.h"

#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
#include <immintrin.h>
#endif

namespace bytefast {

DenseMatrix::DenseMatrix() : DenseMatrix(0, 0) {}

DenseMatrix::DenseMatrix(int64_t m, int64_t n) : Matrix(m, n), data_(m * n) {}

DenseMatrix::DenseMatrix(DenseMatrix&& other) noexcept
    : Matrix(other.m_, other.n_), data_(std::move(other.data_)) {}

void DenseMatrix::zero() {
  std::fill(data_.begin(), data_.end(), 0.0);
}

void DenseMatrix::uniformThread(real a, int block, int32_t seed) {
  std::minstd_rand rng(block + seed);
  std::uniform_real_distribution<> uniform(-a, a);
  int64_t blockSize = (m_ * n_) / 10;
  // Block 10 takes everything after the tenth block.
  const int64_t end = block >= 10 ? m_ * n_ : blockSize * (block + 1);
  for (int64_t i = blockSize * block; i < (m_ * n_) && i < end; i++) {
    data_[i] = uniform(rng);
  }
}

void DenseMatrix::uniform(real a, unsigned int thread, int32_t seed) {
  // The matrix is filled in blocks of a tenth of its size, each with its own
  // seed, so the values do not depend on the number of threads; block 10
  // holds the elements after the tenth block. Upstream fastText filled one
  // block per thread, which left most of the matrix uninitialized with fewer
  // than ten threads; with eleven or more threads the result is unchanged.
  constexpr int kBlocks = 11;
  const int workers = static_cast<int>(
      std::min<unsigned int>(std::max(thread, 1u), kBlocks));
  auto fill = [=](int worker) {
    for (int block = worker; block < kBlocks; block += workers) {
      uniformThread(a, block, seed);
    }
  };
  if (workers > 1) {
    std::vector<std::thread> threads;
    for (int i = 0; i < workers; i++) {
      threads.push_back(std::thread(fill, i));
    }
    for (auto& t : threads) {
      t.join();
    }
  } else {
    // webassembly can't instantiate `std::thread`
    fill(0);
  }
}

void DenseMatrix::multiplyRow(const Vector& nums, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= nums.size());
  for (auto i = ib; i < ie; i++) {
    real n = nums[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) *= n;
      }
    }
  }
}

void DenseMatrix::divideRow(const Vector& denoms, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= denoms.size());
  for (auto i = ib; i < ie; i++) {
    real n = denoms[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) /= n;
      }
    }
  }
}

real DenseMatrix::l2NormRow(int64_t i) const {
  auto norm = 0.0;
  for (auto j = 0; j < n_; j++) {
    norm += at(i, j) * at(i, j);
  }
  if (std::isnan(norm)) {
    throw EncounteredNaNError();
  }
  return std::sqrt(norm);
}

void DenseMatrix::l2NormRow(Vector& norms) const {
  assert(norms.size() == m_);
  for (auto i = 0; i < m_; i++) {
    norms[i] = l2NormRow(i);
  }
}

namespace {

// Independent partial sums let the compiler vectorize the loop (SSE/AVX
// on x86, NEON on ARM); a single accumulator forces a serial chain.
constexpr int64_t kLanes = 8;

real dot(const real* row, const real* v, int64_t n) {
  real partial[kLanes] = {};
  int64_t j = 0;
  for (; j + kLanes <= n; j += kLanes) {
    for (int64_t l = 0; l < kLanes; l++) {
      partial[l] += row[j + l] * v[j + l];
    }
  }
  real d = 0.0;
  for (int64_t l = 0; l < kLanes; l++) {
    d += partial[l];
  }
  for (; j < n; j++) {
    d += row[j] * v[j];
  }
  return d;
}

} // namespace

real DenseMatrix::dotRow(const Vector& vec, int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  real d = dot(data_.data() + i * n_, vec.data(), n_);
  if (std::isnan(d)) {
    throw EncounteredNaNError();
  }
  return d;
}

void DenseMatrix::multiplyVector(Vector& out, const Vector& vec) const {
  assert(out.size() == m_);
  assert(vec.size() == n_);
  // Eight rows per pass share each load of vec, avoid a call per row and give
  // the FMA units independent accumulators. Each row keeps dot()'s lanes and
  // summation order, so the results are the same as dotRow's.
  constexpr int64_t kRows = 8;
  const real* v = vec.data();
  real* o = out.data();
  const int64_t nv = n_ - n_ % kLanes;
  int64_t i = 0;
  for (; i + kRows <= m_; i += kRows) {
    const real* row = data_.data() + i * n_;
    real partial[kRows][kLanes] = {};
    for (int64_t j = 0; j < nv; j += kLanes) {
      for (int64_t r = 0; r < kRows; r++) {
        for (int64_t l = 0; l < kLanes; l++) {
          partial[r][l] += row[r * n_ + j + l] * v[j + l];
        }
      }
    }
    for (int64_t r = 0; r < kRows; r++) {
      real d = 0.0;
      for (int64_t l = 0; l < kLanes; l++) {
        d += partial[r][l];
      }
      for (int64_t j = nv; j < n_; j++) {
        d += row[r * n_ + j] * v[j];
      }
      o[i + r] = d;
    }
  }
  for (; i < m_; i++) {
    o[i] = dot(data_.data() + i * n_, v, n_);
  }
  for (int64_t k = 0; k < m_; k++) {
    if (std::isnan(o[k])) {
      throw EncounteredNaNError();
    }
  }
}

void DenseMatrix::addVectorToRow(const Vector& vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] += a * vec[j];
  }
}

void DenseMatrix::addRowToVector(Vector& x, int32_t i) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  for (int64_t j = 0; j < n_; j++) {
    x[j] += at(i, j);
  }
}

void DenseMatrix::addRowToVector(Vector& x, int32_t i, real a) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  for (int64_t j = 0; j < n_; j++) {
    x[j] += a * at(i, j);
  }
}

/* Abstract over AVX512F, AVX, and SSE intrinsics, using the one available on this machine. */
#if defined(__AVX512F__)
using Register = __m512;
inline Register Add(Register first, Register second) { return _mm512_add_ps(first, second); }
inline Register Set1(float to) { return _mm512_set1_ps(to); }
inline Register Multiply(Register first, Register second) { return _mm512_mul_ps(first, second); }
#elif defined(__AVX__)
using Register = __m256;
inline Register Add(Register first, Register second) { return _mm256_add_ps(first, second); }
inline Register Set1(float to) { return _mm256_set1_ps(to); }
inline Register Multiply(Register first, Register second) { return _mm256_mul_ps(first, second); }
#elif defined(__SSE__)
using Register = __m128;
inline Register Add(Register first, Register second) { return _mm_add_ps(first, second); }
inline Register Set1(float to) { return _mm_set1_ps(to); }
inline Register Multiply(Register first, Register second) { return _mm_mul_ps(first, second); }
#endif

/* Faster routine for averaging rows of a matrix on x86.
 * The idea here is to keep the accumulators in registers if possible. */
#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
template <unsigned Cols> void averageRowsFast(Vector& x, const std::vector<int32_t>& rows, const DenseMatrix &matrix) {
  // Columns must be a multiple of how many floats fit in a register.
  static_assert(Cols % (sizeof(Register) / 4) == 0);
  constexpr unsigned RegisterCount = Cols / (sizeof(Register) / 4);
  // These should be aligned by aligned.h
  assert(reinterpret_cast<uintptr_t>(x.data()) % sizeof(Register) == 0);
  assert(reinterpret_cast<uintptr_t>(matrix.data()) % sizeof(Register) == 0);

  // Guard against empty list of rows with default NaN behavior.
  if (rows.empty()) {
    x.zero();
    x.mul(1.0 / rows.size());
    return;
  }

  // Copy the first row to accumulation registers.
  Register accum[RegisterCount];
  auto row = rows.cbegin();
  const Register *base = reinterpret_cast<const Register*>(matrix.data() + matrix.cols() * *row);
  for (unsigned i = 0; i < RegisterCount; ++i) {
    accum[i] = base[i];
  }
  // Add the rows after the first.
  for (++row; row != rows.cend(); ++row) {
    base = reinterpret_cast<const Register*>(matrix.data() + matrix.cols() * *row);
    for (unsigned i = 0; i < RegisterCount; ++i) {
      accum[i] = Add(accum[i], base[i]);
    }
  }
  // Multiply by (1.0 / rows.size()) and write to x.
  Register mul = Set1(1.0 / rows.size());
  for (unsigned i = 0; i < RegisterCount; ++i) {
    reinterpret_cast<Register*>(x.data())[i] = Multiply(accum[i], mul);
  }
}
#endif

#if !(defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__))
/* Portable version of averageRowsFast: a fixed-size local accumulator that
 * the compiler can keep in vector registers (e.g. NEON on ARM). */
template <unsigned Cols> void averageRowsFast(Vector& x, const std::vector<int32_t>& rows, const DenseMatrix &matrix) {
  if (rows.empty()) {
    x.zero();
    x.mul(1.0 / rows.size());
    return;
  }
  real accum[Cols];
  const real* base = matrix.data() + matrix.cols() * rows[0];
  for (unsigned i = 0; i < Cols; ++i) {
    accum[i] = base[i];
  }
  for (size_t r = 1; r < rows.size(); ++r) {
    base = matrix.data() + matrix.cols() * rows[r];
    for (unsigned i = 0; i < Cols; ++i) {
      accum[i] += base[i];
    }
  }
  const real mul = 1.0 / rows.size();
  for (unsigned i = 0; i < Cols; ++i) {
    x[i] = accum[i] * mul;
  }
}
#endif

void DenseMatrix::averageRowsToVector(Vector& x, const std::vector<int32_t>& rows) const {
  switch (cols()) {
    case 512:
      // Maximum number that can fit all in registers on AVX512F.
      averageRowsFast<512>(x, rows, *this);
      return;
    case 256:
      averageRowsFast<256>(x, rows, *this);
      return;
    case 64:
      averageRowsFast<64>(x, rows, *this);
      return;
    case 32:
      averageRowsFast<32>(x, rows, *this);
      return;
    case 128:
      averageRowsFast<128>(x, rows, *this);
      return;
    case 16:
      averageRowsFast<16>(x, rows, *this);
      return;
  }
  x.zero();
  for (auto it = rows.cbegin(); it != rows.cend(); ++it) {
    addRowToVector(x, *it);
  }
  x.mul(1.0 / rows.size());
}

void DenseMatrix::save(SafeTensorsWriter& out, const std::string& name) const {
  out.tensor(name, "F32", {m_, n_}, data_.data(), m_ * n_ * sizeof(real));
}

void DenseMatrix::load(SafeTensorsReader& in, const std::string& name) {
  const std::vector<int64_t>& shape = in.shape(name);
  if (shape.size() != 2) {
    throw std::invalid_argument(name + " is not a matrix");
  }
  m_ = shape[0];
  n_ = shape[1];
  data_ = intgemm::AlignedVector<real>(m_ * n_);
  in.read(name, "F32", data_.data(), m_ * n_ * sizeof(real));
}

void DenseMatrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  data_ = intgemm::AlignedVector<real>(m_ * n_);
  in.read((char*)data_.data(), m_ * n_ * sizeof(real));
}

} // namespace bytefast
