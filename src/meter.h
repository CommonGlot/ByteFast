/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <ostream>
#include <vector>

#include "utils.h"
#include "real.h"

namespace bytefast {

// Precision and recall at k over a labeled file (the test command).
class Meter {
 public:
  void log(const std::vector<int32_t>& labels, const Predictions& predictions);
  double precision() const;
  double recall() const;
  uint64_t nexamples() const {
    return nexamples_;
  }
  void writeGeneralMetrics(std::ostream& out, int32_t k) const;

 private:
  uint64_t nexamples_ = 0;
  uint64_t gold_ = 0;
  uint64_t predicted_ = 0;
  uint64_t predictedGold_ = 0;
};

} // namespace bytefast
