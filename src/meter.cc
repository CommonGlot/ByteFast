/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meter.h"
#include "utils.h"

#include <iomanip>
#include <limits>

namespace bytefast {

void Meter::log(
    const std::vector<int32_t>& labels,
    const Predictions& predictions) {
  nexamples_++;
  gold_ += labels.size();
  predicted_ += predictions.size();
  for (const auto& prediction : predictions) {
    if (utils::contains(labels, prediction.second)) {
      predictedGold_++;
    }
  }
}

double Meter::precision() const {
  if (predicted_ == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return predictedGold_ / double(predicted_);
}

double Meter::recall() const {
  if (gold_ == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return predictedGold_ / double(gold_);
}

void Meter::writeGeneralMetrics(std::ostream& out, int32_t k) const {
  out << "N"
      << "\t" << nexamples_ << std::endl;
  out << std::setprecision(3);
  out << "P@" << k << "\t" << precision() << std::endl;
  out << "R@" << k << "\t" << recall() << std::endl;
}

} // namespace bytefast
