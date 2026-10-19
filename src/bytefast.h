/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <time.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <tuple>

#include "args.h"
#include "densematrix.h"
#include "dictionary.h"
#include "matrix.h"
#include "meter.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace bytefast {

class ByteFast {
 public:
  using TrainCallback =
      std::function<void(float, float, double, double, int64_t)>;

 protected:
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> dict_;
  std::shared_ptr<Matrix> input_;
  std::shared_ptr<Matrix> output_;
  std::shared_ptr<Model> model_;
  std::atomic<int64_t> tokenCount_{};
  std::atomic<real> loss_{};
  std::chrono::steady_clock::time_point start_;
  // Tokens per epoch of the current training input.
  int64_t trainTokens_ = 0;
  std::exception_ptr trainException_;

  void startThreads(const TrainCallback& callback = {});
  void epochCheckpoint(int32_t epoch);
  void trainThread(int32_t, const TrainCallback& callback);
  void printInfo(real, real, std::ostream&);
  std::shared_ptr<Matrix> createRandomMatrix() const;
  std::shared_ptr<Matrix> createTrainOutputMatrix() const;
  std::vector<int64_t> getTargetCounts() const;
  std::shared_ptr<Loss> createLoss(std::shared_ptr<Matrix>& output);
  void supervised(
      Model::State& state,
      real lr,
      const std::vector<int32_t>& line,
      const std::vector<int32_t>& labels);
  bool keepTraining(const int64_t ntokens) const;
  void buildModel();
  std::tuple<int64_t, double, double> progressInfo(real progress);

 public:
  ByteFast();

  int32_t getLabelId(const std::string& label) const;

  const Args getArgs() const;

  std::shared_ptr<const Dictionary> getDictionary() const;

  void saveModel(const std::string& filename);

  void loadModel(const std::string& filename);

  std::tuple<int64_t, double, double>
  test(std::istream& in, int32_t k, real threshold = 0.0);

  void test(std::istream& in, int32_t k, real threshold, Meter& meter) const;

  void predict(
      int32_t k,
      const std::vector<int32_t>& words,
      Predictions& predictions,
      real threshold = 0.0) const;

  bool predictLine(
      std::istream& in,
      std::vector<std::pair<real, std::string>>& predictions,
      int32_t k,
      real threshold) const;

  void train(const Args& args, const TrainCallback& callback = {});

  void abort();

  int getDimension() const;

  class AbortError : public std::runtime_error {
   public:
    AbortError() : std::runtime_error("Aborted.") {}
  };
};
} // namespace bytefast
