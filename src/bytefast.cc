/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bytefast.h"
#include "loss.h"
#include "safetensors.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace bytefast {

// Models are safetensors files with this format name and version in their
// metadata.
const char* const kFormat = "bytefast";
const char* const kFormatVersion = "1";

std::shared_ptr<Loss> ByteFast::createLoss(std::shared_ptr<Matrix>& output) {
  loss_name lossName = args_->loss;
  switch (lossName) {
    case loss_name::softmax:
      return std::make_shared<SoftmaxLoss>(output);
    default:
      throw std::runtime_error("Unknown loss");
  }
}

ByteFast::ByteFast() : trainException_(nullptr) {}

std::shared_ptr<const Dictionary> ByteFast::getDictionary() const {
  return dict_;
}

const Args ByteFast::getArgs() const {
  return *args_.get();
}

int32_t ByteFast::getLabelId(const std::string& label) const {
  int32_t labelId = dict_->getId(label);
  if (labelId != -1) {
    labelId -= dict_->nwords();
  }
  return labelId;
}

void ByteFast::saveModel(const std::string& filename) {
  if (!input_ || !output_) {
    throw std::runtime_error("Model never trained");
  }
  SafeTensorsWriter out;
  out.metadata("format", kFormat);
  out.metadata("format_version", kFormatVersion);
  args_->save(out);
  dict_->save(out);
  input_->save(out, "input");
  output_->save(out, "output");
  out.write(filename);
}

void ByteFast::loadModel(const std::string& filename) {
  if (!SafeTensorsReader::isSafeTensors(filename)) {
    throw std::invalid_argument(filename + " is not a safetensors model");
  }
  SafeTensorsReader in(filename);
  if (!in.hasMetadata("format") || in.metadata("format") != kFormat ||
      in.metadata("format_version") != kFormatVersion) {
    throw std::invalid_argument(filename + " is not a ByteFast model");
  }
  args_ = std::make_shared<Args>();
  args_->load(in);
  dict_ = std::make_shared<Dictionary>(args_, in);
  input_ = std::make_shared<DenseMatrix>();
  output_ = std::make_shared<DenseMatrix>();
  input_->load(in, "input");
  output_->load(in, "output");
  buildModel();
}

std::vector<int64_t> ByteFast::getTargetCounts() const {
  return dict_->getCounts(entry_type::label);
}

void ByteFast::buildModel() {
  auto loss = createLoss(output_);
  model_ = std::make_shared<Model>(input_, output_, loss, true);
}

std::tuple<int64_t, double, double> ByteFast::progressInfo(real progress) {
  double t = utils::getDuration(start_, std::chrono::steady_clock::now());
  double lr = args_->lr * (1.0 - progress);
  double wst = 0;

  int64_t eta = 2592000; // Default to one month in seconds (720 * 3600)

  if (progress > 0 && t >= 0) {
    eta = t * (1 - progress) / progress;
    wst = double(tokenCount_) / t / args_->thread;
  }

  return std::tuple<double, double, int64_t>(wst, lr, eta);
}

void ByteFast::printInfo(real progress, real loss, std::ostream& log_stream) {
  double wst;
  double lr;
  int64_t eta;
  std::tie<double, double, int64_t>(wst, lr, eta) = progressInfo(progress);

  log_stream << std::fixed;
  log_stream << "Progress: ";
  log_stream << std::setprecision(1) << std::setw(5) << (progress * 100) << "%";
  log_stream << " words/sec/thread: " << std::setw(7) << int64_t(wst);
  log_stream << " lr: " << std::setw(9) << std::setprecision(6) << lr;
  log_stream << " avg.loss: " << std::setw(9) << std::setprecision(6) << loss;
  log_stream << " ETA: " << utils::ClockPrint(eta);
  log_stream << std::flush;
}

void ByteFast::supervised(
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line,
    const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) {
    return;
  }
  // A line with several labels trains on one of them, chosen at random.
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(state.rng);
  model_->update(line, labels, i, lr, state);
}

std::tuple<int64_t, double, double>
ByteFast::test(std::istream& in, int32_t k, real threshold) {
  Meter meter(false);
  test(in, k, threshold, meter);

  return std::tuple<int64_t, double, double>(
      meter.nexamples(), meter.precision(), meter.recall());
}

void ByteFast::test(std::istream& in, int32_t k, real threshold, Meter& meter)
    const {
  std::vector<int32_t> line;
  std::vector<int32_t> labels;
  Predictions predictions;
  Model::State state(args_->dim, dict_->nlabels(), 0);
  in.clear();
  in.seekg(0, std::ios_base::beg);

  while (in.peek() != EOF) {
    line.clear();
    labels.clear();
    dict_->getLine(in, line, labels);

    if (!labels.empty() && !line.empty()) {
      predictions.clear();
      predict(k, line, predictions, threshold);
      meter.log(labels, predictions);
    }
  }
}

void ByteFast::predict(
    int32_t k,
    const std::vector<int32_t>& words,
    Predictions& predictions,
    real threshold) const {
  if (words.empty()) {
    return;
  }
  Model::State state(args_->dim, dict_->nlabels(), 0);
  model_->predict(words, k, threshold, predictions, state);
}

bool ByteFast::predictLine(
    std::istream& in,
    std::vector<std::pair<real, std::string>>& predictions,
    int32_t k,
    real threshold) const {
  predictions.clear();
  if (in.peek() == EOF) {
    return false;
  }

  std::vector<int32_t> words, labels;
  dict_->getLine(in, words, labels);
  Predictions linePredictions;
  predict(k, words, linePredictions, threshold);
  for (const auto& p : linePredictions) {
    predictions.push_back(
        std::make_pair(std::exp(p.first), dict_->getLabel(p.second)));
  }

  return true;
}

bool ByteFast::keepTraining(const int64_t ntokens) const {
  return tokenCount_ < args_->epoch * ntokens && !trainException_;
}

void ByteFast::trainThread(int32_t threadId, const TrainCallback& callback) {
  std::ifstream ifs(args_->input);
  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

  Model::State state(args_->dim, output_->size(0), threadId + args_->seed);

  const int64_t ntokens = trainTokens_;
  int64_t localTokenCount = 0;
  std::vector<int32_t> line, labels;
  uint64_t callbackCounter = 0;
  try {
    while (keepTraining(ntokens)) {
      real progress = real(tokenCount_) / (args_->epoch * ntokens);
      if (callback && ((callbackCounter++ % 64) == 0)) {
        double wst;
        double lr;
        int64_t eta;
        std::tie<double, double, int64_t>(wst, lr, eta) =
            progressInfo(progress);
        callback(progress, loss_, wst, lr, eta);
      }
      real lr = args_->lr * (1.0 - progress);
      localTokenCount += dict_->getLine(ifs, line, labels, state.rng);
      supervised(state, lr, line, labels);
      if (localTokenCount > args_->lrUpdateRate) {
        tokenCount_ += localTokenCount;
        localTokenCount = 0;
        if (threadId == 0 && args_->verbose > 1) {
          loss_ = state.getLoss();
        }
      }
    }
  } catch (DenseMatrix::EncounteredNaNError&) {
    trainException_ = std::current_exception();
  }
  if (threadId == 0)
    loss_ = state.getLoss();
  ifs.close();
}

std::shared_ptr<Matrix> ByteFast::createRandomMatrix() const {
  std::shared_ptr<DenseMatrix> input = std::make_shared<DenseMatrix>(
      dict_->nwords() + args_->bucket, args_->dim);
  input->uniform(1.0 / args_->dim, args_->thread, args_->seed);

  return input;
}

std::shared_ptr<Matrix> ByteFast::createTrainOutputMatrix() const {
  std::shared_ptr<DenseMatrix> output =
      std::make_shared<DenseMatrix>(dict_->nlabels(), args_->dim);
  output->zero();

  return output;
}

void ByteFast::train(const Args& args, const TrainCallback& callback) {
  args.validate();
  args_ = std::make_shared<Args>(args);
  dict_ = std::make_shared<Dictionary>(args_);
  if (args_->input == "-") {
    // manage expectations
    throw std::invalid_argument("Cannot use stdin for training!");
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    throw std::invalid_argument(
        args_->input + " cannot be opened for training!");
  }
  dict_->readFromFile(ifs);
  ifs.close();
  trainTokens_ = dict_->ntokens();

  input_ = createRandomMatrix();
  output_ = createTrainOutputMatrix();
  buildModel();
  startThreads(callback);
}

void ByteFast::abort() {
  try {
    throw AbortError();
  } catch (AbortError&) {
    trainException_ = std::current_exception();
  }
}

void ByteFast::startThreads(const TrainCallback& callback) {
  start_ = std::chrono::steady_clock::now();
  tokenCount_ = 0;
  loss_ = -1;
  trainException_ = nullptr;
  std::vector<std::thread> threads;
  if (args_->thread > 1) {
    for (int32_t i = 0; i < args_->thread; i++) {
      threads.push_back(std::thread([=]() { trainThread(i, callback); }));
    }
  } else {
    // webassembly can't instantiate `std::thread`
    trainThread(0, callback);
  }
  const int64_t ntokens = trainTokens_;
  const bool checkpoints = args_->saveEpochs || !args_->validation.empty();
  int32_t nextEpoch = 1;
  // Same condition as trainThread
  while (keepTraining(ntokens)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (loss_ >= 0 && args_->verbose > 1) {
      real progress = real(tokenCount_) / (args_->epoch * ntokens);
      std::cerr << "\r";
      printInfo(progress, loss_, std::cerr);
    }
    // The workers keep training while a checkpoint is written or scored,
    // as in Hogwild!; the learning-rate schedule is not affected.
    if (checkpoints && nextEpoch < args_->epoch &&
        tokenCount_ >= int64_t(nextEpoch) * ntokens) {
      epochCheckpoint(nextEpoch++);
    }
  }
  for (int32_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  if (checkpoints && nextEpoch < args_->epoch && !trainException_) {
    std::cerr << "\nno checkpoint for epochs " << nextEpoch << "-" << args_->epoch - 1
              << ": training finished before they were seen" << std::endl;
  }
  if (trainException_) {
    std::exception_ptr exception = trainException_;
    trainException_ = nullptr;
    std::rethrow_exception(exception);
  }
  if (args_->verbose > 0) {
    std::cerr << "\r";
    printInfo(1.0, loss_, std::cerr);
    std::cerr << std::endl;
  }
  if (!args_->validation.empty()) {
    epochCheckpoint(args_->epoch);
  }
}

// After epoch `epoch` (of args_->epoch): save OUTPUT.epN.bin with
// -saveEpochs (not after the last epoch, whose model is saved as usual) and
// print precision@1 on -validation.
void ByteFast::epochCheckpoint(int32_t epoch) {
  if (args_->saveEpochs && epoch < args_->epoch) {
    saveModel(
        args_->output + ".ep" + std::to_string(epoch) + ".safetensors");
  }
  if (!args_->validation.empty()) {
    std::ifstream ifs(args_->validation);
    if (!ifs.is_open()) {
      throw std::invalid_argument(args_->validation + " cannot be opened for validation!");
    }
    Meter meter(false);
    test(ifs, 1, 0.0, meter);
    std::cerr << "\nepoch " << epoch << "/" << args_->epoch << " validation: N "
              << meter.nexamples() << "  P@1 " << std::fixed << std::setprecision(4)
              << meter.precision() << std::endl;
  }
}

int ByteFast::getDimension() const {
  return args_->dim;
}

} // namespace bytefast
