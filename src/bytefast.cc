/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bytefast.h"
#include "loss.h"
#include "quantmatrix.h"

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

constexpr int32_t FASTTEXT_VERSION = 12; /* Version 1b */
constexpr int32_t FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;
// Models that use ByteFast text features ("BFCM" on disk). Others are
// written in the fastText format above, so upstream fastText reads them.
constexpr int32_t BYTEFAST_FILEFORMAT_MAGIC_INT32 = 0x4D434642;
constexpr int32_t BYTEFAST_VERSION = 1;

std::shared_ptr<Loss> ByteFast::createLoss(std::shared_ptr<Matrix>& output) {
  loss_name lossName = args_->loss;
  switch (lossName) {
    case loss_name::hs:
      return std::make_shared<HierarchicalSoftmaxLoss>(
          output, getTargetCounts());
    case loss_name::ns:
      return std::make_shared<NegativeSamplingLoss>(
          output, args_->neg, getTargetCounts());
    case loss_name::softmax:
      return std::make_shared<SoftmaxLoss>(output);
    case loss_name::ova:
      return std::make_shared<OneVsAllLoss>(output);
    default:
      throw std::runtime_error("Unknown loss");
  }
}

ByteFast::ByteFast() : quant_(false), trainException_(nullptr) {}

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

bool ByteFast::checkModel(std::istream& in) {
  int32_t magic;
  in.read((char*)&(magic), sizeof(int32_t));
  if (magic == BYTEFAST_FILEFORMAT_MAGIC_INT32) {
    int32_t bytefastVersion;
    in.read((char*)&(bytefastVersion), sizeof(int32_t));
    if (!in || bytefastVersion != BYTEFAST_VERSION) {
      return false;
    }
    version = FASTTEXT_VERSION;
    textFeatures_ = true;
    return true;
  }
  if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
    return false;
  }
  in.read((char*)&(version), sizeof(int32_t));
  if (version > FASTTEXT_VERSION) {
    return false;
  }
  textFeatures_ = false;
  return true;
}

void ByteFast::signModel(std::ostream& out) {
  const bool textFeatures = args_->usesTextFeatures();
  const int32_t magic = textFeatures ? BYTEFAST_FILEFORMAT_MAGIC_INT32
                                     : FASTTEXT_FILEFORMAT_MAGIC_INT32;
  const int32_t version = textFeatures ? BYTEFAST_VERSION : FASTTEXT_VERSION;
  out.write((char*)&(magic), sizeof(int32_t));
  out.write((char*)&(version), sizeof(int32_t));
}

void ByteFast::saveModel(const std::string& filename) {
  std::ofstream ofs(filename, std::ofstream::binary);
  if (!ofs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for saving!");
  }
  if (!input_ || !output_) {
    throw std::runtime_error("Model never trained");
  }
  signModel(ofs);
  args_->save(ofs);
  if (args_->usesTextFeatures()) {
    args_->saveTextFeatures(ofs);
  }
  dict_->save(ofs);

  ofs.write((char*)&(quant_), sizeof(bool));
  input_->save(ofs);

  ofs.write((char*)&(args_->qout), sizeof(bool));
  output_->save(ofs);

  ofs.close();
}

void ByteFast::loadModel(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  if (!checkModel(ifs)) {
    throw std::invalid_argument(filename + " has wrong file format!");
  }
  loadModel(ifs);
  ifs.close();
}

std::vector<int64_t> ByteFast::getTargetCounts() const {
  return dict_->getCounts(entry_type::label);
}

void ByteFast::buildModel() {
  auto loss = createLoss(output_);
  model_ = std::make_shared<Model>(input_, output_, loss, true);
}

void ByteFast::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();
  input_ = std::make_shared<DenseMatrix>();
  output_ = std::make_shared<DenseMatrix>();
  args_->load(in);
  if (textFeatures_) {
    args_->loadTextFeatures(in);
  }
  dict_ = std::make_shared<Dictionary>(args_, in);

  bool quant_input;
  in.read((char*)&quant_input, sizeof(bool));
  if (quant_input) {
    quant_ = true;
    input_ = std::make_shared<QuantMatrix>();
  }
  input_->load(in);

  if (!quant_input && dict_->isPruned()) {
    throw std::invalid_argument(
        "Invalid model file.\n"
        "Please download the updated model from www.fasttext.cc.\n"
        "See issue #332 on Github for more information.\n");
  }

  in.read((char*)&args_->qout, sizeof(bool));
  if (quant_ && args_->qout) {
    output_ = std::make_shared<QuantMatrix>();
  }
  output_->load(in);

  buildModel();
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

std::vector<int32_t> ByteFast::selectEmbeddings(int32_t cutoff) const {
  std::shared_ptr<DenseMatrix> input =
      std::dynamic_pointer_cast<DenseMatrix>(input_);
  Vector norms(input->size(0));
  input->l2NormRow(norms);
  std::vector<int32_t> idx(input->size(0), 0);
  std::iota(idx.begin(), idx.end(), 0);
  auto eosid = dict_->getId(Dictionary::EOS);
  std::sort(idx.begin(), idx.end(), [&norms, eosid](size_t i1, size_t i2) {
    if (i1 == eosid && i2 == eosid) { // satisfy strict weak ordering
      return false;
    }
    return eosid == i1 || (eosid != i2 && norms[i1] > norms[i2]);
  });
  idx.erase(idx.begin() + cutoff, idx.end());
  return idx;
}

void ByteFast::quantize(const Args& qargs, const TrainCallback& callback) {
  args_->input = qargs.input;
  args_->qout = qargs.qout;
  args_->output = qargs.output;
  std::shared_ptr<DenseMatrix> input =
      std::dynamic_pointer_cast<DenseMatrix>(input_);
  std::shared_ptr<DenseMatrix> output =
      std::dynamic_pointer_cast<DenseMatrix>(output_);
  if (qargs.cutoff > 0 && qargs.cutoff < input->size(0)) {
    auto idx = selectEmbeddings(qargs.cutoff);
    dict_->prune(idx);
    std::shared_ptr<DenseMatrix> ninput =
        std::make_shared<DenseMatrix>(idx.size(), args_->dim);
    for (auto i = 0; i < idx.size(); i++) {
      for (auto j = 0; j < args_->dim; j++) {
        ninput->at(i, j) = input->at(idx[i], j);
      }
    }
    input = ninput;
    if (qargs.retrain) {
      args_->epoch = qargs.epoch;
      args_->lr = qargs.lr;
      args_->thread = qargs.thread;
      args_->verbose = qargs.verbose;
      auto loss = createLoss(output_);
      model_ = std::make_shared<Model>(input, output, loss, true);
      trainTokens_ = dict_->ntokens();
      startThreads(callback);
    }
  }
  input_ = std::make_shared<QuantMatrix>(
      std::move(*(input.get())), qargs.dsub, qargs.qnorm);

  if (args_->qout) {
    output_ = std::make_shared<QuantMatrix>(
        std::move(*(output.get())), 2, qargs.qnorm);
  }
  quant_ = true;
  auto loss = createLoss(output_);
  model_ = std::make_shared<Model>(input_, output_, loss, true);
}

void ByteFast::supervised(
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line,
    const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) {
    return;
  }
  if (args_->loss == loss_name::ova) {
    model_->update(line, labels, Model::kAllLabelsAsTarget, lr, state);
  } else {
    std::uniform_int_distribution<> uniform(0, labels.size() - 1);
    int32_t i = uniform(state.rng);
    model_->update(line, labels, i, lr, state);
  }
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
  if (!args.inputModel.empty()) {
    continueTraining(args, callback);
    return;
  }
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
  quant_ = false;
  buildModel();
  startThreads(callback);
}

// Continue training a saved supervised model on args.input. The model's
// dictionary, features and architecture are kept; the training schedule
// (input, lr, epoch, threads, seed, cropping) comes from args. Lines whose
// labels the model does not know are skipped.
void ByteFast::continueTraining(const Args& args, const TrainCallback& callback) {
  loadModel(args.inputModel);
  if (quant_) {
    throw std::invalid_argument("-inputModel cannot be a quantized model");
  }
  // The dictionary shares args_, so update it in place.
  args_->input = args.input;
  args_->output = args.output;
  args_->lr = args.lr;
  args_->lrUpdateRate = args.lrUpdateRate;
  args_->epoch = args.epoch;
  args_->thread = args.thread;
  args_->seed = args.seed;
  args_->verbose = args.verbose;
  args_->inputModel = args.inputModel;
  args_->cropProb = args.cropProb;
  args_->cropMin = args.cropMin;
  args_->cropMax = args.cropMax;
  args_->cropUnit = args.cropUnit;
  args_->normProb = args.normProb;
  args_->saveEpochs = args.saveEpochs;
  args_->validation = args.validation;
  if (args_->input == "-") {
    throw std::invalid_argument("Cannot use stdin for training!");
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    throw std::invalid_argument(
        args_->input + " cannot be opened for training!");
  }
  trainTokens_ = dict_->countTokens(ifs);
  ifs.close();
  if (trainTokens_ == 0) {
    throw std::invalid_argument(args_->input + " is empty");
  }
  if (args_->verbose > 0) {
    std::cerr << "Continuing " << args_->inputModel << " on "
              << trainTokens_ / 1000000 << "M tokens" << std::endl;
  }
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
    saveModel(args_->output + ".ep" + std::to_string(epoch) + ".bin");
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

bool ByteFast::isQuant() const {
  return quant_;
}

} // namespace bytefast
