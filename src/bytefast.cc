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
#include <fstream>
#include <cmath>
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
// Tokens a thread processes between updates of the shared progress counter.
const int64_t kLrUpdateRate = 100;

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
  save(out);
  out.write(filename);
}

// Experts are stored after the model under the names expertN.*.
void ByteFast::save(SafeTensorsWriter& out) const {
  args_->save(out);
  dict_->save(out);
  lexicon_.save(out);
  input_->save(out, "input");
  output_->save(out, "output");
  if (!experts_.empty()) {
    out.metadata("experts", std::to_string(experts_.size()));
  }
  for (size_t i = 0; i < experts_.size(); i++) {
    const std::string prefix = "expert" + std::to_string(i) + ".";
    out.metadata(prefix + "gate", std::to_string(experts_[i].gate));
    out.setPrefix(prefix);
    experts_[i].model->save(out);
    out.setPrefix("");
  }
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
  load(in);
}

void ByteFast::load(SafeTensorsReader& in) {
  args_ = std::make_shared<Args>();
  args_->load(in);
  dict_ = std::make_shared<Dictionary>(args_, in);
  lexicon_.load(in);
  input_ = std::make_shared<DenseMatrix>();
  output_ = std::make_shared<DenseMatrix>();
  input_->load(in, "input");
  output_->load(in, "output");
  buildModel();
  experts_.clear();
  expertOf_.assign(dict_->nlabels(), -1);
  const int64_t n = in.hasMetadata("experts") ? in.metadataInt("experts") : 0;
  for (int64_t i = 0; i < n; i++) {
    const std::string prefix = "expert" + std::to_string(i) + ".";
    const real gate = std::stof(in.metadata(prefix + "gate"));
    auto expert = std::make_shared<ByteFast>();
    in.setPrefix(prefix);
    expert->load(in);
    in.setPrefix("");
    attachExpert(expert, gate);
  }
}

void ByteFast::addExpert(const std::string& expertModel, real gate) {
  if (!(gate > 0 && gate <= 1)) {
    throw std::invalid_argument("The expert gate must be in (0, 1]");
  }
  auto expert = std::make_shared<ByteFast>();
  expert->loadModel(expertModel);
  if (expert->numExperts() > 0) {
    throw std::invalid_argument(expertModel + " has experts of its own");
  }
  if (expertOf_.size() != size_t(dict_->nlabels())) {
    expertOf_.assign(dict_->nlabels(), -1);
  }
  attachExpert(expert, gate);
}

void ByteFast::attachExpert(std::shared_ptr<ByteFast> expert, real gate) {
  Expert e{expert, gate, {}};
  const int32_t index = experts_.size();
  int32_t known = 0;
  for (int32_t i = 0; i < expert->dict_->nlabels(); i++) {
    const int32_t id = getLabelId(expert->dict_->getLabel(i));
    e.toMain.push_back(id);
    if (id < 0) {
      continue;
    }
    if (expertOf_[id] >= 0) {
      throw std::invalid_argument(
          expert->dict_->getLabel(i) + " already belongs to an expert");
    }
    expertOf_[id] = index;
    known++;
  }
  if (known == 0) {
    throw std::invalid_argument("The expert has none of this model's labels");
  }
  experts_.push_back(e);
}

int32_t ByteFast::numExperts() const {
  return experts_.size();
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
  // A line with several labels trains on one of them, chosen at random. A
  // label of weight w trains floor(w) times, plus once more with
  // probability w - floor(w).
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(state.rng);
  int32_t repeats = 1;
  if (!labelWeights_.empty()) {
    const real w = labelWeights_[labels[i]];
    repeats = int32_t(w);
    if (std::uniform_real_distribution<real>(0, 1)(state.rng) < w - repeats) {
      repeats++;
    }
  }
  for (int32_t r = 0; r < repeats; r++) {
    model_->update(line, labels, i, lr, state);
  }
}

// -labelWeights: LABEL<TAB>WEIGHT lines; the label prefix is optional and
// labels that are not in the training data are skipped.
void ByteFast::readLabelWeights() {
  labelWeights_.clear();
  if (args_->labelWeights.empty()) {
    return;
  }
  std::ifstream ifs(args_->labelWeights);
  if (!ifs.is_open()) {
    throw std::invalid_argument(args_->labelWeights + " cannot be opened!");
  }
  labelWeights_.assign(dict_->nlabels(), 1.0);
  std::string line;
  int32_t n = 0;
  while (std::getline(ifs, line)) {
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string label = line.substr(0, tab);
    int32_t id = getLabelId(label);
    if (id < 0) {
      id = getLabelId(args_->label + label);
    }
    const real w = std::stof(line.substr(tab + 1));
    if (!(w >= 0)) {
      throw std::invalid_argument("Label weights must be >= 0: " + line);
    }
    if (id >= 0) {
      labelWeights_[id] = w;
      n++;
    }
  }
  if (args_->verbose > 0) {
    std::cerr << "Label weights: " << n << " labels" << std::endl;
  }
}

std::tuple<int64_t, double, double>
ByteFast::test(std::istream& in, int32_t k, real threshold) {
  Meter meter;
  test(in, k, threshold, meter);

  return std::tuple<int64_t, double, double>(
      meter.nexamples(), meter.precision(), meter.recall());
}

void ByteFast::test(std::istream& in, int32_t k, real threshold, Meter& meter)
    const {
  std::vector<std::string> tokens;
  std::vector<int32_t> labels;
  Predictions predictions;
  in.clear();
  in.seekg(0, std::ios_base::beg);

  while (in.peek() != EOF) {
    dict_->readLine(in, tokens, labels);
    if (!labels.empty()) {
      predictions.clear();
      predict(k, tokens, predictions, threshold);
      if (!predictions.empty()) {
        meter.log(labels, predictions);
      }
    }
  }
}

void ByteFast::predict(
    int32_t k,
    const std::vector<std::string>& tokens,
    Predictions& predictions,
    real threshold) const {
  std::vector<int32_t> features;
  dict_->addFeatures(tokens, features);
  if (features.empty()) {
    return;
  }
  Model::State state(args_->dim, dict_->nlabels(), 0);
  const bool check = !lexicon_.empty() && args_->lexThreshold > 0;
  if (!check && experts_.empty()) {
    model_->predict(features, k, threshold, predictions, state);
    return;
  }
  model_->predict(features, check ? std::max(k, args_->lexTopK) : k, 0.0,
      predictions, state);
  // The first of the top -lexTopK labels that passes the lexicon check goes
  // first; if none passes, the order is kept.
  if (check) {
    const std::vector<uint64_t> words = Lexicon::words(tokens);
    const size_t tried = std::min<size_t>(args_->lexTopK, predictions.size());
    for (size_t i = 0; i < tried; i++) {
      if (lexicon_.passes(predictions[i].second, words, args_->lexThreshold,
              args_->lexMinWords)) {
        std::rotate(predictions.begin(), predictions.begin() + i,
            predictions.begin() + i + 1);
        break;
      }
    }
  }
  // An expert replaces an unsure top label of its group with its own pick,
  // which gets this model's probability for that label.
  const int32_t top = predictions.empty() ? -1 : predictions[0].second;
  if (top >= 0 && expertOf_[top] >= 0 &&
      state.output[top] < experts_[expertOf_[top]].gate) {
    const Expert& e = experts_[expertOf_[top]];
    Predictions picked;
    e.model->predict(1, tokens, picked, 0.0);
    const int32_t label = picked.empty() ? -1 : e.toMain[picked[0].second];
    if (label >= 0 && label != top) {
      predictions.erase(
          std::remove_if(predictions.begin(), predictions.end(),
              [&](const std::pair<real, int32_t>& p) { return p.second == label; }),
          predictions.end());
      predictions.insert(predictions.begin(),
          std::make_pair(model_->std_log(state.output[label]), label));
    }
  }
  if (predictions.size() > size_t(k)) {
    predictions.resize(k);
  }
  const real logThreshold = std::log(threshold);
  predictions.erase(
      std::remove_if(predictions.begin(), predictions.end(),
          [&](const std::pair<real, int32_t>& p) { return p.first < logThreshold; }),
      predictions.end());
}

void ByteFast::setLexiconThreshold(real threshold) {
  args_->lexThreshold = threshold;
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

  std::vector<std::string> tokens;
  std::vector<int32_t> labels;
  dict_->readLine(in, tokens, labels);
  Predictions linePredictions;
  predict(k, tokens, linePredictions, threshold);
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
      if (localTokenCount > kLrUpdateRate) {
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
  lexicon_ = Lexicon();
  if (!args_->lexicon.empty()) {
    lexicon_.build(args_->lexicon, *dict_, args_->label, args_->verbose);
  }
  readLabelWeights();
  experts_.clear();
  expertOf_.assign(dict_->nlabels(), -1);

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
  // Same condition as trainThread
  while (keepTraining(ntokens)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (loss_ >= 0 && args_->verbose > 1) {
      real progress = real(tokenCount_) / (args_->epoch * ntokens);
      std::cerr << "\r";
      printInfo(progress, loss_, std::cerr);
    }
  }
  for (int32_t i = 0; i < threads.size(); i++) {
    threads[i].join();
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
}

int ByteFast::getDimension() const {
  return args_->dim;
}

} // namespace bytefast
