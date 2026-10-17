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

bool comparePairs(
    const std::pair<real, std::string>& l,
    const std::pair<real, std::string>& r);

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

ByteFast::ByteFast()
    : quant_(false), wordVectors_(nullptr), trainException_(nullptr) {}

void ByteFast::addInputVector(Vector& vec, int32_t ind) const {
  vec.addRow(*input_, ind);
}

std::shared_ptr<const Dictionary> ByteFast::getDictionary() const {
  return dict_;
}

const Args ByteFast::getArgs() const {
  return *args_.get();
}

std::shared_ptr<const DenseMatrix> ByteFast::getInputMatrix() const {
  if (quant_) {
    throw std::runtime_error("Can't export quantized matrix");
  }
  assert(input_.get());
  return std::dynamic_pointer_cast<DenseMatrix>(input_);
}

void ByteFast::setMatrices(
    const std::shared_ptr<DenseMatrix>& inputMatrix,
    const std::shared_ptr<DenseMatrix>& outputMatrix) {
  assert(input_->size(1) == output_->size(1));

  input_ = std::dynamic_pointer_cast<Matrix>(inputMatrix);
  output_ = std::dynamic_pointer_cast<Matrix>(outputMatrix);
  wordVectors_.reset();
  args_->dim = input_->size(1);

  buildModel();
}

std::shared_ptr<const DenseMatrix> ByteFast::getOutputMatrix() const {
  if (quant_ && args_->qout) {
    throw std::runtime_error("Can't export quantized matrix");
  }
  assert(output_.get());
  return std::dynamic_pointer_cast<DenseMatrix>(output_);
}

int32_t ByteFast::getWordId(const std::string& word) const {
  return dict_->getId(word);
}

int32_t ByteFast::getSubwordId(const std::string& subword) const {
  int32_t h = dict_->hash(subword) % args_->bucket;
  return dict_->nwords() + h;
}

int32_t ByteFast::getLabelId(const std::string& label) const {
  int32_t labelId = dict_->getId(label);
  if (labelId != -1) {
    labelId -= dict_->nwords();
  }
  return labelId;
}

void ByteFast::getWordVector(Vector& vec, const std::string& word) const {
  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
  vec.zero();
  for (int i = 0; i < ngrams.size(); i++) {
    addInputVector(vec, ngrams[i]);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}

void ByteFast::getSubwordVector(Vector& vec, const std::string& subword) const {
  vec.zero();
  int32_t h = dict_->hash(subword) % args_->bucket;
  h = h + dict_->nwords();
  addInputVector(vec, h);
}

void ByteFast::saveVectors(const std::string& filename) {
  if (!input_ || !output_) {
    throw std::runtime_error("Model never trained");
  }
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        filename + " cannot be opened for saving vectors!");
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getWordVector(vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void ByteFast::saveOutput(const std::string& filename) {
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        filename + " cannot be opened for saving vectors!");
  }
  if (quant_) {
    throw std::invalid_argument(
        "Option -saveOutput is not supported for quantized models.");
  }
  int32_t n =
      (args_->model == model_name::sup) ? dict_->nlabels() : dict_->nwords();
  ofs << n << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < n; i++) {
    std::string word = (args_->model == model_name::sup) ? dict_->getLabel(i)
                                                         : dict_->getWord(i);
    vec.zero();
    vec.addRow(*output_, i);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
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
  if (args_->model == model_name::sup) {
    return dict_->getCounts(entry_type::label);
  } else {
    return dict_->getCounts(entry_type::word);
  }
}

void ByteFast::buildModel() {
  auto loss = createLoss(output_);
  bool normalizeGradient = (args_->model == model_name::sup);
  model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
}

void ByteFast::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();
  input_ = std::make_shared<DenseMatrix>();
  output_ = std::make_shared<DenseMatrix>();
  args_->load(in);
  if (textFeatures_) {
    args_->loadTextFeatures(in);
  }
  if (version == 11 && args_->model == model_name::sup) {
    // backward compatibility: old supervised models do not use char ngrams.
    args_->maxn = 0;
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
  if (args_->model != model_name::sup) {
    throw std::invalid_argument(
        "For now we only support quantization of supervised models");
  }
  args_->input = qargs.input;
  args_->qout = qargs.qout;
  args_->output = qargs.output;
  std::shared_ptr<DenseMatrix> input =
      std::dynamic_pointer_cast<DenseMatrix>(input_);
  std::shared_ptr<DenseMatrix> output =
      std::dynamic_pointer_cast<DenseMatrix>(output_);
  bool normalizeGradient = (args_->model == model_name::sup);

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
      model_ = std::make_shared<Model>(input, output, loss, normalizeGradient);
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
  model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
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

void ByteFast::cbow(
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model_->update(bow, line, w, lr, state);
  }
}

void ByteFast::skipgram(
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        model_->update(ngrams, line, w + c, lr, state);
      }
    }
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
  if (args_->model != model_name::sup) {
    throw std::invalid_argument("Model needs to be supervised for prediction!");
  }
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

void ByteFast::getSentenceVector(std::istream& in, bytefast::Vector& svec) {
  svec.zero();
  if (args_->model == model_name::sup) {
    std::vector<int32_t> line, labels;
    dict_->getLine(in, line, labels);
    for (int32_t i = 0; i < line.size(); i++) {
      addInputVector(svec, line[i]);
    }
    if (!line.empty()) {
      svec.mul(1.0 / line.size());
    }
  } else {
    Vector vec(args_->dim);
    std::string sentence;
    std::getline(in, sentence);
    std::istringstream iss(sentence);
    std::string word;
    int32_t count = 0;
    while (iss >> word) {
      getWordVector(vec, word);
      real norm = vec.norm();
      if (norm > 0) {
        vec.mul(1.0 / norm);
        svec.addVector(vec);
        count++;
      }
    }
    if (count > 0) {
      svec.mul(1.0 / count);
    }
  }
}

std::vector<std::pair<std::string, Vector>> ByteFast::getNgramVectors(
    const std::string& word) const {
  std::vector<std::pair<std::string, Vector>> result;
  std::vector<int32_t> ngrams;
  std::vector<std::string> substrings;
  dict_->getSubwords(word, ngrams, substrings);
  assert(ngrams.size() <= substrings.size());
  for (int32_t i = 0; i < ngrams.size(); i++) {
    Vector vec(args_->dim);
    if (ngrams[i] >= 0) {
      vec.addRow(*input_, ngrams[i]);
    }
    result.emplace_back(substrings[i], std::move(vec));
  }
  return result;
}

void ByteFast::precomputeWordVectors(DenseMatrix& wordVectors) {
  Vector vec(args_->dim);
  wordVectors.zero();
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getWordVector(vec, word);
    real norm = vec.norm();
    if (norm > 0) {
      wordVectors.addVectorToRow(vec, i, 1.0 / norm);
    }
  }
}

void ByteFast::lazyComputeWordVectors() {
  if (!wordVectors_) {
    wordVectors_ = std::unique_ptr<DenseMatrix>(
        new DenseMatrix(dict_->nwords(), args_->dim));
    precomputeWordVectors(*wordVectors_);
  }
}

std::vector<std::pair<real, std::string>> ByteFast::getNN(
    const std::string& word,
    int32_t k) {
  Vector query(args_->dim);

  getWordVector(query, word);

  lazyComputeWordVectors();
  assert(wordVectors_);
  return getNN(*wordVectors_, query, k, {word});
}

std::vector<std::pair<real, std::string>> ByteFast::getNN(
    const DenseMatrix& wordVectors,
    const Vector& query,
    int32_t k,
    const std::set<std::string>& banSet) {
  std::vector<std::pair<real, std::string>> heap;

  real queryNorm = query.norm();
  if (std::abs(queryNorm) < 1e-8) {
    queryNorm = 1;
  }

  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    if (banSet.find(word) == banSet.end()) {
      real dp = wordVectors.dotRow(query, i);
      real similarity = dp / queryNorm;

      if (heap.size() == k && similarity < heap.front().first) {
        continue;
      }
      heap.push_back(std::make_pair(similarity, word));
      std::push_heap(heap.begin(), heap.end(), comparePairs);
      if (heap.size() > k) {
        std::pop_heap(heap.begin(), heap.end(), comparePairs);
        heap.pop_back();
      }
    }
  }
  std::sort_heap(heap.begin(), heap.end(), comparePairs);

  return heap;
}

std::vector<std::pair<real, std::string>> ByteFast::getAnalogies(
    int32_t k,
    const std::string& wordA,
    const std::string& wordB,
    const std::string& wordC) {
  Vector query(args_->dim);
  query.zero();

  Vector buffer(args_->dim);
  getWordVector(buffer, wordA);
  query.addVector(buffer, 1.0 / (buffer.norm() + 1e-8));
  getWordVector(buffer, wordB);
  query.addVector(buffer, -1.0 / (buffer.norm() + 1e-8));
  getWordVector(buffer, wordC);
  query.addVector(buffer, 1.0 / (buffer.norm() + 1e-8));

  lazyComputeWordVectors();
  assert(wordVectors_);
  return getNN(*wordVectors_, query, k, {wordA, wordB, wordC});
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
      if (args_->model == model_name::sup) {
        localTokenCount += dict_->getLine(ifs, line, labels, state.rng);
        supervised(state, lr, line, labels);
      } else if (args_->model == model_name::cbow) {
        localTokenCount += dict_->getLine(ifs, line, state.rng);
        cbow(state, lr, line);
      } else if (args_->model == model_name::sg) {
        localTokenCount += dict_->getLine(ifs, line, state.rng);
        skipgram(state, lr, line);
      }
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

std::shared_ptr<Matrix> ByteFast::getInputMatrixFromFile(
    const std::string& filename) const {
  std::ifstream in(filename);
  std::vector<std::string> words;
  std::shared_ptr<DenseMatrix> mat; // temp. matrix for pretrained vectors
  int64_t n, dim;
  if (!in.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  in >> n >> dim;
  if (dim != args_->dim) {
    throw std::invalid_argument(
        "Dimension of pretrained vectors (" + std::to_string(dim) +
        ") does not match dimension (" + std::to_string(args_->dim) + ")!");
  }
  mat = std::make_shared<DenseMatrix>(n, dim);
  for (size_t i = 0; i < n; i++) {
    std::string word;
    in >> word;
    words.push_back(word);
    dict_->add(word);
    for (size_t j = 0; j < dim; j++) {
      in >> mat->at(i, j);
    }
  }
  in.close();

  dict_->threshold(1, 0);
  dict_->init();
  std::shared_ptr<DenseMatrix> input = std::make_shared<DenseMatrix>(
      dict_->nwords() + args_->bucket, args_->dim);
  input->uniform(1.0 / args_->dim, args_->thread, args_->seed);

  for (size_t i = 0; i < n; i++) {
    int32_t idx = dict_->getId(words[i]);
    if (idx < 0 || idx >= dict_->nwords()) {
      continue;
    }
    for (size_t j = 0; j < dim; j++) {
      input->at(idx, j) = mat->at(i, j);
    }
  }
  return input;
}

std::shared_ptr<Matrix> ByteFast::createRandomMatrix() const {
  std::shared_ptr<DenseMatrix> input = std::make_shared<DenseMatrix>(
      dict_->nwords() + args_->bucket, args_->dim);
  input->uniform(1.0 / args_->dim, args_->thread, args_->seed);

  return input;
}

std::shared_ptr<Matrix> ByteFast::createTrainOutputMatrix() const {
  int64_t m =
      (args_->model == model_name::sup) ? dict_->nlabels() : dict_->nwords();
  std::shared_ptr<DenseMatrix> output =
      std::make_shared<DenseMatrix>(m, args_->dim);
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

  if (!args_->pretrainedVectors.empty()) {
    input_ = getInputMatrixFromFile(args_->pretrainedVectors);
  } else {
    input_ = createRandomMatrix();
  }
  output_ = createTrainOutputMatrix();
  quant_ = false;
  auto loss = createLoss(output_);
  bool normalizeGradient = (args_->model == model_name::sup);
  model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
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
  if (args_->model != model_name::sup || args.model != model_name::sup) {
    throw std::invalid_argument("-inputModel only applies to supervised models");
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
  args_->saveOutput = args.saveOutput;
  args_->inputModel = args.inputModel;
  args_->cropProb = args.cropProb;
  args_->cropMin = args.cropMin;
  args_->cropMax = args.cropMax;
  args_->cropUnit = args.cropUnit;
  args_->normProb = args.normProb;
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

bool ByteFast::isQuant() const {
  return quant_;
}

bool comparePairs(
    const std::pair<real, std::string>& l,
    const std::pair<real, std::string>& r) {
  return l.first > r.first;
}

} // namespace bytefast
