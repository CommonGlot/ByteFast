/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <emscripten.h>
#include <emscripten/bind.h>
#include <bytefast.h>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

using namespace emscripten;
using namespace bytefast;

std::vector<std::pair<float, std::string>>
predict(ByteFast* fasttext, std::string text, int k, double threshold) {
  std::stringstream ioss(text + std::string("\n"));

  std::vector<std::pair<float, std::string>> predictions;
  fasttext->predictLine(ioss, predictions, k, threshold);

  return predictions;
}

void train(ByteFast* fasttext, Args* args, emscripten::val jsCallback) {
  assert(args);
  assert(fasttext);
  fasttext->train(
      *args,
      [=](float progress, float loss, double wst, double lr, int64_t eta) {
        jsCallback(progress, loss, wst, lr, static_cast<int32_t>(eta));
      });
}

std::pair<std::vector<std::string>, std::vector<int32_t>> getTokens(
    const ByteFast& fasttext,
    const std::function<std::string(const Dictionary&, int32_t)> getter,
    entry_type entryType) {
  std::vector<std::string> tokens;
  std::vector<int32_t> retVocabFrequencies;
  std::shared_ptr<const Dictionary> d = fasttext.getDictionary();
  std::vector<int64_t> vocabFrequencies = d->getCounts(entryType);
  for (int32_t i = 0; i < vocabFrequencies.size(); i++) {
    tokens.push_back(getter(*d, i));
    retVocabFrequencies.push_back(vocabFrequencies[i]);
  }
  return std::pair<std::vector<std::string>, std::vector<int32_t>>(
      tokens, retVocabFrequencies);
}

std::pair<std::vector<std::string>, std::vector<int32_t>> getWords(
    ByteFast* fasttext) {
  assert(fasttext);
  return getTokens(*fasttext, &Dictionary::getWord, entry_type::word);
}

std::pair<std::vector<std::string>, std::vector<int32_t>> getLabels(
    ByteFast* fasttext) {
  assert(fasttext);
  return getTokens(*fasttext, &Dictionary::getLabel, entry_type::label);
}

std::pair<std::vector<std::string>, std::vector<std::string>> getLine(
    ByteFast* fasttext,
    const std::string text) {
  assert(fasttext);
  std::shared_ptr<const Dictionary> d = fasttext->getDictionary();
  std::stringstream ioss(text);
  std::string token;
  std::vector<std::string> words;
  std::vector<std::string> labels;
  while (d->readWord(ioss, token)) {
    uint32_t h = d->hash(token);
    int32_t wid = d->getId(token, h);
    entry_type type = wid < 0 ? d->getType(token) : d->getType(wid);

    if (type == entry_type::word) {
      words.push_back(token);
    } else if (type == entry_type::label && wid >= 0) {
      labels.push_back(token);
    }
    if (token == Dictionary::EOS)
      break;
  }
  return std::pair<std::vector<std::string>, std::vector<std::string>>(
      words, labels);
}

Meter test(
    ByteFast* fasttext,
    const std::string& filename,
    int32_t k,
    float threshold) {
  assert(fasttext);
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    throw std::invalid_argument("Test file cannot be opened!");
  }
  Meter meter(false);
  fasttext->test(ifs, k, threshold, meter);
  ifs.close();

  return meter;
}

EMSCRIPTEN_BINDINGS(fasttext) {
  class_<Args>("Args")
      .constructor<>()
      .property("input", &Args::input)
      .property("output", &Args::output)
      .property("lr", &Args::lr)
      .property("lrUpdateRate", &Args::lrUpdateRate)
      .property("dim", &Args::dim)
      .property("epoch", &Args::epoch)
      .property("minCount", &Args::minCount)
      .property("minCountLabel", &Args::minCountLabel)
      .property("neg", &Args::neg)
      .property("wordNgrams", &Args::wordNgrams)
      .property("loss", &Args::loss)
      .property("bucket", &Args::bucket)
      .property("minn", &Args::minn)
      .property("maxn", &Args::maxn)
      .property("thread", &Args::thread)
      .property("label", &Args::label)
      .property("verbose", &Args::verbose)
      .property("seed", &Args::seed)
      .property("qout", &Args::qout)
      .property("retrain", &Args::retrain)
      .property("qnorm", &Args::qnorm)
      .property("cutoff", &Args::cutoff)
      .property("dsub", &Args::dsub)
      .property("minb", &Args::minb)
      .property("maxb", &Args::maxb)
      .property("minc", &Args::minc)
      .property("maxc", &Args::maxc)
      .property("maxBytes", &Args::maxBytes)
      .property("hashWords", &Args::hashWords)
      .property("wordlist", &Args::wordlist)
      .property("maxVocab", &Args::maxVocab)
      .property("cropProb", &Args::cropProb)
      .property("cropMin", &Args::cropMin)
      .property("cropMax", &Args::cropMax)
      .property("cropUnit", &Args::cropUnit)
      .property("normProb", &Args::normProb);

  class_<ByteFast>("ByteFast")
      .constructor<>()
      .function(
          "loadModel",
          select_overload<void(const std::string&)>(&ByteFast::loadModel))
      .function("getWords", &getWords, allow_raw_pointers())
      .function("getLabels", &getLabels, allow_raw_pointers())
      .function("getLine", &getLine, allow_raw_pointers())
      .function("test", &test, allow_raw_pointers())
      .function("predict", &predict, allow_raw_pointers())
      .function("train", &train, allow_raw_pointers())
      .function("saveModel", &ByteFast::saveModel)
      .property("isQuant", &ByteFast::isQuant)
      .property("args", &ByteFast::getArgs);

  class_<Meter>("Meter")
      .constructor<bool>()
      .property(
          "precision", select_overload<double(void) const>(&Meter::precision))
      .property("recall", select_overload<double(void) const>(&Meter::recall))
      .property("f1Score", select_overload<double(void) const>(&Meter::f1Score))
      .function(
          "nexamples",
          optional_override(
              [](const Meter* self) -> int32_t { return self->nexamples(); }),
          allow_raw_pointers());

  enum_<loss_name>("LossName")
      .value("hs", loss_name::hs)
      .value("ns", loss_name::ns)
      .value("softmax", loss_name::softmax)
      .value("ova", loss_name::ova);

  emscripten::value_array<std::pair<float, std::string>>(
      "std::pair<float, std::string>")
      .element(&std::pair<float, std::string>::first)
      .element(&std::pair<float, std::string>::second);

  emscripten::register_vector<std::pair<float, std::string>>(
      "std::vector<std::pair<float, std::string>>");

  emscripten::value_array<
      std::pair<std::vector<std::string>, std::vector<int32_t>>>(
      "std::pair<std::vector<std::string>, std::vector<int32_t>>")
      .element(
          &std::pair<std::vector<std::string>, std::vector<int32_t>>::first)
      .element(
          &std::pair<std::vector<std::string>, std::vector<int32_t>>::second);

  emscripten::value_array<
      std::pair<std::vector<std::string>, std::vector<std::string>>>(
      "std::pair<std::vector<std::string>, std::vector<std::string>>")
      .element(
          &std::pair<std::vector<std::string>, std::vector<std::string>>::first)
      .element(&std::pair<std::vector<std::string>, std::vector<std::string>>::
                   second);

  emscripten::register_vector<float>("std::vector<float>");

  emscripten::register_vector<int32_t>("std::vector<int32_t>");

  emscripten::register_vector<std::string>("std::vector<std::string>");
}
