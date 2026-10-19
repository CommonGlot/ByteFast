/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <args.h>
#include <bytefast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <real.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

py::str castToPythonString(const std::string& s, const char* onUnicodeError) {
  PyObject* handle = PyUnicode_DecodeUTF8(s.data(), s.length(), onUnicodeError);
  if (!handle) {
    throw py::error_already_set();
  }

  py::str handle_str = py::str(handle);
  Py_DECREF(handle);
  return handle_str;
}

std::vector<std::pair<bytefast::real, py::str>> castToPythonString(
    const std::vector<std::pair<bytefast::real, std::string>>& predictions,
    const char* onUnicodeError) {
  std::vector<std::pair<bytefast::real, py::str>> transformedPredictions;

  for (const auto& prediction : predictions) {
    transformedPredictions.emplace_back(
        prediction.first,
        castToPythonString(prediction.second, onUnicodeError));
  }

  return transformedPredictions;
}

PYBIND11_MODULE(_core_pybind, m) {
  py::class_<bytefast::Args>(m, "args", py::module_local())
      .def(py::init<>())
      .def_readwrite("input", &bytefast::Args::input)
      .def_readwrite("output", &bytefast::Args::output)
      .def_readwrite("lr", &bytefast::Args::lr)
      .def_readwrite("dim", &bytefast::Args::dim)
      .def_readwrite("epoch", &bytefast::Args::epoch)
      .def_readwrite("minCount", &bytefast::Args::minCount)
      .def_readwrite("loss", &bytefast::Args::loss)
      .def_readwrite("bucket", &bytefast::Args::bucket)
      .def_readwrite("minn", &bytefast::Args::minn)
      .def_readwrite("maxn", &bytefast::Args::maxn)
      .def_readwrite("minb", &bytefast::Args::minb)
      .def_readwrite("maxb", &bytefast::Args::maxb)
      .def_readwrite("maxBytes", &bytefast::Args::maxBytes)
      .def_readwrite("wordlist", &bytefast::Args::wordlist)
      .def_readwrite("lexicon", &bytefast::Args::lexicon)
      .def_readwrite("lexThreshold", &bytefast::Args::lexThreshold)
      .def_readwrite("lexMinWords", &bytefast::Args::lexMinWords)
      .def_readwrite("lexTopK", &bytefast::Args::lexTopK)
      .def_readwrite("cropProb", &bytefast::Args::cropProb)
      .def_readwrite("cropMin", &bytefast::Args::cropMin)
      .def_readwrite("cropMax", &bytefast::Args::cropMax)
      .def_readwrite("normProb", &bytefast::Args::normProb)
      .def_readwrite("stripProb", &bytefast::Args::stripProb)
      .def_readwrite("thread", &bytefast::Args::thread)
      .def_readwrite("label", &bytefast::Args::label)
      .def_readwrite("verbose", &bytefast::Args::verbose)
      .def_readwrite("seed", &bytefast::Args::seed);

  py::enum_<bytefast::loss_name>(m, "loss_name", py::module_local())
      .value("softmax", bytefast::loss_name::softmax)
      .export_values();

  m.def(
      "train",
      [](bytefast::ByteFast& ft, bytefast::Args& a) { ft.train(a); },
      py::call_guard<py::gil_scoped_release>());

  py::class_<bytefast::ByteFast>(m, "bytefast", py::module_local())
      .def(py::init<>())
      .def("getArgs", &bytefast::ByteFast::getArgs)
      .def(
          "loadModel",
          [](bytefast::ByteFast& m, std::string s) { m.loadModel(s); })
      .def(
          "saveModel",
          [](bytefast::ByteFast& m, std::string s) { m.saveModel(s); })
      .def(
          "test",
          [](bytefast::ByteFast& m,
             const std::string& filename,
             int32_t k,
             bytefast::real threshold) {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            bytefast::Meter meter;
            m.test(ifs, k, threshold, meter);
            ifs.close();
            return std::tuple<int64_t, double, double>(
                meter.nexamples(), meter.precision(), meter.recall());
          })
      .def(
          "setLexiconThreshold",
          [](bytefast::ByteFast& m, bytefast::real threshold) {
            m.setLexiconThreshold(threshold);
          })
      .def(
          "getVocab",
          [](bytefast::ByteFast& m, const char* onUnicodeError) {
            py::str s;
            std::vector<py::str> vocab_list;
            std::vector<int64_t> vocab_freq;
            std::shared_ptr<const bytefast::Dictionary> d = m.getDictionary();
            vocab_freq = d->getCounts(bytefast::entry_type::word);
            for (int32_t i = 0; i < vocab_freq.size(); i++) {
              vocab_list.push_back(
                  castToPythonString(d->getWord(i), onUnicodeError));
            }
            return std::pair<std::vector<py::str>, std::vector<int64_t>>(
                vocab_list, vocab_freq);
          })
      .def(
          "getLabels",
          [](bytefast::ByteFast& m, const char* onUnicodeError) {
            std::vector<py::str> labels_list;
            std::vector<int64_t> labels_freq;
            std::shared_ptr<const bytefast::Dictionary> d = m.getDictionary();
            labels_freq = d->getCounts(bytefast::entry_type::label);
            for (int32_t i = 0; i < labels_freq.size(); i++) {
              labels_list.push_back(
                  castToPythonString(d->getLabel(i), onUnicodeError));
            }
            return std::pair<std::vector<py::str>, std::vector<int64_t>>(
                labels_list, labels_freq);
          })
      .def(
          "predict",
          // NOTE: text needs to end in a newline
          // to exactly mimic the behavior of the cli
          [](bytefast::ByteFast& m,
             const std::string text,
             int32_t k,
             bytefast::real threshold,
             const char* onUnicodeError) {
            std::stringstream ioss(text);
            std::vector<std::pair<bytefast::real, std::string>> predictions;
            m.predictLine(ioss, predictions, k, threshold);

            return castToPythonString(predictions, onUnicodeError);
          })
      .def(
          "multilinePredict",
          // NOTE: text needs to end in a newline
          // to exactly mimic the behavior of the cli
          [](bytefast::ByteFast& m,
             const std::vector<std::string>& lines,
             int32_t k,
             bytefast::real threshold,
             const char* onUnicodeError) {
            std::vector<py::array_t<bytefast::real>> allProbabilities;
            std::vector<std::vector<py::str>> allLabels;
            std::vector<std::pair<bytefast::real, std::string>> predictions;

            for (const std::string& text : lines) {
              std::stringstream ioss(text);
              m.predictLine(ioss, predictions, k, threshold);
              std::vector<bytefast::real> probabilities;
              std::vector<py::str> labels;

              for (const auto& prediction : predictions) {
                probabilities.push_back(prediction.first);
                labels.push_back(
                    castToPythonString(prediction.second, onUnicodeError));
              }

              allProbabilities.emplace_back(
                  probabilities.size(), probabilities.data());
              allLabels.push_back(labels);
            }

            return make_pair(allLabels, allProbabilities);
          });
}
