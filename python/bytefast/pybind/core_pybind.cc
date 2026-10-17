/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <args.h>
#include <autotune.h>
#include <densematrix.h>
#include <bytefast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <real.h>
#include <vector.h>
#include <cmath>
#include <iterator>
#include <sstream>
#include <stdexcept>

using namespace pybind11::literals;
namespace py = pybind11;

py::str castToPythonString(const std::string& s, const char* onUnicodeError) {
  PyObject* handle = PyUnicode_DecodeUTF8(s.data(), s.length(), onUnicodeError);
  if (!handle) {
    throw py::error_already_set();
  }

  // py::str's constructor from a PyObject assumes the string has been encoded
  // for python 2 and not encoded for python 3 :
  // https://github.com/pybind/pybind11/blob/ccbe68b084806dece5863437a7dc93de20bd9b15/include/pybind11/pytypes.h#L930
#if PY_MAJOR_VERSION < 3
  PyObject* handle_encoded =
      PyUnicode_AsEncodedString(handle, "utf-8", onUnicodeError);
  Py_DECREF(handle);
  handle = handle_encoded;
#endif

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

std::pair<std::vector<py::str>, std::vector<py::str>> getLineText(
    bytefast::ByteFast& m,
    const std::string text,
    const char* onUnicodeError) {
  std::shared_ptr<const bytefast::Dictionary> d = m.getDictionary();
  std::stringstream ioss(text);
  std::string token;
  std::vector<py::str> words;
  std::vector<py::str> labels;
  while (d->readWord(ioss, token)) {
    uint32_t h = d->hash(token);
    int32_t wid = d->getId(token, h);
    bytefast::entry_type type = wid < 0 ? d->getType(token) : d->getType(wid);

    if (type == bytefast::entry_type::word) {
      words.push_back(castToPythonString(token, onUnicodeError));
      // Labels must not be OOV!
    } else if (type == bytefast::entry_type::label && wid >= 0) {
      labels.push_back(castToPythonString(token, onUnicodeError));
    }
    if (token == bytefast::Dictionary::EOS) {
      break;
}
  }
  return std::pair<std::vector<py::str>, std::vector<py::str>>(words, labels);
}

PYBIND11_MODULE(_core_pybind, m) {
  py::class_<bytefast::Args>(m, "args", py::module_local())
      .def(py::init<>())
      .def_readwrite("input", &bytefast::Args::input)
      .def_readwrite("output", &bytefast::Args::output)
      .def_readwrite("lr", &bytefast::Args::lr)
      .def_readwrite("lrUpdateRate", &bytefast::Args::lrUpdateRate)
      .def_readwrite("dim", &bytefast::Args::dim)
      .def_readwrite("ws", &bytefast::Args::ws)
      .def_readwrite("epoch", &bytefast::Args::epoch)
      .def_readwrite("minCount", &bytefast::Args::minCount)
      .def_readwrite("minCountLabel", &bytefast::Args::minCountLabel)
      .def_readwrite("neg", &bytefast::Args::neg)
      .def_readwrite("wordNgrams", &bytefast::Args::wordNgrams)
      .def_readwrite("loss", &bytefast::Args::loss)
      .def_readwrite("model", &bytefast::Args::model)
      .def_readwrite("bucket", &bytefast::Args::bucket)
      .def_readwrite("minn", &bytefast::Args::minn)
      .def_readwrite("maxn", &bytefast::Args::maxn)
      .def_readwrite("minb", &bytefast::Args::minb)
      .def_readwrite("maxb", &bytefast::Args::maxb)
      .def_readwrite("minc", &bytefast::Args::minc)
      .def_readwrite("maxc", &bytefast::Args::maxc)
      .def_readwrite("maxBytes", &bytefast::Args::maxBytes)
      .def_readwrite("fold", &bytefast::Args::fold)
      .def_readwrite("hashWords", &bytefast::Args::hashWords)
      .def_readwrite("wordlist", &bytefast::Args::wordlist)
      .def_readwrite("maxVocab", &bytefast::Args::maxVocab)
      .def_readwrite("inputModel", &bytefast::Args::inputModel)
      .def_readwrite("cropProb", &bytefast::Args::cropProb)
      .def_readwrite("cropMin", &bytefast::Args::cropMin)
      .def_readwrite("cropMax", &bytefast::Args::cropMax)
      .def_readwrite("cropUnit", &bytefast::Args::cropUnit)
      .def_readwrite("thread", &bytefast::Args::thread)
      .def_readwrite("t", &bytefast::Args::t)
      .def_readwrite("label", &bytefast::Args::label)
      .def_readwrite("verbose", &bytefast::Args::verbose)
      .def_readwrite("pretrainedVectors", &bytefast::Args::pretrainedVectors)
      .def_readwrite("saveOutput", &bytefast::Args::saveOutput)
      .def_readwrite("seed", &bytefast::Args::seed)

      .def_readwrite("qout", &bytefast::Args::qout)
      .def_readwrite("retrain", &bytefast::Args::retrain)
      .def_readwrite("qnorm", &bytefast::Args::qnorm)
      .def_readwrite("cutoff", &bytefast::Args::cutoff)
      .def_readwrite("dsub", &bytefast::Args::dsub)

      .def_readwrite(
          "autotuneValidationFile", &bytefast::Args::autotuneValidationFile)
      .def_readwrite("autotuneMetric", &bytefast::Args::autotuneMetric)
      .def_readwrite(
          "autotunePredictions", &bytefast::Args::autotunePredictions)
      .def_readwrite("autotuneDuration", &bytefast::Args::autotuneDuration)
      .def_readwrite("autotuneModelSize", &bytefast::Args::autotuneModelSize)
      .def("setManual", [](bytefast::Args& m, const std::string& argName) {
        m.setManual(argName);
      });

  py::enum_<bytefast::model_name>(m, "model_name", py::module_local())
      .value("cbow", bytefast::model_name::cbow)
      .value("skipgram", bytefast::model_name::sg)
      .value("supervised", bytefast::model_name::sup)
      .export_values();

  py::enum_<bytefast::fold_name>(m, "fold_name", py::module_local())
      .value("none", bytefast::fold_name::none)
      .value("ascii", bytefast::fold_name::ascii)
      .value("unicode", bytefast::fold_name::unicode);

  py::enum_<bytefast::loss_name>(m, "loss_name", py::module_local())
      .value("hs", bytefast::loss_name::hs)
      .value("ns", bytefast::loss_name::ns)
      .value("softmax", bytefast::loss_name::softmax)
      .value("ova", bytefast::loss_name::ova)
      .export_values();

  py::enum_<bytefast::metric_name>(m, "metric_name", py::module_local())
      .value("f1score", bytefast::metric_name::f1score)
      .value("f1scoreLabel", bytefast::metric_name::f1scoreLabel)
      .value("precisionAtRecall", bytefast::metric_name::precisionAtRecall)
      .value(
          "precisionAtRecallLabel",
          bytefast::metric_name::precisionAtRecallLabel)
      .value("recallAtPrecision", bytefast::metric_name::recallAtPrecision)
      .value(
          "recallAtPrecisionLabel",
          bytefast::metric_name::recallAtPrecisionLabel)
      .export_values();

  m.def(
      "train",
      [](bytefast::ByteFast& ft, bytefast::Args& a) {
        if (a.hasAutotune()) {
          bytefast::Autotune autotune(std::shared_ptr<bytefast::ByteFast>(
              &ft, [](bytefast::ByteFast*) {}));
          autotune.train(a);
        } else {
          ft.train(a);
        }
      },
      py::call_guard<py::gil_scoped_release>());

  py::class_<bytefast::Vector>(m, "Vector", py::buffer_protocol(), py::module_local())
      .def(py::init<ssize_t>())
      .def_buffer([](bytefast::Vector& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),
            sizeof(bytefast::real),
            py::format_descriptor<bytefast::real>::format(),
            1,
            {m.size()},
            {sizeof(bytefast::real)});
      });

  py::class_<bytefast::DenseMatrix>(
      m, "DenseMatrix", py::buffer_protocol(), py::module_local())
      .def(py::init<>())
      .def(py::init<ssize_t, ssize_t>())
      .def_buffer([](bytefast::DenseMatrix& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),
            sizeof(bytefast::real),
            py::format_descriptor<bytefast::real>::format(),
            2,
            {m.size(0), m.size(1)},
            {sizeof(bytefast::real) * m.size(1),
             sizeof(bytefast::real) * (int64_t)1});
      });

  py::class_<bytefast::Meter>(m, "Meter", py::module_local())
      .def(py::init<bool>())
      .def("scoreVsTrue", &bytefast::Meter::scoreVsTrue)
      .def(
          "precisionRecallCurveLabel",
          (std::vector<std::pair<double, double>>(bytefast::Meter::*)(int32_t)
               const) &
              bytefast::Meter::precisionRecallCurve)
      .def(
          "precisionRecallCurve",
          (std::vector<std::pair<double, double>>(bytefast::Meter::*)() const) &
              bytefast::Meter::precisionRecallCurve)
      .def(
          "precisionAtRecallLabel",
          (double (bytefast::Meter::*)(int32_t, double) const) &
              bytefast::Meter::precisionAtRecall)
      .def(
          "precisionAtRecall",
          (double (bytefast::Meter::*)(double) const) &
              bytefast::Meter::precisionAtRecall)
      .def(
          "recallAtPrecisionLabel",
          (double (bytefast::Meter::*)(int32_t, double) const) &
              bytefast::Meter::recallAtPrecision)
      .def(
          "recallAtPrecision",
          (double (bytefast::Meter::*)(double) const) &
              bytefast::Meter::recallAtPrecision);

  py::class_<bytefast::ByteFast>(m, "bytefast", py::module_local())
      .def(py::init<>())
      .def("getArgs", &bytefast::ByteFast::getArgs)
      .def(
          "getInputMatrix",
          [](bytefast::ByteFast& m) {
            std::shared_ptr<const bytefast::DenseMatrix> mm =
                m.getInputMatrix();
            return mm.get();
          },
          pybind11::return_value_policy::reference)
      .def(
          "getOutputMatrix",
          [](bytefast::ByteFast& m) {
            std::shared_ptr<const bytefast::DenseMatrix> mm =
                m.getOutputMatrix();
            return mm.get();
          },
          pybind11::return_value_policy::reference)
      .def(
          "setMatrices",
          [](bytefast::ByteFast& m,
             py::buffer inputMatrixBuffer,
             py::buffer outputMatrixBuffer) {
            py::buffer_info inputMatrixInfo = inputMatrixBuffer.request();
            py::buffer_info outputMatrixInfo = outputMatrixBuffer.request();

            m.setMatrices(
                std::make_shared<bytefast::DenseMatrix>(
                    inputMatrixInfo.shape[0],
                    inputMatrixInfo.shape[1],
                    static_cast<float*>(inputMatrixInfo.ptr)),
                std::make_shared<bytefast::DenseMatrix>(
                    outputMatrixInfo.shape[0],
                    outputMatrixInfo.shape[1],
                    static_cast<float*>(outputMatrixInfo.ptr)));
          })
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
            bytefast::Meter meter(false);
            m.test(ifs, k, threshold, meter);
            ifs.close();
            return std::tuple<int64_t, double, double>(
                meter.nexamples(), meter.precision(), meter.recall());
          })
      .def(
          "getMeter",
          [](bytefast::ByteFast& m, const std::string& filename, int32_t k) {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            bytefast::Meter meter(true);
            m.test(ifs, k, 0.0, meter);
            ifs.close();

            return meter;
          })
      .def(
          "getSentenceVector",
          [](bytefast::ByteFast& m,
             bytefast::Vector& v,
             const std::string text) {
            std::stringstream ioss(text);
            m.getSentenceVector(ioss, v);
          })
      .def(
          "tokenize",
          [](bytefast::ByteFast& m, const std::string text) {
            std::vector<std::string> text_split;
            std::shared_ptr<const bytefast::Dictionary> d = m.getDictionary();
            std::stringstream ioss(text);
            std::string token;
            while (!ioss.eof()) {
              while (d->readWord(ioss, token)) {
                text_split.push_back(token);
              }
            }
            return text_split;
          })
      .def("getLine", &getLineText)
      .def(
          "multilineGetLine",
          [](bytefast::ByteFast& m,
             const std::vector<std::string> lines,
             const char* onUnicodeError) {
            std::shared_ptr<const bytefast::Dictionary> d = m.getDictionary();
            std::vector<std::vector<py::str>> all_words;
            std::vector<std::vector<py::str>> all_labels;
            for (const auto& text : lines) {
              auto pair = getLineText(m, text, onUnicodeError);
              all_words.push_back(pair.first);
              all_labels.push_back(pair.second);
            }
            return std::pair<
                std::vector<std::vector<py::str>>,
                std::vector<std::vector<py::str>>>(all_words, all_labels);
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
          "quantize",
          [](bytefast::ByteFast& m,
             const std::string input,
             bool qout,
             int32_t cutoff,
             bool retrain,
             int epoch,
             double lr,
             int thread,
             int verbose,
             int32_t dsub,
             bool qnorm) {
            bytefast::Args qa = bytefast::Args();
            qa.input = input;
            qa.qout = qout;
            qa.cutoff = cutoff;
            qa.retrain = retrain;
            qa.epoch = epoch;
            qa.lr = lr;
            qa.thread = thread;
            qa.verbose = verbose;
            qa.dsub = dsub;
            qa.qnorm = qnorm;
            m.quantize(qa);
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
          })
      .def(
          "testLabel",
          [](bytefast::ByteFast& m,
             const std::string filename,
             int32_t k,
             bytefast::real threshold) {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            bytefast::Meter meter(false);
            m.test(ifs, k, threshold, meter);
            std::shared_ptr<const bytefast::Dictionary> d = m.getDictionary();
            std::unordered_map<std::string, py::dict> returnedValue;
            for (int32_t i = 0; i < d->nlabels(); i++) {
              returnedValue[d->getLabel(i)] = py::dict(
                  "precision"_a = meter.precision(i),
                  "recall"_a = meter.recall(i),
                  "f1score"_a = meter.f1Score(i));
            }

            return returnedValue;
          })
      .def(
          "getWordId",
          [](bytefast::ByteFast& m, const std::string& word) {
            return m.getWordId(word);
          })
      .def(
          "getSubwordId",
          [](bytefast::ByteFast& m, const std::string word) {
            return m.getSubwordId(word);
          })
      .def(
          "getLabelId",
          [](bytefast::ByteFast& m, const std::string& label) {
            return m.getLabelId(label);
          })
      .def(
          "getInputVector",
          [](bytefast::ByteFast& m, bytefast::Vector& vec, int32_t ind) {
            m.getInputVector(vec, ind);
          })
      .def(
          "getWordVector",
          [](bytefast::ByteFast& m,
             bytefast::Vector& vec,
             const std::string word) { m.getWordVector(vec, word); })
      .def(
          "getNN",
          [](bytefast::ByteFast& m,
             const std::string& word,
             int32_t k,
             const char* onUnicodeError) {
            return castToPythonString(m.getNN(word, k), onUnicodeError);
          })
      .def(
          "getAnalogies",
          [](bytefast::ByteFast& m,
             const std::string& wordA,
             const std::string& wordB,
             const std::string& wordC,
             int32_t k,
             const char* onUnicodeError) {
            return castToPythonString(
                m.getAnalogies(k, wordA, wordB, wordC), onUnicodeError);
          })
      .def(
          "getSubwords",
          [](bytefast::ByteFast& m,
             const std::string word,
             const char* onUnicodeError) {
            std::vector<std::string> subwords;
            std::vector<int32_t> ngrams;
            std::shared_ptr<const bytefast::Dictionary> d = m.getDictionary();
            d->getSubwords(word, ngrams, subwords);
            std::vector<py::str> transformedSubwords;

            for (const auto& subword : subwords) {
              transformedSubwords.push_back(
                  castToPythonString(subword, onUnicodeError));
            }

            return std::pair<std::vector<py::str>, std::vector<int32_t>>(
                transformedSubwords, ngrams);
          })
      .def("isQuant", [](bytefast::ByteFast& m) { return m.isQuant(); });
}
