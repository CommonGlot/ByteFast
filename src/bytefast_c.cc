// Copyright (c) 2026 CommonGlot contributors. MIT License.

#include "bytefast_c.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "bytefast.h"

struct bytefast_model {
  bytefast::ByteFast model;
  std::vector<std::string> labels;
};

namespace {

void setError(char** error, const char* message) {
  if (error) {
    *error = static_cast<char*>(std::malloc(std::strlen(message) + 1));
    if (*error) {
      std::strcpy(*error, message);
    }
  }
}

} // namespace

extern "C" {

bytefast_model* bytefast_load(const char* path, char** error) {
  try {
    auto* m = new bytefast_model();
    try {
      m->model.loadModel(path);
      auto dict = m->model.getDictionary();
      for (int32_t i = 0; i < dict->nlabels(); i++) {
        m->labels.push_back(dict->getLabel(i));
      }
      return m;
    } catch (...) {
      delete m;
      throw;
    }
  } catch (const std::exception& e) {
    setError(error, e.what());
  } catch (...) {
    setError(error, "unknown error");
  }
  return nullptr;
}

void bytefast_free(bytefast_model* model) {
  delete model;
}

void bytefast_free_error(char* error) {
  std::free(error);
}

int32_t bytefast_predict(
    const bytefast_model* model,
    const char* text,
    size_t length,
    int32_t k,
    float threshold,
    bytefast_prediction* out,
    char** error) {
  try {
    if (!model || k < 1 || (!text && length > 0)) {
      throw std::invalid_argument("bytefast_predict: invalid argument");
    }
    // As on the command line, the line ends with a newline (the EOS token).
    std::string line(text ? text : "", length);
    line.push_back('\n');
    std::istringstream in(line);
    std::vector<std::string> tokens;
    std::vector<int32_t> labels;
    model->model.getDictionary()->readLine(in, tokens, labels);
    bytefast::Predictions predictions;
    model->model.predict(k, tokens, predictions, threshold);
    int32_t n = 0;
    for (const auto& p : predictions) {
      out[n].label = p.second;
      out[n].probability = std::exp(p.first);
      n++;
    }
    return n;
  } catch (const std::exception& e) {
    setError(error, e.what());
  } catch (...) {
    setError(error, "unknown error");
  }
  return -1;
}

int32_t bytefast_num_labels(const bytefast_model* model) {
  return model ? int32_t(model->labels.size()) : 0;
}

const char* bytefast_label(const bytefast_model* model, int32_t label) {
  if (!model || label < 0 || label >= int32_t(model->labels.size())) {
    return nullptr;
  }
  return model->labels[label].c_str();
}

int32_t bytefast_dimension(const bytefast_model* model) {
  return model ? model->model.getDimension() : 0;
}

} // extern "C"
