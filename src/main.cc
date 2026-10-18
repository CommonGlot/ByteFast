/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include "args.h"
#include "bytefast.h"

using namespace bytefast;

void printUsage() {
  std::cerr
      << "usage: bytefast <command> <args>\n\n"
      << "The commands supported by bytefast are:\n\n"
      << "  supervised              train a supervised classifier\n"
      << "  quantize                quantize a model to reduce the memory "
         "usage\n"
      << "  test                    evaluate a supervised classifier\n"
      << "  test-label              print labels with precision and recall "
         "scores\n"
      << "  predict                 predict most likely labels\n"
      << "  predict-prob            predict most likely labels with "
         "probabilities\n"
      << std::endl;
}

void printQuantizeUsage() {
  std::cerr << "usage: bytefast quantize <args>" << std::endl;
}

void printTestUsage() {
  std::cerr
      << "usage: bytefast test <model> <test-data> [<k>] [<th>]\n\n"
      << "  <model>      model filename\n"
      << "  <test-data>  test data filename (if -, read from stdin)\n"
      << "  <k>          (optional; 1 by default) predict top k labels\n"
      << "  <th>         (optional; 0.0 by default) probability threshold\n"
      << std::endl;
}

void printPredictUsage() {
  std::cerr
      << "usage: bytefast predict[-prob] <model> <test-data> [<k>] [<th>] [<threads>]\n\n"
      << "  <model>      model filename\n"
      << "  <test-data>  test data filename (if -, read from stdin)\n"
      << "  <k>          (optional; 1 by default) predict top k labels\n"
      << "  <th>         (optional; 0.0 by default) probability threshold\n"
      << "  <threads>    (optional; 1 by default) threads; output order is kept\n"
      << std::endl;
}

void printTestLabelUsage() {
  std::cerr
      << "usage: bytefast test-label <model> <test-data> [<k>] [<th>]\n\n"
      << "  <model>      model filename\n"
      << "  <test-data>  test data filename\n"
      << "  <k>          (optional; 1 by default) predict top k labels\n"
      << "  <th>         (optional; 0.0 by default) probability threshold\n"
      << std::endl;
}

void quantize(const std::vector<std::string>& args) {
  Args a = Args();
  if (args.size() < 3) {
    printQuantizeUsage();
    a.printHelp();
    exit(EXIT_FAILURE);
  }
  a.parseArgs(args);
  ByteFast fasttext;
  // parseArgs checks if a->output is given.
  fasttext.loadModel(a.output + ".bin");
  fasttext.quantize(a);
  fasttext.saveModel(a.output + ".ftz");
  exit(0);
}

void test(const std::vector<std::string>& args) {
  bool perLabel = args[1] == "test-label";

  if (args.size() < 4 || args.size() > 6) {
    perLabel ? printTestLabelUsage() : printTestUsage();
    exit(EXIT_FAILURE);
  }

  const auto& model = args[2];
  const auto& input = args[3];
  int32_t k = args.size() > 4 ? std::stoi(args[4]) : 1;
  real threshold = args.size() > 5 ? std::stof(args[5]) : 0.0;

  ByteFast fasttext;
  fasttext.loadModel(model);

  Meter meter(false);

  if (input == "-") {
    fasttext.test(std::cin, k, threshold, meter);
  } else {
    std::ifstream ifs(input);
    if (!ifs.is_open()) {
      std::cerr << "Test file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.test(ifs, k, threshold, meter);
  }

  if (perLabel) {
    std::cout << std::fixed << std::setprecision(6);
    auto writeMetric = [](const std::string& name, double value) {
      std::cout << name << " : ";
      if (std::isfinite(value)) {
        std::cout << value;
      } else {
        std::cout << "--------";
      }
      std::cout << "  ";
    };

    std::shared_ptr<const Dictionary> dict = fasttext.getDictionary();
    for (int32_t labelId = 0; labelId < dict->nlabels(); labelId++) {
      writeMetric("F1-Score", meter.f1Score(labelId));
      writeMetric("Precision", meter.precision(labelId));
      writeMetric("Recall", meter.recall(labelId));
      std::cout << " " << dict->getLabel(labelId) << std::endl;
    }
  }
  meter.writeGeneralMetrics(std::cout, k);

  exit(0);
}

void printPredictions(
    const std::vector<std::pair<real, std::string>>& predictions,
    bool printProb,
    bool multiline) {
  bool first = true;
  for (const auto& prediction : predictions) {
    if (!first && !multiline) {
      std::cout << " ";
    }
    first = false;
    std::cout << prediction.second;
    if (printProb) {
      std::cout << " " << prediction.first;
    }
    if (multiline) {
      std::cout << std::endl;
    }
  }
  if (!multiline) {
    std::cout << std::endl;
  }
}

// Predicts batches of lines on several threads and prints them in input order.
void predictParallel(
    const ByteFast& model,
    std::istream& in,
    int32_t k,
    real threshold,
    bool printProb,
    int32_t threads) {
  const size_t batchSize = 1 << 16;
  std::vector<std::string> lines;
  std::vector<std::string> outputs(batchSize);
  bool more = true;
  while (more) {
    lines.clear();
    std::string line;
    while (lines.size() < batchSize && std::getline(in, line)) {
      // Keep the end-of-line token that the single-threaded path sees.
      if (!in.eof()) {
        line.push_back('\n');
      }
      lines.push_back(std::move(line));
    }
    more = lines.size() == batchSize;
    std::vector<std::thread> workers;
    for (int32_t t = 0; t < threads; t++) {
      workers.emplace_back([&, t]() {
        std::vector<std::pair<real, std::string>> predictions;
        for (size_t i = t; i < lines.size(); i += threads) {
          std::istringstream iss(lines[i]);
          predictions.clear();
          model.predictLine(iss, predictions, k, threshold);
          std::ostringstream out;
          bool first = true;
          for (const auto& prediction : predictions) {
            out << (first ? "" : " ") << prediction.second;
            if (printProb) {
              out << " " << prediction.first;
            }
            first = false;
          }
          outputs[i] = out.str();
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    for (size_t i = 0; i < lines.size(); i++) {
      std::cout << outputs[i] << '\n';
    }
    std::cout.flush();
  }
}

void predict(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 7) {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  real threshold = 0.0;
  int32_t threads = 1;
  if (args.size() > 4) {
    k = std::stoi(args[4]);
    if (args.size() >= 6) {
      threshold = std::stof(args[5]);
    }
    if (args.size() == 7) {
      threads = std::stoi(args[6]);
    }
  }
  if (threads < 1) {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }

  bool printProb = args[1] == "predict-prob";
  ByteFast fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::ifstream ifs;
  std::string infile(args[3]);
  bool inputIsStdIn = infile == "-";
  if (!inputIsStdIn) {
    ifs.open(infile);
    if (!inputIsStdIn && !ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::istream& in = inputIsStdIn ? std::cin : ifs;
  if (threads > 1) {
    predictParallel(fasttext, in, k, threshold, printProb, threads);
  } else {
    std::vector<std::pair<real, std::string>> predictions;
    while (fasttext.predictLine(in, predictions, k, threshold)) {
      printPredictions(predictions, printProb, false);
    }
  }
  if (ifs.is_open()) {
    ifs.close();
  }

  exit(0);
}

void train(const std::vector<std::string> args) {
  Args a = Args();
  a.parseArgs(args);
  std::shared_ptr<ByteFast> fasttext = std::make_shared<ByteFast>();
  std::string outputFileName = a.output + ".bin";
  std::ofstream ofs(outputFileName);
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        outputFileName + " cannot be opened for saving.");
  }
  ofs.close();
  fasttext->train(a);
  fasttext->saveModel(outputFileName);
}

int main(int argc, char** argv) {
  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(args[1]);
  if (command == "supervised") {
    train(args);
  } else if (command == "test" || command == "test-label") {
    test(args);
  } else if (command == "quantize") {
    quantize(args);
  } else if (command == "predict" || command == "predict-prob") {
    predict(args);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  return 0;
}
