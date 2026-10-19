/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace bytefast {

class SafeTensorsReader;
class SafeTensorsWriter;

// The value matches the loss field of model files from before safetensors.
enum class loss_name : int { softmax = 3 };

class Args {
 protected:
  std::string boolToString(bool) const;
  std::unordered_set<std::string> manualArgs_;

 public:
  Args();
  std::string input;
  std::string output;
  double lr;
  int lrUpdateRate;
  int dim;
  int epoch;
  int minCount;
  int minCountLabel;
  int wordNgrams;
  loss_name loss;
  int bucket;
  int minn;
  int maxn;
  // Text-level features over the whole input line (ByteFast).
  int minb;
  int maxb;
  int minc;
  int maxc;
  int maxBytes;
  std::string wordlist;
  int maxVocab;
  // Training only (not stored in models).
  std::string inputModel;
  double cropProb;
  int cropMin;
  int cropMax;
  std::string cropUnit;
  double normProb;
  double stripProb;
  bool saveEpochs;
  std::string validation;
  int thread;
  std::string label;
  int verbose;
  int seed;

  bool qout;
  bool retrain;
  bool qnorm;
  size_t cutoff;
  size_t dsub;

  void parseArgs(const std::vector<std::string>& args);
  void printHelp();
  void printBasicHelp();
  void printDictionaryHelp();
  void printTrainingHelp();
  void printQuantizationHelp();
  void save(SafeTensorsWriter&) const;
  void load(SafeTensorsReader&);
  // Read the settings of a model file from before safetensors.
  void load(std::istream&);
  void loadTextFeatures(std::istream&);
  bool usesTextFeatures() const;
  bool usesHashedFeatures() const;
  void validate() const;
  bool isManual(const std::string& argName) const;
  void setManual(const std::string& argName);
  std::string lossToString(loss_name) const;
  static loss_name lossFromString(const std::string&);
};
} // namespace bytefast
