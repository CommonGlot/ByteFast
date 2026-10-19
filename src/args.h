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
#include <vector>

namespace bytefast {

class SafeTensorsReader;
class SafeTensorsWriter;

// Softmax is the only loss; the name is kept in model metadata.
enum class loss_name : int { softmax = 3 };

class Args {
 public:
  Args();
  std::string input;
  std::string output;
  double lr;
  int dim;
  int epoch;
  int minCount;
  loss_name loss;
  int bucket;
  int minn;
  int maxn;
  // Byte n-grams over the whole input line (ByteFast).
  int minb;
  int maxb;
  int maxBytes;
  std::string wordlist;
  int maxVocab;
  // Percent-threshold lexicon check at prediction (stored in models; the
  // words come from -lexicon at training).
  std::string lexicon;
  double lexThreshold;
  int lexMinWords;
  int lexTopK;
  // Training only (not stored in models).
  double cropProb;
  int cropMin;
  int cropMax;
  double normProb;
  double stripProb;
  // LABEL<TAB>WEIGHT lines: a line is trained WEIGHT times in expectation.
  std::string labelWeights;
  int thread;
  std::string label;
  int verbose;
  int seed;

  void parseArgs(const std::vector<std::string>& args);
  // Sets the options of a named recipe; options given after it override.
  void applyPreset(const std::string& name);
  void printHelp();
  void printBasicHelp();
  void printDictionaryHelp();
  void printTrainingHelp();
  void save(SafeTensorsWriter&) const;
  void load(SafeTensorsReader&);
  bool usesTextFeatures() const;
  bool usesHashedFeatures() const;
  void validate() const;
  std::string lossToString(loss_name) const;
  static loss_name lossFromString(const std::string&);
};
} // namespace bytefast
