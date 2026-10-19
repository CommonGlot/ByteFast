/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "args.h"
#include "safetensors.h"

#include <cstdlib>
#include <cstdint>

#include <iostream>
#include <stdexcept>
#include <string>

namespace bytefast {

Args::Args() {
  lr = 0.1;
  dim = 100;
  epoch = 5;
  minCount = 1;
  loss = loss_name::softmax;
  bucket = 2000000;
  minn = 0;
  maxn = 0;
  minb = 0;
  maxb = 0;
  maxBytes = 0;
  wordlist = "";
  lexicon = "";
  lexThreshold = 0.5;
  lexMinWords = 2;
  lexTopK = 5;
  cropProb = 0.0;
  cropMin = 1;
  cropMax = 0;
  normProb = 0.0;
  stripProb = 0.0;
  thread = 12;
  label = "__label__";
  verbose = 2;
  seed = 0;
}

std::string Args::lossToString(loss_name ln) const {
  (void)ln;
  return "softmax";
}

loss_name Args::lossFromString(const std::string& name) {
  if (name == "softmax") {
    return loss_name::softmax;
  } else if (name == "one-vs-all" || name == "ova" || name == "hs" ||
             name == "ns") {
    throw std::invalid_argument(
        "-loss " + name + " is no longer supported (softmax only)");
  }
  throw std::invalid_argument("Unknown loss: " + name);
}

void Args::parseArgs(const std::vector<std::string>& args) {
  for (int ai = 2; ai < args.size(); ai += 2) {
    if (args[ai][0] != '-') {
      std::cerr << "Provided argument without a dash! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    try {
      if (args[ai] == "-h") {
        std::cerr << "Here is the help! Usage:" << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      } else if (args[ai] == "-input") {
        input = std::string(args.at(ai + 1));
      } else if (args[ai] == "-output") {
        output = std::string(args.at(ai + 1));
      } else if (args[ai] == "-lr") {
        lr = std::stof(args.at(ai + 1));
      } else if (args[ai] == "-dim") {
        dim = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-epoch") {
        epoch = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minCount") {
        minCount = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-loss") {
        loss = lossFromString(args.at(ai + 1));
      } else if (args[ai] == "-bucket") {
        bucket = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minn") {
        minn = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-maxn") {
        maxn = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minb") {
        minb = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-maxb") {
        maxb = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-maxBytes") {
        maxBytes = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-wordlist") {
        wordlist = std::string(args.at(ai + 1));
      } else if (args[ai] == "-lexicon") {
        lexicon = std::string(args.at(ai + 1));
      } else if (args[ai] == "-lexThreshold") {
        lexThreshold = std::stod(args.at(ai + 1));
      } else if (args[ai] == "-lexMinWords") {
        lexMinWords = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-lexTopK") {
        lexTopK = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-cropProb") {
        cropProb = std::stod(args.at(ai + 1));
      } else if (args[ai] == "-cropMin") {
        cropMin = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-cropMax") {
        cropMax = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-normProb") {
        normProb = std::stod(args.at(ai + 1));
      } else if (args[ai] == "-stripProb") {
        stripProb = std::stod(args.at(ai + 1));
      } else if (args[ai] == "-thread") {
        thread = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-label") {
        label = std::string(args.at(ai + 1));
      } else if (args[ai] == "-verbose") {
        verbose = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-seed") {
        seed = std::stoi(args.at(ai + 1));
      } else {
        std::cerr << "Unknown argument: " << args[ai] << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } catch (std::out_of_range) {
      std::cerr << args[ai] << " is missing an argument" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    } catch (const std::invalid_argument& e) {
      std::cerr << args[ai] << ": invalid value (" << e.what() << ")"
                << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
  }
  if (input.empty() || output.empty()) {
    std::cerr << "Empty input or output path." << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
  try {
    validate();
  } catch (const std::invalid_argument& e) {
    std::cerr << e.what() << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
  if (!usesHashedFeatures()) {
    bucket = 0;
  }
}

bool Args::usesTextFeatures() const {
  return maxb > 0 || maxBytes > 0;
}

bool Args::usesHashedFeatures() const {
  return maxn > 0 || maxb > 0;
}

void Args::validate() const {
  if (maxb < 0 || minb < 0 || (maxb > 0 && (minb < 1 || minb > maxb))) {
    throw std::invalid_argument("-minb/-maxb need 1 <= minb <= maxb (maxb 0 = off)");
  }
  if (maxBytes < 0) {
    throw std::invalid_argument("-maxBytes must be >= 0 (0 = no limit)");
  }
  if (!(lexThreshold >= 0.0 && lexThreshold <= 1.0)) {
    throw std::invalid_argument("-lexThreshold must be in [0, 1]");
  }
  if (lexMinWords < 1 || lexTopK < 1) {
    throw std::invalid_argument("-lexMinWords and -lexTopK must be >= 1");
  }
  if (maxb > 0 && bucket <= 0) {
    throw std::invalid_argument("-minb/-maxb need -bucket > 0");
  }
  if (!(normProb >= 0.0 && normProb <= 1.0)) {
    throw std::invalid_argument("-normProb must be in [0, 1]");
  }
  if (!(stripProb >= 0.0 && stripProb <= 1.0)) {
    throw std::invalid_argument("-stripProb must be in [0, 1]");
  }
  if (!(cropProb >= 0.0 && cropProb <= 1.0)) {
    throw std::invalid_argument("-cropProb must be in [0, 1]");
  }
  if (cropProb > 0.0 && (cropMin < 1 || cropMax < cropMin)) {
    throw std::invalid_argument("-cropProb needs 1 <= cropMin <= cropMax");
  }
}

void Args::printHelp() {
  printBasicHelp();
  printDictionaryHelp();
  printTrainingHelp();
}

void Args::printBasicHelp() {
  std::cerr << "\nThe following arguments are mandatory:\n"
            << "  -input              training file path\n"
            << "  -output             output file path\n"
            << "\nThe following arguments are optional:\n"
            << "  -verbose            verbosity level [" << verbose << "]\n";
}

void Args::printDictionaryHelp() {
  std::cerr << "\nThe following arguments for the dictionary are optional:\n"
            << "  -minCount           minimal number of word occurences ["
            << minCount << "]\n"
            << "  -bucket             number of buckets [" << bucket << "]\n"
            << "  -minn               min length of char ngram [" << minn
            << "]\n"
            << "  -maxn               max length of char ngram [" << maxn
            << "]\n"
            << "  -minb               min length of text byte ngram [" << minb
            << "]\n"
            << "  -maxb               max length of text byte ngram, 0 = off ["
            << maxb << "]\n"
            << "  -maxBytes           bytes of text used for text ngrams, 0 = all ["
            << maxBytes << "]\n"
            << "  -wordlist           file with the allowed vocabulary, one word per line ["
            << wordlist << "]\n"
            << "  -lexicon            file of LABEL<TAB>WORD lines: a prediction keeps a\n"
            << "                      label only if its words cover -lexThreshold of the\n"
            << "                      line, else the next of the top -lexTopK that does ["
            << lexicon << "]\n"
            << "  -lexThreshold       share of the line's words in the label's lexicon ["
            << lexThreshold << "]\n"
            << "  -lexMinWords        lines with fewer words are not checked ["
            << lexMinWords << "]\n"
            << "  -lexTopK            labels tried for the check [" << lexTopK << "]\n"
            << "  -cropProb           probability of training on a random span of a line ["
            << cropProb << "]\n"
            << "  -cropMin            min span length in characters [" << cropMin << "]\n"
            << "  -cropMax            max span length in characters [" << cropMax << "]\n"
            << "  -normProb           probability of training on the CommonLID-style\n"
            << "                      normalized line (lowercase, no digits/punctuation) ["
            << normProb << "]\n"
            << "  -stripProb          probability of training on the line without "
               "optional\n"
            << "                      diacritics (tones, accents, Arabic/Hebrew "
               "vowel marks) ["
            << stripProb << "]\n"
            << "  -label              labels prefix [" << label << "]\n";
}

void Args::printTrainingHelp() {
  std::cerr
      << "\nThe following arguments for training are optional:\n"
      << "  -lr                 learning rate [" << lr << "]\n"
      << "  -dim                size of word vectors [" << dim << "]\n"
      << "  -epoch              number of epochs [" << epoch << "]\n"
      << "  -loss               loss function {softmax} ["
      << lossToString(loss) << "]\n"
      << "  -thread             number of threads (set to 1 to ensure "
         "reproducible results) ["
      << thread << "]\n"
      << "  -seed               random generator seed  [" << seed << "]\n";
}

// Model settings are stored as safetensors metadata strings.
void Args::save(SafeTensorsWriter& out) const {
  const std::vector<std::pair<std::string, std::string>> fields = {
      {"dim", std::to_string(dim)},
      {"epoch", std::to_string(epoch)},
      {"minCount", std::to_string(minCount)},
      {"loss", lossToString(loss)},
      {"bucket", std::to_string(bucket)},
      {"minn", std::to_string(minn)},
      {"maxn", std::to_string(maxn)},
      {"minb", std::to_string(minb)},
      {"maxb", std::to_string(maxb)},
      {"maxBytes", std::to_string(maxBytes)},
      {"label", label},
      {"lexThreshold", std::to_string(lexThreshold)},
      {"lexMinWords", std::to_string(lexMinWords)},
      {"lexTopK", std::to_string(lexTopK)}};
  for (const auto& field : fields) {
    out.metadata("args." + field.first, field.second);
  }
}

void Args::load(SafeTensorsReader& in) {
  auto get = [&](const std::string& name) {
    return std::stoi(in.metadata("args." + name));
  };
  dim = get("dim");
  epoch = get("epoch");
  minCount = get("minCount");
  loss = lossFromString(in.metadata("args.loss"));
  bucket = get("bucket");
  minn = get("minn");
  maxn = get("maxn");
  minb = get("minb");
  maxb = get("maxb");
  maxBytes = get("maxBytes");
  // Options of earlier versions that were removed (wordNgrams 1 is off).
  for (const char* removed : {"maxc", "hashWords", "wordNgrams"}) {
    const int off = std::string(removed) == "wordNgrams" ? 1 : 0;
    if (in.hasMetadata(std::string("args.") + removed) && get(removed) != off) {
      throw std::invalid_argument(
          std::string("This model uses -") + removed +
          ", which ByteFast no longer supports");
    }
  }
  label = in.metadata("args.label");
  // Models saved before the lexicon check keep the defaults.
  if (in.hasMetadata("args.lexThreshold")) {
    lexThreshold = std::stod(in.metadata("args.lexThreshold"));
    lexMinWords = get("lexMinWords");
    lexTopK = get("lexTopK");
  }
  validate();
}

} // namespace bytefast
