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
  minCountLabel = 0;
  wordNgrams = 1;
  loss = loss_name::softmax;
  bucket = 2000000;
  minn = 0;
  maxn = 0;
  minb = 0;
  maxb = 0;
  minc = 0;
  maxc = 0;
  maxBytes = 0;
  wordlist = "";
  maxVocab = 0;
  inputModel = "";
  cropProb = 0.0;
  cropMin = 1;
  cropMax = 0;
  cropUnit = "char";
  normProb = 0.0;
  stripProb = 0.0;
  saveEpochs = false;
  validation = "";
  thread = 12;
  lrUpdateRate = 100;
  label = "__label__";
  verbose = 2;
  seed = 0;

  qout = false;
  retrain = false;
  qnorm = false;
  cutoff = 0;
  dsub = 2;
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

std::string Args::boolToString(bool b) const {
  if (b) {
    return "true";
  } else {
    return "false";
  }
}

void Args::parseArgs(const std::vector<std::string>& args) {
  for (int ai = 2; ai < args.size(); ai += 2) {
    if (args[ai][0] != '-') {
      std::cerr << "Provided argument without a dash! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    try {
      setManual(args[ai].substr(1));

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
      } else if (args[ai] == "-lrUpdateRate") {
        lrUpdateRate = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-dim") {
        dim = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-epoch") {
        epoch = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minCount") {
        minCount = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minCountLabel") {
        minCountLabel = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-wordNgrams") {
        wordNgrams = std::stoi(args.at(ai + 1));
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
      } else if (args[ai] == "-minc") {
        minc = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-maxc") {
        maxc = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-maxBytes") {
        maxBytes = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-wordlist") {
        wordlist = std::string(args.at(ai + 1));
      } else if (args[ai] == "-maxVocab") {
        maxVocab = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-inputModel") {
        inputModel = std::string(args.at(ai + 1));
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
      } else if (args[ai] == "-saveEpochs") {
        saveEpochs = true;
        ai--;
      } else if (args[ai] == "-validation") {
        validation = std::string(args.at(ai + 1));
      } else if (args[ai] == "-cropUnit") {
        cropUnit = std::string(args.at(ai + 1));
      } else if (args[ai] == "-thread") {
        thread = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-label") {
        label = std::string(args.at(ai + 1));
      } else if (args[ai] == "-verbose") {
        verbose = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-seed") {
        seed = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-qnorm") {
        qnorm = true;
        ai--;
      } else if (args[ai] == "-retrain") {
        retrain = true;
        ai--;
      } else if (args[ai] == "-qout") {
        qout = true;
        ai--;
      } else if (args[ai] == "-cutoff") {
        cutoff = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-dsub") {
        dsub = std::stoi(args.at(ai + 1));
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
  return maxb > 0 || maxc > 0 || maxBytes > 0;
}

bool Args::usesHashedFeatures() const {
  return wordNgrams > 1 || maxn > 0 || maxb > 0 || maxc > 0;
}

void Args::validate() const {
  if (maxb < 0 || minb < 0 || (maxb > 0 && (minb < 1 || minb > maxb))) {
    throw std::invalid_argument("-minb/-maxb need 1 <= minb <= maxb (maxb 0 = off)");
  }
  if (maxc < 0 || minc < 0 || (maxc > 0 && (minc < 1 || minc > maxc))) {
    throw std::invalid_argument("-minc/-maxc need 1 <= minc <= maxc (maxc 0 = off)");
  }
  if (maxBytes < 0) {
    throw std::invalid_argument("-maxBytes must be >= 0 (0 = no limit)");
  }
  if (maxVocab < 0) {
    throw std::invalid_argument("-maxVocab must be >= 0 (0 = no limit)");
  }
  if ((maxb > 0 || maxc > 0) && bucket <= 0) {
    throw std::invalid_argument("-minb/-maxb and -minc/-maxc need -bucket > 0");
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
  if (cropUnit != "char" && cropUnit != "word" && cropUnit != "byte") {
    throw std::invalid_argument("-cropUnit must be char, word or byte");
  }
}

void Args::printHelp() {
  printBasicHelp();
  printDictionaryHelp();
  printTrainingHelp();
  printQuantizationHelp();
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
            << "  -minCountLabel      minimal number of label occurences ["
            << minCountLabel << "]\n"
            << "  -wordNgrams         max length of word ngram [" << wordNgrams
            << "]\n"
            << "  -bucket             number of buckets [" << bucket << "]\n"
            << "  -minn               min length of char ngram [" << minn
            << "]\n"
            << "  -maxn               max length of char ngram [" << maxn
            << "]\n"
            << "  -minb               min length of text byte ngram [" << minb
            << "]\n"
            << "  -maxb               max length of text byte ngram, 0 = off ["
            << maxb << "]\n"
            << "  -minc               min length of text character ngram ["
            << minc << "]\n"
            << "  -maxc               max length of text character ngram, 0 = off ["
            << maxc << "]\n"
            << "  -maxBytes           bytes of text used for text ngrams, 0 = all ["
            << maxBytes << "]\n"
            << "  -wordlist           file with the allowed vocabulary, one word per line ["
            << wordlist << "]\n"
            << "  -maxVocab           keep at most this many words, 0 = no limit ["
            << maxVocab << "]\n"
            << "  -cropProb           probability of training on a random span of a line ["
            << cropProb << "]\n"
            << "  -cropMin            min span length [" << cropMin << "]\n"
            << "  -cropMax            max span length [" << cropMax << "]\n"
            << "  -cropUnit           span unit: char, word or byte [" << cropUnit << "]\n"
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
      << "  -lrUpdateRate       change the rate of updates for the learning "
         "rate ["
      << lrUpdateRate << "]\n"
      << "  -dim                size of word vectors [" << dim << "]\n"
      << "  -epoch              number of epochs [" << epoch << "]\n"
      << "  -loss               loss function {softmax} ["
      << lossToString(loss) << "]\n"
      << "  -thread             number of threads (set to 1 to ensure "
         "reproducible results) ["
      << thread << "]\n"
      << "  -inputModel         continue training this supervised model; its "
         "features\n"
      << "                      and architecture are kept, -lr/-epoch/-thread "
         "are new ["
      << inputModel << "]\n"
      << "  -saveEpochs         also save the model after every epoch but the "
         "last, as\n"
      << "                      OUTPUT.epN.bin (needs -thread > 1) ["
      << boolToString(saveEpochs) << "]\n"
      << "  -validation         print precision@1 on this labeled file after "
         "every epoch ["
      << validation << "]\n"
      << "  -seed               random generator seed  [" << seed << "]\n";
}

void Args::printQuantizationHelp() {
  std::cerr
      << "\nThe following arguments for quantization are optional:\n"
      << "  -cutoff             number of words and ngrams to retain ["
      << cutoff << "]\n"
      << "  -retrain            whether embeddings are finetuned if a cutoff "
         "is applied ["
      << boolToString(retrain) << "]\n"
      << "  -qnorm              whether the norm is quantized separately ["
      << boolToString(qnorm) << "]\n"
      << "  -qout               whether the classifier is quantized ["
      << boolToString(qout) << "]\n"
      << "  -dsub               size of each sub-vector [" << dsub << "]\n";
}

// Model settings are stored as safetensors metadata strings.
void Args::save(SafeTensorsWriter& out) const {
  const std::vector<std::pair<std::string, std::string>> fields = {
      {"dim", std::to_string(dim)},
      {"epoch", std::to_string(epoch)},
      {"minCount", std::to_string(minCount)},
      {"wordNgrams", std::to_string(wordNgrams)},
      {"loss", lossToString(loss)},
      {"bucket", std::to_string(bucket)},
      {"minn", std::to_string(minn)},
      {"maxn", std::to_string(maxn)},
      {"minb", std::to_string(minb)},
      {"maxb", std::to_string(maxb)},
      {"minc", std::to_string(minc)},
      {"maxc", std::to_string(maxc)},
      {"maxBytes", std::to_string(maxBytes)},
      {"maxVocab", std::to_string(maxVocab)},
      {"label", label},
      {"lrUpdateRate", std::to_string(lrUpdateRate)}};
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
  wordNgrams = get("wordNgrams");
  loss = lossFromString(in.metadata("args.loss"));
  bucket = get("bucket");
  minn = get("minn");
  maxn = get("maxn");
  minb = get("minb");
  maxb = get("maxb");
  minc = get("minc");
  maxc = get("maxc");
  maxBytes = get("maxBytes");
  if (in.hasMetadata("args.hashWords") && get("hashWords") != 0) {
    throw std::invalid_argument("-hashWords is no longer supported");
  }
  maxVocab = get("maxVocab");
  label = in.metadata("args.label");
  lrUpdateRate = get("lrUpdateRate");
  validate();
}

// Model files from before safetensors: the fastText fields (the window size,
// model type and subsampling threshold only applied to word vectors), then
// the text-feature block in BFCM files.
void Args::load(std::istream& in) {
  int ws, model, neg;
  double t;
  in.read((char*)&(dim), sizeof(int));
  in.read((char*)&(ws), sizeof(int));
  in.read((char*)&(epoch), sizeof(int));
  in.read((char*)&(minCount), sizeof(int));
  in.read((char*)&(neg), sizeof(int));
  (void)neg;
  in.read((char*)&(wordNgrams), sizeof(int));
  in.read((char*)&(loss), sizeof(loss_name));
  in.read((char*)&(model), sizeof(int));
  in.read((char*)&(bucket), sizeof(int));
  in.read((char*)&(minn), sizeof(int));
  in.read((char*)&(maxn), sizeof(int));
  in.read((char*)&(lrUpdateRate), sizeof(int));
  in.read((char*)&(t), sizeof(double));
  if (!in || model != 3) {
    throw std::invalid_argument("Not a supervised classifier model");
  }
}

void Args::loadTextFeatures(std::istream& in) {
  int32_t version, foldValue, hashWordsValue, posBinsValue = 0, scopeValue = 0;
  in.read((char*)&(version), sizeof(int32_t));
  if (!in || version < 1 || version > 3) {
    throw std::invalid_argument("Unsupported ByteFast text feature block");
  }
  in.read((char*)&(minb), sizeof(int));
  in.read((char*)&(maxb), sizeof(int));
  in.read((char*)&(minc), sizeof(int));
  in.read((char*)&(maxc), sizeof(int));
  in.read((char*)&(maxBytes), sizeof(int));
  in.read((char*)&(foldValue), sizeof(int32_t));
  in.read((char*)&(hashWordsValue), sizeof(int32_t));
  in.read((char*)&(maxVocab), sizeof(int));
  if (version >= 2) {
    in.read((char*)&(posBinsValue), sizeof(int32_t));
  }
  if (version >= 3) {
    in.read((char*)&(scopeValue), sizeof(int32_t));
  }
  if (!in || foldValue < 0 || foldValue > 2) {
    throw std::invalid_argument("Corrupt ByteFast text feature block");
  }
  if (foldValue != 0 || posBinsValue != 0 || scopeValue != 0) {
    throw std::invalid_argument(
        "This model uses -fold, -posBins or -textScope word, which ByteFast no "
        "longer supports");
  }
  if (hashWordsValue != 0) {
    throw std::invalid_argument("-hashWords is no longer supported");
  }
  validate();
}

bool Args::isManual(const std::string& argName) const {
  return (manualArgs_.count(argName) != 0);
}

void Args::setManual(const std::string& argName) {
  manualArgs_.emplace(argName);
}

} // namespace bytefast
