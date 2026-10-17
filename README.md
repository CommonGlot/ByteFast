# ByteFast

ByteFast is a general-purpose text classification library. Train it on any
labels: topics, sentiment, intent, document types, spam, or languages. It is
built on the [fastText](https://github.com/facebookresearch/fastText) training
code and keeps what makes fastText practical: a linear model, multithreaded
training on CPUs, small quantized models and fast prediction.

ByteFast adds input features that fastText lacks, and every feature can be
switched on or off independently:

- **Bytes**: byte n-grams over the whole text, crossing word boundaries. No
  tokenizer, and nothing is out of vocabulary.
- **Characters**: UTF-8 character n-grams over the whole text, so an n-gram
  has the same length in characters in every script.
- **Words**: a vocabulary built from the training data, a vocabulary you
  supply as a word list, and hashed vectors for words outside it.
- **Normalization**: ASCII or Unicode lowercasing, and a limit on how much
  of each text the n-grams read.

The original fastText features (character n-grams within words, word n-grams,
pretrained vectors) and losses (softmax, one-vs-all, hierarchical softmax,
negative sampling) are unchanged. ByteFast reads fastText `.bin` and `.ftz`
models, and writes models that use only fastText features in the fastText
format, so upstream `fasttext` can load them.

## Installation

```bash
git clone https://github.com/CommonGlot/ByteFast.git
cd ByteFast
make opt                      # builds ./bytefast
python -m pip install .       # Python package "bytefast"
```

The Python package is named `bytefast` and can be installed next to the
upstream `fasttext` package.

## Training a classifier

Training data has one example per line, with one or more labels first:

```text
__label__sports The team won its third consecutive match.
__label__technology The processor uses a new chip architecture.
__label__finance The company reported its quarterly earnings.
```

Command line:

```bash
./bytefast supervised -input train.txt -output model \
    -minb 2 -maxb 5 -bucket 1000000 -dim 64 -epoch 5 -lr 0.5 -thread 8
./bytefast test model.bin valid.txt
echo "The striker scored twice" | ./bytefast predict-prob model.bin - 3
```

Python:

```python
import bytefast

model = bytefast.train_supervised(
    input="train.txt",
    minb=2, maxb=5,          # byte n-grams over the whole text
    minCount=5,              # word vectors for words seen at least 5 times
    fold="unicode",          # lowercase before extracting features
    bucket=1_000_000, dim=64, epoch=5, lr=0.5, thread=8,
)
labels, probabilities = model.predict("The striker scored twice", k=3)
model.save_model("model.bin")
model = bytefast.load_model("model.bin")
```

## Features and options

Each example is represented by the average of its feature vectors; a linear
layer maps that average to the labels.

| Feature | Options | Default (supervised) |
| --- | --- | --- |
| Word vectors | `-minCount N`, `-wordlist FILE`, `-maxVocab N` | on, `minCount 1` |
| Hashed out-of-vocabulary words | `-hashWords` | off |
| Character n-grams inside words | `-minn N -maxn N` | off (`maxn 0`) |
| Word n-grams | `-wordNgrams N` | 1 (off) |
| Byte n-grams over the text | `-minb N -maxb N` | off (`maxb 0`) |
| Character n-grams over the text | `-minc N -maxc N` | off (`maxc 0`) |
| Bytes of text read by text n-grams | `-maxBytes N` | 0 (all) |
| Case folding | `-fold none\|ascii\|unicode` | `none` |
| Position bins for text n-grams | `-posBins K` | 0 (off) |
| Scope of text n-grams | `-textScope line\|word` | `line` |

- **Text n-grams** read the words of a line joined by single spaces, with one
  space added at each end. So an n-gram can span the end of one word and the
  start of the next, and the first and last words get the same boundary
  n-grams as the others. With `-textScope word`, text n-grams are taken
  inside each word only, with no padding, so none of them contains a space.
- **`-wordlist FILE`** restricts the vocabulary to the first field of each line
  in `FILE`, folded like the input. Words outside it can still contribute
  through `-hashWords`, subwords or text n-grams.
- **Disabling word vectors:** set `-minCount` above the corpus size.
- **Hashed features** (`-hashWords`, subwords, word n-grams and text n-grams)
  share the `-bucket` table.

These training options help with inputs shorter or noisier than the training
data and with staged training:

- `-cropProb P -cropMin N -cropMax M -cropUnit char|word|byte`: with
  probability `P`, a training line is replaced by a random contiguous span of
  `N`–`M` characters, words or bytes (prediction is unaffected). Byte crops
  never split a UTF-8 character.
- `-normProb P`: with probability `P`, a training line is normalized like the
  OpenLID-v2 / CommonLID input cleaner: Unicode lowercase, then digits and
  every character matching `[^\p{Word}\p{Zs}]` removed. Lines are drawn
  again every epoch, so the model sees both the original (cased, punctuated)
  and the normalized form of the data. It runs after the crop. The removed set
  is generated from the `regex` module by `tools/gen_norm_table.py`; on 50,000
  CommonLID lines the output equals the Python cleaner's.
- `-inputModel model.bin`: continue training a saved supervised model on new
  data, for example a mix of short and long lines. The model's features and
  architecture are kept; `-lr`, `-epoch`, `-thread`, `-seed` and the crop
  and `-normProb` options apply to the new run.

`./bytefast supervised -h` lists every option. In Python the same names are
keyword arguments of `bytefast.train_supervised`, with snake_case aliases
such as `hash_words`, `max_bytes` and `max_vocab`.

## Model files

| File | Reader |
| --- | --- |
| ByteFast model with text features (`BFCM` header) | `bytefast` only |
| ByteFast model with only fastText features | `bytefast` and upstream `fasttext` |
| Original fastText `.bin` / `.ftz` | `bytefast` and upstream `fasttext` |
| Prototype byte models and bundles | `bytefast.proto` |

`bytefast.load_model()` chooses the reader from the file header.
`./bytefast quantize` works for every model type.

## Prototype byte classifier

`bytefast.proto` and the `bytefast-proto`, `bytefast-lexical`,
`bytefast-calibrated` and `bytefast-wordemb` executables contain the earlier
byte-only classifier. It has a statistical word-to-class sidecar and learned
fusion weights, and is kept as a baseline for the ablations in
[docs/architecture.md](docs/architecture.md). See
[docs/bytefast-prototype.md](docs/bytefast-prototype.md).

```python
model = bytefast.proto.train_supervised(
    "train.txt", output="proto.bin", dim=32, bucket=16384, epoch=10,
    lexical=True, lexical_vocab=200000,
)
```

## Project status

ByteFast is under development. The design, the ablations that decide the
defaults, and the evaluation protocol are in
[docs/architecture.md](docs/architecture.md). Benchmarks and their workflows
are in [benchmarks](benchmarks/) and [.github/workflows](.github/workflows/).

## License and history

ByteFast is a fork of fastText by Facebook, Inc., renamed and extended by the
CommonGlot contributors. The original copyright notices are preserved under
the [MIT license](LICENSE). The original fastText README is kept in
[docs/original-fasttext.md](docs/original-fasttext.md).
