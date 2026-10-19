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
- **Limits**: how much of each text the n-grams read.

fastText's character n-grams within words and word n-grams are available
too. ByteFast is a classifier only: it does not train word vectors, and it
does not aim to read or write fastText model files.

## Installation

```bash
git clone https://github.com/CommonGlot/ByteFast.git
cd ByteFast
make opt                      # builds ./bytefast
python -m pip install .       # Python package "bytefast"
```

The Python API is described in [docs/python-module.md](docs/python-module.md).
Prediction is also available from other languages, all through one C API
([src/bytefast_c.h](src/bytefast_c.h); `make lib` builds `libbytefast.a`):

| Language | Package | Source |
| --- | --- | --- |
| Python | `pip install bytefast` (training too) | [python/](python/) |
| Rust | `bytefast` crate | [bindings/rust](bindings/rust/README.md) |
| JavaScript (Node, browsers) | `npm install bytefast` | [bindings/js](bindings/js/README.md) |
| C / C++ | `libbytefast.a` + `bytefast_c.h` | [src/bytefast_c.h](src/bytefast_c.h) |

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
./bytefast test model.safetensors valid.txt
echo "The striker scored twice" | ./bytefast predict-prob model.safetensors - 3
```

Python:

```python
import bytefast

model = bytefast.train_supervised(
    input="train.txt",
    minb=2, maxb=5,          # byte n-grams over the whole text
    minCount=5,              # word vectors for words seen at least 5 times
    bucket=1_000_000, dim=64, epoch=5, lr=0.5, thread=8,
)
labels, probabilities = model.predict("The striker scored twice", k=3)
model.save_model("model.safetensors")
model = bytefast.load_model("model.safetensors")
```

## Features and options

Each example is represented by the average of its feature vectors; a linear
layer maps that average to the labels.

| Feature | Options | Default (supervised) |
| --- | --- | --- |
| Word vectors | `-minCount N`, `-wordlist FILE`, `-maxVocab N` | on, `minCount 1` |
| Character n-grams inside words | `-minn N -maxn N` | off (`maxn 0`) |
| Word n-grams | `-wordNgrams N` | 1 (off) |
| Byte n-grams over the text | `-minb N -maxb N` | off (`maxb 0`) |
| Character n-grams over the text | `-minc N -maxc N` | off (`maxc 0`) |
| Bytes of text read by text n-grams | `-maxBytes N` | 0 (all) |

- **Text n-grams** read the words of a line joined by single spaces, with one
  space added at each end. So an n-gram can span the end of one word and the
  start of the next, and the first and last words get the same boundary
  n-grams as the others.
- **`-wordlist FILE`** restricts the vocabulary to the first field of each line
  in `FILE`; entries match words exactly. Words outside it can still contribute
  through subwords or text n-grams.
- **Disabling word vectors:** set `-minCount` above the corpus size.
- **Hashed features** (subwords, word n-grams and text n-grams) share the
  `-bucket` table.

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
- `-saveEpochs` and `-validation dev.txt`: after every epoch, save the model so
  far as `OUTPUT.epN.safetensors` and print precision@1 on a labeled file, to see when
  more epochs stop helping. The learning-rate schedule is unchanged (it still
  decays over all `-epoch` epochs); the snapshots are written while training
  continues and need `-thread` above 1.
- `-inputModel model.safetensors`: continue training a saved supervised model on new
  data, for example a mix of short and long lines. The model's features and
  architecture are kept; `-lr`, `-epoch`, `-thread`, `-seed` and the crop
  and `-normProb` options apply to the new run.

`./bytefast supervised -h` lists every option. In Python the same names are
keyword arguments of `bytefast.train_supervised`, with snake_case aliases
such as `max_bytes` and `max_vocab`.

## Model files

Models are [safetensors](https://github.com/huggingface/safetensors) files:
the input table and output layer are F32 tensors (`input`, `output`), the
vocabulary is in `vocab.strings`, `vocab.counts` and `vocab.types`, and the
settings are metadata strings (`args.dim`, `args.minb`, ...). Any safetensors
reader can open them. Training writes `OUTPUT.safetensors`, and
`./bytefast quantize` writes a compressed `OUTPUT.quant.safetensors`.
`./bytefast convert old.bin new.safetensors` rewrites a model in the earlier
`.bin`/`.ftz` format.

## Project status

ByteFast is under development. The design, the ablations that decide the
defaults, and the evaluation protocol are in
[docs/architecture.md](docs/architecture.md). Benchmarks and their workflows
are in [benchmarks](benchmarks/) and [.github/workflows](.github/workflows/).

## License and history

ByteFast is a fork of fastText by Facebook, Inc., renamed and extended by the
CommonGlot contributors. The original copyright notices are preserved under
the [MIT license](LICENSE).
