# ByteFast

ByteFast is a general-purpose text classification library. Train it on any
labels: topics, sentiment, intent, document types, spam, or languages. It is
built on the [fastText](https://github.com/facebookresearch/fastText) training
code and keeps what makes fastText practical: a linear model, multithreaded
training on CPUs and fast prediction.

ByteFast adds what fastText lacks, and keeps only what ablations showed to be
useful ([docs/architecture.md](docs/architecture.md)):

- **Bytes**: byte n-grams over the whole text, crossing word boundaries. No
  tokenizer, and nothing is out of vocabulary.
- **Words**: a vocabulary built from the training data, capped or given as a
  word list.
- **Training on harder inputs**: random character spans, CommonLID-style
  normalized lines and lines without optional diacritics.
- **Model files** in the [safetensors](https://github.com/huggingface/safetensors)
  format, and bindings for Python, Rust, JavaScript and C.

fastText's character n-grams within words and word n-grams are available
too. ByteFast is a classifier only: it does not train word vectors, and it
does not read fastText model files.

## Installation

```bash
git clone https://github.com/CommonGlot/ByteFast.git
cd ByteFast
make opt                      # builds ./bytefast
python -m pip install .       # Python package "bytefast"
```

The Python API is described in [bindings/python/README.md](bindings/python/README.md).
Prediction is also available from other languages, all through one C API
([src/bytefast_c.h](src/bytefast_c.h); `make lib` builds `libbytefast.a`):

| Language | Package | Source |
| --- | --- | --- |
| Python | `pip install bytefast` (training too) | [bindings/python](bindings/python/README.md) |
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
| Byte n-grams over the text | `-minb N -maxb N` | off (`maxb 0`) |
| Bytes of text read by byte n-grams | `-maxBytes N` | 0 (all) |

- **Byte n-grams** read the words of a line joined by single spaces, with one
  space added at each end. So an n-gram can span the end of one word and the
  start of the next, and the first and last words get the same boundary
  n-grams as the others.
- **`-wordlist FILE`** restricts the vocabulary to the first field of each line
  in `FILE`; entries match words exactly. Words outside it can still contribute
  through subwords or byte n-grams.
- **Disabling word vectors:** set `-minCount` above the corpus size.
- **Hashed features** (subwords and byte n-grams) share the `-bucket` table.

These training options help with inputs shorter or noisier than the training
data:

- `-cropProb P -cropMin N -cropMax M`: with probability `P`, a training line
  is replaced by a random contiguous span of `N`–`M` characters (prediction is
  unaffected).
- `-normProb P`: with probability `P`, a training line is normalized like the
  OpenLID-v2 / CommonLID input cleaner: Unicode lowercase, then digits and
  every character matching `[^\p{Word}\p{Zs}]` removed. Lines are drawn
  again every epoch, so the model sees both the original (cased, punctuated)
  and the normalized form of the data. It runs after the crop. The removed set
  is generated from the `regex` module by `tools/gen_norm_table.py`; on 50,000
  CommonLID lines the output equals the Python cleaner's.
- `-stripProb P`: with probability `P`, a training line loses its optional
  diacritics: tone and accent marks on Latin, Greek and Cyrillic letters
  (combining or precomposed, e.g. `à` → `a`, `ɛ́` → `ɛ`) and Arabic and Hebrew
  vowel marks. Marks that are part of a letter in other scripts (Indic vowel
  signs) are kept. The model then sees languages written both with and without
  tone marks. It runs after `-normProb`; the table is generated by
  `tools/gen_strip_table.py`. On UDHR it raised macro F1 from 0.815 to 0.824
  (2M GlotLID lines, `-stripProb 0.3`).

- `-labelWeights FILE`: `LABEL<TAB>WEIGHT` lines; each line of a label is
  trained `WEIGHT` times in expectation (`floor(WEIGHT)` times, plus once
  more with the remaining fraction as probability), which resamples labels
  without rewriting the data. Labels not in the file keep weight 1.
- `-preset lid` / `-preset lid-accurate`: the language-ID settings of
  [docs/architecture.md](docs/architecture.md) (bytes 2-5, the 20k most
  frequent words, crops, normalized and stripped lines, 3 epochs at lr 0.8;
  `lid-accurate` adds subwords). Options given with a preset override it.
  [recipes/lid](recipes/lid) builds the whole model from a corpus.

### Lexicon check

`-lexicon FILE` stores per-label word lists in the model, for the
percent-threshold check of [Caswell et al. (2020)](https://aclanthology.org/2020.coling-main.579/).
`FILE` has `LABEL<TAB>WORD` lines (further columns are ignored; the label
prefix is optional). At prediction, a line's words are normalized like
`-normProb` does, and the top label is kept only if at least `-lexThreshold`
(0.5) of them are in its list. Otherwise the first of the top `-lexTopK` (5)
labels that passes comes first, and if none passes the order is unchanged.
Lines with fewer than `-lexMinWords` (2) words, and labels without a list,
always pass. The lists are a Bloom filter (10 bits per word, about 1% of
unknown words counted as known), so 8.4M words take about 10 MB. The check
only changes prediction; `bytefast predict` takes a threshold after the
thread count (0 turns the check off), and Python has
`model.set_lexicon_threshold`. On a 2,000-label language-ID model with lists
built from its training data, it raised CommonLID macro F1 from 0.601 to
0.607 and its 5-word view from 0.535 to 0.547.

### Experts

`bytefast add-expert MODEL EXPERT OUTPUT [GATE]` stores `EXPERT`, a model
trained on the lines of a group of close labels, inside `MODEL`. When the top
label of a line is one of the expert's labels and its probability is below
`GATE` (0.8), the expert picks the label, which keeps the main model's
probability for it. Several experts can be added (their labels must not
overlap); they are saved as `expertN.*` tensors in the same file, and the C
API, Python, Rust and JS use them. In Python: `model.add_expert(path,
gate=0.8)`. On FLORES the five language-ID experts raised macro F1 from 0.930
to 0.939.

`./bytefast supervised -h` lists every option. In Python the same names are
keyword arguments of `bytefast.train_supervised`, with snake_case aliases
such as `max_bytes`, `max_vocab`, `label_weights` and `lex_threshold`, and
`preset=` selects a preset.

## Model files

Models are [safetensors](https://github.com/huggingface/safetensors) files:
the input table and output layer are F32 tensors (`input`, `output`), the
vocabulary is in `vocab.strings`, `vocab.counts` and `vocab.types`, and the
settings are metadata strings (`args.dim`, `args.minb`, ...). Any safetensors
reader can open them. Training writes `OUTPUT.safetensors`.

## Project status

ByteFast is under development. The design, the ablations that decide the
defaults, and the evaluation protocol are in
[docs/architecture.md](docs/architecture.md). Benchmarks and their workflows
are in [benchmarks](benchmarks/) and [.github/workflows](.github/workflows/).

## License and history

ByteFast is a fork of fastText by Facebook, Inc., renamed and extended by the
CommonGlot contributors. The original copyright notices are preserved under
the [MIT license](LICENSE).
