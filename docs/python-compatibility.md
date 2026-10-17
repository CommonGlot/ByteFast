# Python package and model-file compatibility

ByteFast is a general-purpose text classification library based on the
original [fastText](https://github.com/facebookresearch/fastText) implementation.
The distribution and Python import name are `bytefast`. The upstream package
remains `fasttext`, and both can be installed together.

The original fastText documentation is maintained separately in
[original-fasttext.md](original-fasttext.md).

## Installation

This package is available from the source repository; it has not been
published to PyPI:

```bash
git clone https://github.com/CommonGlot/ByteFast.git
cd ByteFast
python -m pip install .
```

The upstream distribution may be installed alongside it:

```bash
python -m pip install fasttext
```

## Load existing models

```python
import bytefast

native = bytefast.load_model("bytefast-model.bin")
original = bytefast.load_model("original-fasttext-model.bin")
quantized = bytefast.load_model("original-fasttext-model.ftz")
combined = bytefast.load_model("combined-bytefast-model.bin")

labels, probabilities = combined.predict("The team won the tournament", k=2)
```

The loader identifies model formats by the first bytes of the file rather than
the filename extension. The supported model types are:

| Type | Reader |
| --- | --- |
| Original fastText `.bin` | Original fastText backend |
| Quantized fastText `.ftz` | Original fastText backend |
| ByteFast byte classifier | ByteFast byte model |
| ByteFast lexical bundle | ByteFast byte model, lexicon, and learned fusion |

The original fastText Python package cannot read ByteFast models directly.
Use `bytefast.load_model()` for those models.

## Train an arbitrary classification dataset

The input format is one labeled record per line:

```text
__label__sports The team won its match.
__label__finance The company reported earnings.
__label__technology The computer uses a new processor.
```

```python
model = bytefast.train_supervised(
    "train.txt",
    output="classifier.bin",
    dim=32,
    bucket=16384,
    minn=2,
    maxn=5,
    epoch=10,
)
```

All class labels are supplied by the training data. The byte encoder uses
continuous UTF-8 n-grams, including n-grams across whitespace. Training data
with longer records may require a larger `max_bytes` setting than its default.

For additional lexical evidence, use the separate native C++ tools:

```bash
make bytefast-proto bytefast-lexical bytefast-calibrated
./bytefast-proto train train.txt base.bin -dim 32 -bucket 16384 -epoch 10
./bytefast-lexical build base.bin train.txt words.lex 200000 1
./bytefast-calibrated fit base.bin words.lex \
    calibration.txt validation.txt weights.fuse
```

To store the three ByteFast components in one file:

```python
bytefast.bundle_model(
    "base.bin", "words.lex", "weights.fuse", "complete.bin"
)
model = bytefast.load_model("complete.bin")
```

The single-file bundle uses a versioned and size-checked header. During
loading, the components are unpacked into a temporary directory that remains
available while the model object exists.

## Original fastText algorithms

The existing fastText supervised, unsupervised and word-embedding algorithms
remain available:

```python
import bytefast

classic = bytefast.train_fasttext_supervised(input="train.txt")
vectors = bytefast.train_unsupervised(input="corpus.txt")
```

They are also accessible through `bytefast.legacy`.

The native C++ programs remain separate: `./bytefast` retains the original
fastText commands; `./bytefast` uses its own model format.

## Limitations

- ByteFast does not accept every original fastText training parameter through
  its `train_supervised` wrapper. Use `train_fasttext_supervised` for the
  original algorithm and its options.
- Original fastText model loading and native ByteFast model loading are
  separate implementations selected by the header.
- Benchmarks depend on the classification dataset, label inventory, and model
  configuration. Results from a particular application do not establish
  performance on other datasets.

The automated package and model-compatibility tests are in
[bytefast-package.yml](../.github/workflows/bytefast-package.yml).
