# ByteFast

ByteFast is a general-purpose text classification library based on
[fastText](https://github.com/facebookresearch/fastText). It supports
supervised classification using contiguous UTF-8 byte n-grams and optional
whole-word features. It also includes the original fastText learning
algorithms and model reader.

ByteFast can be trained with any set of class labels, such as topics, document
types, sentiment categories, or languages. It does not require a predefined
label inventory.

The [original fastText README](docs/original-fasttext.md) is maintained
separately. The original implementations and copyright notices are preserved
under the [MIT license](LICENSE).

## Installation

The Python distribution is currently installed from source and has not been
published to PyPI:

```bash
git clone https://github.com/CommonGlot/ByteFast.git
cd ByteFast
python -m pip install .
```

The installed Python package is named `bytefast`, so it can coexist with the
upstream `fasttext` package.

## Training a classifier

Training data uses fastText-style labeled lines:

```text
__label__sports The team won its third consecutive match.
__label__technology The processor uses a new chip architecture.
__label__finance The company reported its quarterly earnings.
```

Train and save a ByteFast byte classifier:

```python
import bytefast

model = bytefast.train_supervised(
    "train.txt",
    output="classifier.bin",
    dim=32,
    bucket=16384,
    minn=2,
    maxn=5,
    epoch=10,
)

labels, scores = model.predict("The team won the tournament", k=2)
print(labels, scores)
```

The primary ByteFast encoder extracts n-grams from the raw UTF-8 byte
sequence. Unlike the original fastText word tokenizer, its n-grams may cross
word and whitespace boundaries. The optional lexical classifier adds whole-word
features without changing that byte encoder.

## Loading trained models

```python
import bytefast

byte_model = bytefast.load_model("classifier.bin")
original_model = bytefast.load_model("existing-fasttext-model.bin")
lexical_model = bytefast.load_model("combined-bytefast-model.bin")
```

`bytefast.load_model()` identifies the format by inspecting the file header.

| Model file | Supported reader |
| --- | --- |
| Original fastText `.bin` | Original fastText model reader |
| Quantized fastText `.ftz` | Original fastText model reader |
| ByteFast byte classifier `.bin` | ByteFast model reader |
| ByteFast lexical bundle `.bin` | ByteFast byte classifier and learned lexical fusion |

These model formats are not interchangeable in the original `fasttext`
package. Use `bytefast.load_model()` when loading a ByteFast model.

The original fastText supervised and unsupervised training methods are
available through the legacy backend:

```python
classic = bytefast.train_fasttext_supervised(input="train.txt")
vectors = bytefast.train_unsupervised(input="corpus.txt")
```

See [Python API and model compatibility](docs/python-compatibility.md).

## C++ executables

Build the original fastText executable:

```bash
make -j2 opt
./fasttext supervised -input train.txt -output classic-model
```

Build the ByteFast executables:

```bash
make bytefast bytefast-lexical bytefast-calibrated
./bytefast train train.txt byte-model.bin -dim 32 -bucket 16384 -epoch 10
```

The `fasttext` executable retains its original commands and model format.
The ByteFast executables use separate model formats.

## Optional lexical classification

For classification tasks where whole words help distinguish similar classes,
ByteFast supports a compact word-to-class association table and learned
fusion coefficients. The base classifier remains byte-based.

Configure lexical selection in the same Python training call:

```python
import bytefast

model = bytefast.train_supervised(
    "train.txt",
    output="classifier.bin",
    dim=32,
    bucket=16384,
    minn=2,
    maxn=5,
    epoch=10,
    lexical=True,
    lexical_vocab=200000,
    lexical_min_df=1,
    lexical_selection="global",
    lexical_split=(0.8, 0.1, 0.1),
    lexical_fusion_epochs=90,
    lexical_fusion_lr=0.07,
    lexical_fusion_l2=0.01,
    lexical_fusion_patience=16,
)

labels, scores = model.predict("The team won the tournament", k=2)
```

The `lexical=True` option automatically makes class-stratified training,
calibration and validation partitions from `train.txt` (`lexical_split`,
80/10/10 by default). The
byte classifier and lexical vocabulary are fitted to the training partition;
the remaining partitions are used to fit and validate learned fusion weights.
The output is a single versioned ByteFast model file that loads through
`bytefast.load_model()`. Every class needs at least ten examples for the
automatic split.

For explicitly disjoint files, pass `lexical_calibration_data="calibration.txt"`
and `lexical_validation_data="validation.txt"`. In that case the full
`train.txt` file is used for the byte classifier and vocabulary. The caller
must ensure the datasets do not share examples.

Currently, `lexical_selection="global"` uses document frequency multiplied
by the strongest positive word-to-class evidence. Class-balanced and
confusion-aware selection are not implemented in this API yet. The number of
stored class associations per word remains fixed at four. Without
`lexical=True`, the original byte-only training behavior is unchanged.

The fusion layer is fitted with full-batch Adam: `lexical_fusion_epochs` is
the epoch limit, `lexical_fusion_patience` the number of epochs without a
validation improvement before stopping, `lexical_fusion_lr` the step size and
`lexical_fusion_l2` the penalty on per-class offsets. The byte encoder's
`hash_count` (1–4) and `position_bins` (0 or 4) ablations are also arguments of
`train_supervised`.

The equivalent native tools remain available:

```bash
./bytefast-lexical build byte-model.bin train.txt words.lex 200000 1
./bytefast-calibrated fit byte-model.bin words.lex \
    calibration.txt validation.txt weights.fuse \
    -epochs 90 -lr 0.07 -l2 0.01 -patience 16
```

The lexical and fusion parameters can be bundled with the byte model:

```python
bytefast.bundle_model(
    "byte-model.bin", "words.lex", "weights.fuse", "complete.bin"
)
model = bytefast.load_model("complete.bin")
print(model.predict("The processor has a new architecture"))
```

Use separate calibration and validation examples rather than fitting and
evaluating the fusion layer on identical examples.

## Project status

ByteFast is under development. Its experimental benchmarks and test
workflows are in the [benchmarks](benchmarks/) and
[.github/workflows](.github/workflows/) directories. Performance and accuracy
depend on the training dataset, model size, and selected features.

## Original fastText documentation

The original repository README is available in
[docs/original-fasttext.md](docs/original-fasttext.md). It documents the
original fastText C++ commands, Python interface, word embeddings, training
methods, and references.

See [LICENSE](LICENSE) and the copyright notices in source files.
