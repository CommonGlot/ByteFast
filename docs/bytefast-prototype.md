# ByteFast C++ classification commands

ByteFast contains standalone C++ executables for byte-level supervised text
classification. The original `fasttext` executable remains available and
accepts its original commands and model files.

## Build

```bash
make -j2 opt
make bytefast bytefast-lexical bytefast-calibrated
```

A C++17 compiler is required.

## Training input

One labeled example per line:

```text
__label__sports The tournament begins tomorrow.
__label__finance The company published its financial report.
__label__technology The device uses a new processor.
```

The `__label__` prefix is parsed separately from text features. Byte
n-grams are extracted from the remainder of the line, including its spaces,
punctuation and UTF-8 bytes. Newlines delimit examples. The implementation
accepts one label per training line.

## Byte classifier commands

```bash
./bytefast train train.txt classifier.bin \
    -dim 32 -bucket 16384 -epoch 10 -lr 0.15 \
    -minn 2 -maxn 5 -hashCount 1

./bytefast test classifier.bin test-labeled.txt
./bytefast predict classifier.bin inputs.txt
./bytefast bench classifier.bin inputs.txt
```

The optional `-asciiFold 1` setting folds ASCII uppercase letters to
lowercase when hashing features. The default maximum input length is
512 bytes; change it with `-maxBytes N`. The optional
`-hashCount 2` and `-positionBins 4` settings are experimental.

The saved model uses ByteFast's own versioned binary format. It is not
an original fastText `.bin` file.

## Lexical classifier and fitted weights

Whole-word evidence can improve classification when similar classes differ
in a small number of characteristic words:

```bash
./bytefast-lexical build classifier.bin train.txt words.lex 200000 1
./bytefast-calibrated fit classifier.bin words.lex \
    calibration.txt validation.txt weights.fuse

./bytefast-calibrated predict classifier.bin words.lex \
    weights.fuse inputs.txt -gate 0.75
```

The number `200000` specifies the maximum word vocabulary size;
`1` specifies the minimum number of training documents in which a
candidate word must appear. These are configurable parameters rather than
mandatory constants.

The fusion model learns the relative importance of byte and lexical evidence
using labeled calibration examples and separate validation examples.

For distribution as a single file, use the Python package:

```python
import bytefast

bytefast.bundle_model(
    "classifier.bin", "words.lex", "weights.fuse", "complete.bin"
)
model = bytefast.load_model("complete.bin")
```

## Model loading

Use `bytefast.load_model()` from Python to load an original fastText
`.bin`/`.ftz` file or a ByteFast model. It checks the header and selects
the corresponding reader.

The `./fasttext` executable continues to read original fastText models;
the `./bytefast` executable reads ByteFast byte models.

## Limitations

- Training currently loads labeled examples into memory rather than streaming
  arbitrarily large datasets.
- The output uses a multiclass linear classifier; training work increases
  with the number of labels.
- Classifiers and lexical models are stored in separate native formats.
- The default maximum byte length is intentionally configurable and may be
  too short for long-document tasks.
- Model quality and CPU performance depend on the classification dataset and
  should be validated for the intended application.

The [original fastText README](original-fasttext.md) documents the original
algorithms and commands. The [Python compatibility documentation](python-compatibility.md)
describes automatic model-format detection.
