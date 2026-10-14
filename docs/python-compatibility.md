# Python compatibility: existing fastText stays usable

ByteFast is a fork of the original fastText repository. **The Python package
continues to be named `fasttext`** and the existing `fasttext` C++ executable
is unchanged. All experimental ByteFast C++ programs are separate, optional
executables, and are deliberately excluded from the upstream Python extension's
source list in `setup.py`.

## For existing fastText users

Installing upstream **`pip install fasttext`** from PyPI still obtains the
regular fastText distribution, not ByteFast. Installing this fork from source
will continue to expose the **same** `import fasttext` API:

```sh
git clone https://github.com/CommonGlot/ByteFast.git
cd ByteFast
# On the compatibility development branch, until these changes are merged:
git switch bytefast/python-compatibility
python -m pip install .
```

```python
import fasttext

model = fasttext.train_supervised(input="train.txt", epoch=5)
model.save_model("classic_fasttext.bin")

reloaded = fasttext.load_model("classic_fasttext.bin")
print(reloaded.predict("Example text"))

# These remain available as before:
# fasttext.train_unsupervised(...)
# fasttext.tokenize(...)
```

Legacy `.bin`/quantized `.ftz` fastText model formats are handled by
the original binding implementation; ByteFast has **not** changed or
extended that binary reader.

## For ByteFast experiments in the *same checkout*

```sh
make fasttext
make bytefast bytefast-lexical bytefast-calibrated
```

The experimental ByteFast commands are separate and use a versioned,
incompatible `bytefast` model format:

```sh
./bytefast train train.txt byte.bin -dim 32 -bucket 16384 -asciiFold 1
./bytefast-lexical build byte.bin train.txt words.lex 200000 1
./bytefast-calibrated fit byte.bin words.lex calib.txt validation.txt weights.fuse
./bytefast-calibrated predict byte.bin words.lex weights.fuse inputs.txt -gate 0.75
```

Do **not** pass `byte.bin` into `fasttext.load_model()`; the file format
is different. The legacy and experimental model loaders have not been
silently interchanged.

## What this does *not* provide (yet)

- Installing the public PyPI `fasttext` package does **not** automatically
  install ByteFast functionality. We don't control upstream's distribution.
- The ByteFast-specific models cannot yet be loaded using the legacy
  `fasttext.load_model()` call. A separate `bytefast` Python package
  and native extension would be the clean way to expose that functionality.
- Installing the fork and the upstream fastText package into the *same*
  Python environment as separate distributions is not supported because both
  use the distribution/import name `fasttext`. Use virtual environments
  to switch versions.

## Regression tests

The workflow `.github/workflows/fasttext-python-compat.yml`
installs this fork into a clean Python environment, checks supervised
training, save/load and prediction, verifies that an unsupervised word model
still trains, and builds the separate ByteFast executables.

We intentionally keep `fasttext.train_supervised`, `train_unsupervised`
and `load_model` unchanged rather than inserting an implicit new mode.
