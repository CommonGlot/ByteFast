# ByteFast Python package and model-file compatibility

ByteFast is a CPU-first multilingual language identification library derived
from the original MIT-licensed Facebook fastText. Its **Python distribution
and import namespace are bytefast**, not fasttext.

## Installation

This is a **development branch**, not yet a public PyPI release:

    git clone https://github.com/CommonGlot/ByteFast.git
    cd ByteFast
    git switch bytefast/consolidated-package
    python -m pip install .

The original upstream fastText package may be installed in the same Python
environment:

    python -m pip install fasttext

The original upstream package still uses import fasttext; our fork exports
import bytefast. The original native C++ implementation and its license
notices remain in the repository for source compatibility.

## Automatically detect model formats

    import bytefast
    native = bytefast.load_model("native-bytefast.bin")
    upstream = bytefast.load_model("my-previous-fasttext.bin")
    compact = bytefast.load_model("full-bytefast-bundle.bin")
    for model in (native, upstream, compact):
        print(model.predict("Bonjour le monde"))

The four-byte magic header determines the model type, regardless of filename
extension. The supported types are:

- Original fastText .bin and .ftz: fastText's original format, including
  quantized models, via the bundled legacy reader
- ByteFast byte model: ByteFast-specific format version 1 or 2, using raw
  UTF-8 2–5-gram hashing
- ByteFast lexical bundle: one portable file that contains the byte model,
  sparse lexical evidence and trained fusion parameters

Do not give ByteFast files to upstream fasttext.load_model(); they are not
upstream fastText binaries.

## ByteFast training

    model = bytefast.train_supervised(
        "train.fasttext", output="base.bin",
        dim=32, bucket=16384, minn=2, maxn=5,
        epoch=10, ascii_fold=True,
    )

ByteFast's byte n-grams **cross whitespace boundaries**. The optional word
lexicon is an additional learned lexical scorer, not a mandatory tokenizer.

For the more elaborate learned lexical pipeline, the native tools currently
provide:

    make bytefast bytefast-lexical bytefast-calibrated

    ./bytefast train train.fasttext base.bin -dim 32 -bucket 16384 -epoch 10 -asciiFold 1
    ./bytefast-lexical build base.bin train.fasttext words.lex 200000 1
    ./bytefast-calibrated fit base.bin words.lex calib.fasttext validation.fasttext weights.fuse

Produce a self-contained single-file model:

    import bytefast
    bytefast.bundle_model("base.bin", "words.lex",
                          "weights.fuse", "complete.bin")
    model = bytefast.load_model("complete.bin")
    print(model.predict("Vlak dolazi sljedeći tjedan"))

The bundle is versioned, length checked and bounded to 1 GB. Loading the
combined model extracts its three parts to a temporary directory, which is
retained as long as the model object is alive.

## Access original fastText training/embedding algorithms

Use the original fastText from the installed upstream distribution:

    import fasttext
    ft = fasttext.train_supervised(input="train.fasttext")
    ft.save_model("classic.bin")

Or use the original fastText algorithm included in ByteFast, with the native
backend loaded in an isolated Python namespace:

    import bytefast
    ft = bytefast.train_fasttext_supervised(
        input="train.fasttext", epoch=10
    )
    # alternatively: bytefast.legacy.train_supervised(...)

Unsupervised word embeddings remain available through
bytefast.train_unsupervised(...) and bytefast.legacy.train_unsupervised(...).

## Backward compatibility and limits

- Installing the upstream fasttext PyPI package remains unaffected and
  does not implicitly install ByteFast.
- The new bytefast package is separate, so both can be installed in the
  same environment and imported in one Python process.
- ByteFast does not pretend that its native .bin is an upstream fastText
  .bin; it dispatches based on file signatures.
- The native C++ programs (./fasttext, ./bytefast, ./bytefast-calibrated)
  remain distinct. The bytefast Python wrapper is **not** a drop-in replacement
  for every fastText training parameter.
- ByteFast models have not yet been shown to outperform full GlotLID on
  an independent external noisy-web benchmark.

CI: .github/workflows/bytefast-package.yml proves both model readers, native
ByteFast training, original fastText training, combined model bundling and
coexistence with the PyPI fasttext package.
