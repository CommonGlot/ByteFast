# Prototype byte classifier design (bytefast.proto)

This describes the earlier stand-alone byte classifier in `src/proto`
(`bytefast-proto`, `bytefast.proto`). The main ByteFast model is described
in [architecture.md](architecture.md); the prototype is kept as an ablation
baseline. The input label names and their
meanings are determined by the training dataset.

## Input handling

- Every labeled training line starts with a `__label__CLASS` prefix.
- The label is parsed as metadata and is excluded from model input features.
- Newlines delimit training records; byte n-grams may cross spaces, tabs,
  punctuation and UTF-8 character boundaries **within** a record.
- Input is truncated at a configurable maximum byte length (default: 512).
  Larger records may require a different value.
- Byte mode does not require a whitespace tokenizer.

## Model structure

- Contiguous UTF-8 byte n-grams, length 2–5 by default.
- Deterministic hashing into an embedding table.
- Mean pooling over input byte features.
- Linear output classifier over the labels present in training data.
- One hash per n-gram, with no position features.

This architecture applies to any labeled classification dataset. Label names
may represent topics, sentiment categories, document types, languages, or other
classes.

## Optional lexical features

Whole-word features are an additional input source. They do not replace the
byte-level encoder.

A separate lexical table stores associations between words and classes. Its
selected vocabulary can be capped by size. A learned fusion layer combines
byte classifier scores and class-specific lexical evidence, using coefficients
fitted on separate calibration data and selected using validation data.

## Existing fastText components

ByteFast retains the original fastText training algorithms, word
representations, and binary-model reader. The `./bytefast` command remains
available alongside the `./bytefast` commands.

The Python `bytefast.load_model()` function detects original fastText models
and ByteFast models by their file headers. Native ByteFast models cannot be
loaded with the upstream `fasttext.load_model()`.

## Implementation and evaluation constraints

- CPU-based C++ training and prediction.
- Versioned, validated model files.
- A maximum downloadable model budget of approximately 1 GB.
- Model size and prediction time must be measured alongside quality.
- Training quality must be measured on a disjoint test partition. When tuning
  or comparing model configurations, keep a separate development partition.
- Include a comparison with the original fastText implementation when
  evaluating a new feature.

See [ByteFast command reference](bytefast-prototype.md),
[Python compatibility](python-compatibility.md), and
[the original fastText README](original-fasttext.md).
