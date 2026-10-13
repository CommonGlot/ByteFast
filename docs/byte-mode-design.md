# ByteFast byte-mode design

ByteFast extends fastText with an optional raw UTF-8 byte feature mode for CPU-first supervised language identification.

## Requirements

- **No whitespace tokenization in byte mode.** Treat spaces, punctuation, tabs, and multibyte UTF-8 bytes as part of the text. A newline separates training examples.
- Parse the supervised `__label__...` prefix **separately**; labels must never become text features.
- Initial byte n-gram lengths: 2–5, with hashed embeddings and fastText's mean pooling and linear classifier.
- Preserve legacy fastText training, prediction, and existing model loading by default.
- Add floret-style multiple hashes and optional position-aware features only as independently switchable, measured follow-up changes.
- No attention or convolution. Native CPU inference. Downloadable models must remain below 1 GB.

## Implementation order

1. Add a byte feature extractor with tests that show n-grams crossing spaces and UTF-8 code-point boundaries.
2. Wire byte mode into both supervised training and prediction, with identical feature extraction and explicit model-version compatibility.
3. Add tiny multilingual train/save/load/predict smoke tests and original-mode regression tests.
4. Benchmark CPU samples per second, macro F1 on held-out data/CommonLID, memory use, and model size.
5. Test multi-hash embeddings and position features as separate ablations.

## Initial hyperparameters to test

- Embedding dimension: 128
- Buckets: 524,288
- Maximum input bytes: 512 (to be ablated; inference should avoid unnecessary padding)
- Byte n-grams: 2–5

These are starting experiment settings, not validated optima.
