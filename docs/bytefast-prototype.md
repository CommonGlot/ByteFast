# ByteFast experimental CPU classifier

This prototype is **not yet a replacement for GlotLID**. It is a standalone supervised C++ executable built in the fastText fork. The legacy `fasttext` binary remains unchanged.

## Build

```bash
make bytefast
```

Requires a C++17 compiler. To build upstream fastText separately: `make opt`.

## Labeled file format

One labeled example per line:

```text
__label__eng The quick brown fox.
__label__fra Bonjour le monde.
__label__spa Hola, ¿qué tal?
```

The prefix is parsed as metadata, **not tokenized or hashed**. After the first label delimiter, the remaining bytes (including spaces, punctuation, UTF-8 continuation bytes and tabs) are the text. Newlines delimit examples. This version supports one label per line.

## Train, evaluate, predict, benchmark

```bash
./bytefast train train.txt lid.bin -dim 64 -bucket 65536 -epoch 10 -lr 0.15 -minn 2 -maxn 5 -hashCount 1
./bytefast test lid.bin heldout-labeled.txt
./bytefast predict lid.bin heldout-unlabeled.txt
./bytefast bench lid.bin heldout-unlabeled.txt
```

Experiments: `-hashCount 2` for two hashes per byte n-gram, or `-positionBins 4` for an additional position-dependent feature per hashed n-gram. Either can be enabled independently. Both increase computation; neither is assumed to improve accuracy.

`-maxBytes 512` controls the maximum raw bytes consumed per input, with no fixed padding.

The saved `.bin` has its own versioned **ByteFast** format, not the upstream fastText `.bin` format.

## Current limitations

- Prototype uses a direct softmax over the full label set, so training cost grows with the number of languages.
- All labeled lines are loaded in memory for prototype training. GlotLID-scale data will need streaming/shuffling.
- Training uses straightforward SGD and float32 matrices, with no quantization or SIMD optimization yet.
- `bench` reports end-to-end per-line prediction including feature hashing and the output head, but its timing does not include model load. Run sufficiently long input files, compare identical hardware, and repeat measurements.
- The toy CI data has only three languages and repeated templates. Passing it is not evidence of CommonLID accuracy or improvements over GlotLID.
- ByteFast code is experimental, and functionality has not been merged into fastText's main training CLI.

See [draft PR #1](https://github.com/CommonGlot/ByteFast/pull/1) for ongoing development.
