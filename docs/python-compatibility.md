# Python package and model-file compatibility

The distribution and import name is `bytefast`. It can be installed next to
the upstream `fasttext` package.

```bash
git clone https://github.com/CommonGlot/ByteFast.git
cd ByteFast
python -m pip install .
```

## Training

`bytefast.train_supervised` trains the ByteFast model. It takes every fastText
supervised argument plus the ByteFast text features:

```python
import bytefast

model = bytefast.train_supervised(
    input="train.txt",
    minb=2, maxb=5,                 # byte n-grams over the text
    minc=0, maxc=0,                 # character n-grams over the text (off)
    minCount=5, max_vocab=200_000,  # word vectors
    wordlist="",                    # optional allowed vocabulary
    hash_words=True,                # hashed vectors for other words
    fold="unicode", max_bytes=0,
    bucket=1_000_000, dim=64, epoch=5, lr=0.5, thread=8, loss="softmax",
)
model.save_model("model.bin")
```

`bytefast.train_unsupervised` trains word vectors as in fastText and accepts
`fold`, `wordlist` and `maxVocab`.

Both functions accept fastText's camelCase names and these snake_case
aliases: `min_count`, `min_count_label`, `word_ngrams`, `lr_update_rate`,
`label_prefix`, `pretrained_vectors`, `max_bytes`, `hash_words` and
`max_vocab`.

## Loading models

`bytefast.load_model(path)` reads the file header and picks the reader:

| File | Reader |
| --- | --- |
| ByteFast model with text features (`BFCM`) | `bytefast.core` |
| ByteFast model with only fastText features | `bytefast.core`, also upstream `fasttext` |
| Original fastText `.bin` / `.ftz` | `bytefast.core`, also upstream `fasttext` |
| Prototype byte model (`TFYB`) and bundle (`BFBN`) | `bytefast.proto` |

Upstream `fasttext` rejects `BFCM` files with "wrong file format".

## Prototype API

The earlier byte-only classifier and its lexical sidecar are in
`bytefast.proto`:

```python
model = bytefast.proto.train_supervised(
    "train.txt", output="proto.bin", dim=32, bucket=16384, epoch=10,
    lexical=True, lexical_vocab=200000, lexical_min_df=1,
    lexical_split=(0.8, 0.1, 0.1), lexical_fusion_epochs=90,
)
bytefast.bundle_model("base.bin", "words.lex", "weights.fuse", "complete.bin")
```

See [bytefast-prototype.md](bytefast-prototype.md) for the matching
command-line tools.

## Renamed names

| Before | Now |
| --- | --- |
| `bytefast.train_supervised` (byte prototype) | `bytefast.proto.train_supervised` |
| `bytefast.train_fasttext_supervised` | `bytefast.train_supervised` (old name still works) |
| `bytefast.legacy` | `bytefast.core` (old name still works) |
| `bytefast.ByteFastModel` | `bytefast.proto.ProtoModel` |
| `bytefast.util.download_model` | removed; download published vectors directly |

The automated tests are in
[bytefast-core.yml](../.github/workflows/bytefast-core.yml) and
[bytefast-package.yml](../.github/workflows/bytefast-package.yml).
