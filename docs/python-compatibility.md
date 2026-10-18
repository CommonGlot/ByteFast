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
    max_bytes=0,
    bucket=1_000_000, dim=64, epoch=5, lr=0.5, thread=8, loss="softmax",
)
model.save_model("model.bin")
```

`bytefast.train_unsupervised` trains word vectors as in fastText and accepts
`wordlist` and `maxVocab`.

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

Upstream `fasttext` rejects `BFCM` files with "wrong file format".

## Renamed names

| Before | Now |
| --- | --- |
| `bytefast.train_fasttext_supervised` | `bytefast.train_supervised` (old name still works) |
| `bytefast.legacy` | `bytefast.core` (old name still works) |
| `bytefast.util.download_model` | removed; download published vectors directly |

The automated tests are in
[bytefast-core.yml](../.github/workflows/bytefast-core.yml) and
[bytefast-package.yml](../.github/workflows/bytefast-package.yml).
