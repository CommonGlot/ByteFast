# Supervised word embeddings + confusion-aware vocabulary (WiLI-2018)

## Why this experiment

The original ByteFast model is a fast, raw UTF-8 byte 2–5-gram classifier. A sparse statistical word-to-language sidecar dramatically improved accuracy on WiLI, especially Bosnian/Croatian/Serbian-related varieties. This experiment asks whether **actual supervised word vectors** can improve the model beyond a fixed lexical log-ratio table, and whether a **confusion-aware** embedding vocabulary outperforms global TF-IDF ranking.

The word branch is optional and separate. Byte n-grams still cross whitespace; new words/OOV are always represented by the unchanged byte encoder.

## Training and feature selection

- **Dataset:** official WiLI-2018, 235 language labels. 400 train and 100 development examples per label from the official training partition; 50 examples per label from the official test partition. No CommonLID training.
- **Byte model:** 32-dimensional, 16,384 n-gram buckets, UTF-8 byte n-grams 2–5, ASCII case-folding on, 10 epochs.
- **Word encoder:** compact 16- or 32-dimensional *trainable* embeddings for selected complete words (simple ASCII delimiters; UTF-8 bytes within a word retained).
- **Training:** supervised multiclass cross-entropy, updating word embeddings, learned language output projection, per-language bias, and learned *byte score coefficient*. The byte embedding table is frozen.
- **Vocabulary candidates:** require a word in >=3 training documents, then rank by log(1+document frequency) x language IDF. The confusion-aware variant adds a multiplier for words that strongly separate the 48 language pairs most frequently confused **on training data**. The comparison uses the same 40,000-word budget.
- **Tuning:** decide among word vocabulary variants by development macro F1 + 0.12 times BCMS macro F1. Test results are used only after choosing the variant.
- **BCMS metric:** macro F1 across the four labels present in WiLI: bos, hbs, hrv, srp. Montenegrin cnr is absent, so these results do not cover it.

## First 6-epoch learned-word experiment

| Configuration | Dev macro F1 | Dev BCMS F1 | Combined serialized size |
|---|---:|---:|---:|
| TF-IDF, 40k words, 16 dims | 0.935399 | 0.635925 | 5,153,026 bytes |
| Confusion-aware, 40k words, 16 dims | **0.935709** | **0.651434** | 5,177,993 bytes |
| Confusion-aware, 100k words, 16 dims | 0.935785 | 0.636975 | 9,767,606 bytes |

The dev-selected candidate was **confusion-aware 40k**. Its trained byte coefficient at epoch 6 was 1.47877, rather than an arbitrary fixed lexical multiplier. The head and word vectors were optimized jointly, so there is no single scalar lexical ratio controlling all word evidence.

### Official held-out test (235 languages, 11,750 samples)

| Model | Overall macro F1 | BCMS macro F1 | Total size | CPU predict wall seconds (median 3, includes load) |
|---|---:|---:|---:|---:|
| Byte-only ByteFast | 0.931137 | 0.445980 | 2.13 MB | 0.2452 |
| **Learned word embeddings** (40k confusion words) | 0.936168 | 0.617607 | 5.18 MB | 0.3545 |
| **Statistical gated lexical sidecar** | **0.954319** | **0.688549** | 7.21 MB | 0.3401 |
| Normal fastText (40 epochs) | 0.942702 | 0.677116 | 192.25 MB | 1.5471 |

**Interpretation:** confusion-aware word *selection* helped on development data at a fixed 40k-word budget. Six-epoch learned embeddings improved BCMS substantially compared with byte-only ByteFast, but lost to the far simpler statistical lexical sidecar, while being marginally slower. Learned embeddings are not yet the recommended default.

Full reproducible first run: https://github.com/CommonGlot/ByteFast/actions/runs/38483055692

## Limitations and next test

- Six epochs may undertrain the 40k-word embedding matrix; a longer-epoch/32-dimensional follow-up is pending.
- Vocabulary confusion scores come from *in-sample training errors*, and may overfit domain or imbalance. Domain-generalization tests remain necessary.
- A test-set comparison on WiLI has already been observed. Further exploratory WiLI tuning should be confirmed on **another fully held-out corpus** before a publication/release claim.
- Not evaluated against full GlotLID v3 with 2,102 labels, word-quantized fastText, true noisy web, per-language false-positive rates, or CommonLID held-out evaluation.
