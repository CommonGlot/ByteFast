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

## Longer-training follow-up: confusion-aware vocabulary works

The six-epoch models above were undertrained. To determine whether that explained the negative results, we compared four new configurations, selecting **only on the official WiLI train-derived development split**. Note that the original WiLI test scores from the first run had already been viewed; thus the follow-up test should be regarded as exploratory until reproduced on a second external corpus.

| Word encoder | Dev overall macro F1 | Dev BCMS F1 | Base + word model bytes |
|---|---:|---:|---:|
| TF-IDF 40k words, dim 16, 18 epochs | **0.943872** | 0.674164 | 5,153,026 |
| Confusion-aware 40k, dim 16, 18 epochs | 0.943358 | **0.717271** | 5,177,993 |
| Confusion-aware 40k, dim 32, 15 epochs | 0.943596 | 0.709358 | 7,753,033 |
| Confusion-aware 100k, dim 32, 15 epochs | 0.943960 | 0.704217 | 16,182,646 |

Confusion-aware ranking thus improved BCMS **dev F1 by 4.31 percentage points** compared to same-size TF-IDF ranking, at a slight cost to overall dev F1. More words or wider vectors did not reliably improve the dev objective.

The dev selection objective was `macro_f1 + 0.12 * bcms_f1`; it selected **40k confusion-aware, dim 16, 18 epochs**.

### Exploratory second test

| Model | Overall F1 | BCMS F1 | Total bytes | Median CPU CLI wall seconds (11,750 examples, incl. loading) |
|---|---:|---:|---:|---:|
| Byte-only | 0.931137 | 0.445980 | 2,129,882 | 0.1346 |
| **Trained word embeddings** | 0.943954 | **0.689364** | 5,177,993 | 0.1996 |
| Statistical lexicon (gated) | **0.954319** | 0.688549 | 7,209,081 | 0.2003 |
| Normal fastText, 40 epochs | 0.942806 | 0.677627 | 192,248,893 | 0.9368 |

All models ran on the same job, but these timings are not portable across different GitHub Action runners. BCMS advantage of learned versus statistical is only 0.000815 F1, far too small to establish superiority. In contrast, the one-point loss in global F1 is material.

The learned byte coefficient after 18 epochs was approximately **1.51** (versus 1.0 at initialization); each word vector and the language output matrix were trained with cross entropy. This is genuinely supervised word representation learning, not merely re-scaling the original statistical lexicon.

**Decision:** keep statistical gated ByteFast as the global-accuracy incumbent; keep supervised word embeddings as a promising lighter BCMS-oriented alternative. Evaluate on genuinely external/non-Wikipedia text and at full GlotLID coverage before changing defaults.

Successful longer-training run: https://github.com/CommonGlot/ByteFast/actions/runs/38483523780


## Equal-epoch width sweep: 16, 32, 64, 128, 256, 512 (2026-10-14)

Follow-up experiment: the same 40,000-word confusion-aware vocabulary budget, minimum document frequency=3, 18 supervised epochs, learning rate 0.035, identical WiLI train-derived development set, same byte model. Trainable word dimensions were the **only** architecture parameter changed. Model loading and training were extended from a 128-dimensional maximum to 512, retaining the 900 MB parameter guard.

| Word dimensions | Dev overall macro F1 | Dev BCMS macro F1 | Serialized model total (bytes) | Median dev prediction wall seconds (23,500 texts, incl. load) |
|---|---:|---:|---:|---:|
| 16 | 0.943358 | 0.717271 | 5,177,993 | 0.6634 |
| 32 | 0.944642 | **0.718026** | 7,753,033 | 0.7184 |
| 64 | 0.945097 | 0.709734 | 12,903,113 | 0.8389 |
| 128 | 0.945641 | 0.712707 | 23,203,273 | 1.0553 |
| 256 | 0.946133 | 0.703580 | 43,803,593 | 1.7325 |
| 512 | **0.947430** | 0.715496 | 85,004,233 | 3.1727 |

The dev selection objective was unchanged: overall_macro_f1 + 0.12 * BCMS_macro_f1. It chose 512 dimensions even though 32 was best for BCMS-specific F1.

The selected 512-d model was evaluated on the official WiLI-2018 test partition against three reference controls on the same CPU runner:

| Model | Official test overall macro F1 | Official test BCMS F1 | Size (bytes) | Median CLI wall seconds (11,750 test inputs) |
|---|---:|---:|---:|---:|
| Byte-only ByteFast | 0.931137 | 0.445980 | 2,129,882 | 0.2389 |
| **512-d learned word embeddings** | 0.947623 | 0.685498 | 85,004,233 | 1.5739 |
| **Gated statistical lexical ByteFast** | **0.954319** | **0.688549** | 7,209,081 | **0.3231** |
| Original fastText, 40 epochs | 0.942288 | 0.672240 | 192,248,893 | 1.4983 |

Earlier 16-d experiments reported test overall F1 0.943954, BCMS F1 0.689364, and a 5.18 MB model. These were obtained on an earlier run, so CPU times across runs should not be compared directly. The 512-d model modestly improves global test F1 but does not improve BCMS test F1. At this scale it was slower than even the ordinary (unquantized) fastText CLI on the same runner, despite being smaller.

**Decision:** Do not make 512 dimensions the default. The 16/32-d learned models are substantially more efficient; the 7.21 MB gated statistical lexical sidecar remains the best measured overall accuracy/size/latency tradeoff.

### Critical caveats

These results are **exploratory**: earlier official WiLI test results have already been inspected. Model selection here used the train-derived development partition, but an independent non-Wikipedia dataset is still required to validate the choice without repeated test-set exposure. The script evaluated the **selected** 512-d model on official WiLI test, not every dimension. No CommonLID training or GlotLID 2,102-class deployment evaluation was performed. The baseline fastText was not quantized or size-matched.

- [Successful GitHub Actions width sweep and logs](https://github.com/CommonGlot/ByteFast/actions/runs/38484470076)
- [Sweep script](../benchmarks/run_wordemb_widths.py)
- [Sweep workflow](../.github/workflows/bytefast-wordemb-widths.yml)
