# ByteFast lexical-hybrid experiment (WiLI-2018)

## Rationale

FUN-LangID (google-research/url-nlp/fun_langid) uses a small per-language set of characteristic stripped character 4-grams. It does **not** train a full word vocabulary, but its boundary-sensitive examples suggest lexical information is useful. FUN-LangID's authors explicitly warn that close varieties remain hard.

In ByteFast, a separate *optional* C++ lexicon learns word-to-language evidence while the original raw UTF-8 byte 2–5-gram classifier remains untouched. Word segmentation is an **additional** ASCII-boundary lexical feature extractor, not a replacement for the byte branch. This enables cheap experiments without introducing attention or convolution.

## First real-data experiment

Official WiLI-2018 test partition was held out. For each of 235 languages: 400 training paragraphs, 100 dev paragraphs, 50 test paragraphs. ByteFast trained 10 epochs, fastText control 40 epochs; both 32-dimensional, 16,384 hash buckets. The word lexicon used 50,000 learned word entries, minimum document frequency 3, up to 4 positive language associations per entry. The lexical score strength was selected on dev only (weight 1.0).

| System | Overall test macro F1 | BCMS (bos, hbs, hrv, srp) macro F1 | Serialized bytes | Median wall time on 11,750 lines |
|---|---:|---:|---:|---:|
| ByteFast, byte-only | 0.931137 | 0.445980 | 2,129,882 | 0.2374 s |
| ByteFast + 50k lexical sidecar | **0.950360** | 0.561809 | 3,765,111 | 0.3881 s |
| Original fastText, 40 epochs | 0.941682 | **0.662660** | 192,248,893 | 1.5167 s |

[Source: successful GitHub Actions training, evaluation and CPU benchmark](https://github.com/CommonGlot/ByteFast/actions/runs/38480009340).

**The lexical sidecar added approximately 1.64 MB**, improved held-out global and BCMS quality, but increased ByteFast prediction wall time by ~63%. It remained about 3.9× faster and 51× smaller than this unquantized fastText baseline. It is not a claim of superiority over GlotLID, tuned/quantized fastText, or across scripts and noisy web documents.

BCMS label-wise test F1 (byte-only / lexical / fastText):

| Label | Byte-only | Lexical | fastText |
|---|---:|---:|---:|
| bos | 0.07692 | 0.29851 | **0.54348** |
| hbs | 0.43023 | 0.50000 | **0.57143** |
| hrv | 0.30769 | 0.46914 | **0.56604** |
| srp | 0.96907 | **0.97959** | 0.96970 |

## Interpretation

The user's hypothesis—that lexical evidence matters for closely related languages—is supported by this experiment. However, a sparse positive-evidence sidecar does not yet match a fully word-aware discriminative classifier on the hardest BCMS labels. Vocabulary optimization, stronger lexical weights, and BCMS-specific hard negatives are promising next experiments.

Caveats: Wikipedia is not noisy web text, labels/domains can carry lexical signatures, and word-based associations can overfit topic. A fair GlotLID-scale benchmark should also measure corpus contamination, sample imbalance, unseen languages, noisy text, latency, model-size constraints, and language-specific FPR.

## How to build

```bash
make bytefast bytefast-lexical
./bytefast train train.fasttext base.bin -dim 32 -bucket 16384 -epoch 10 -asciiFold 1
./bytefast-lexical build base.bin train.fasttext words.lex 50000 3
./bytefast-lexical predict base.bin words.lex unlabeled.txt -weight 1.0
```

No changes have been merged to the original fastText executable. All work is on the `bytefast/lexical-hybrid` branch.

## Expanded vocabulary search (development-selected, independent official test)

Tested requested caps 50,000, 100,000, and 200,000 with weights 0.6–3.0 on the **WiLI train-only development split**. The largest cap yielded only 158,487 eligible words (minimum count 3). A weight of 3.0 maximized dev BCMS F1 while retaining overall macro F1 within 0.01 of the byte-only dev baseline.

| Metric | Byte-only | Expanded lexicon | Tuned fastText |
|---|---:|---:|---:|
| Held-out overall macro F1 | 0.931137 | **0.953170** | 0.942611 |
| Held-out BCMS macro F1 (bos, hbs, hrv, srp) | 0.445980 | **0.687268** | 0.682824 |
| Held-out accuracy | 93.04% | **94.86%** | 94.18% |
| Model bytes | 2,129,882 | **7,209,081** (base + sidecar) | 192,248,893 |
| Median CLI wall time on 11,750 samples | 0.2384 s | 0.4528 s | 1.5225 s |

**Source:** [expanded vocabulary/action run](https://github.com/CommonGlot/ByteFast/actions/runs/38480457663).

This is evidence of the usefulness of complete-word associations for close varieties. The absolute BCMS difference relative to fastText (0.00444 macro F1) is tiny, from only 50 test examples per language; do not call this a statistically-established advantage. No full GlotLID-scale or real-web validation yet.

## Optional confidence-gated lexical fast path

A further code change lets inference skip the word lexicon when the byte model's top-two probability gap is at least `-gateMargin`. The gate is a **compute-routing heuristic**, not an open-set rejection policy or calibrated confidence score.

Gate thresholds 0–2 were selected on the same 23,500-example development split, with constraints of <0.003 overall macro F1 and <0.015 BCMS F1 loss relative to always-on lexical scoring. The development-selected gate was **0.75**.

| Metric | Byte score only (gate 0) | Always lexical (gate 2) | Selected gate 0.75 |
|---|---:|---:|---:|
| Test macro F1 | 0.931137 | 0.953170 | **0.954319** |
| Test BCMS F1 | 0.445980 | 0.687268 | **0.688549** |
| Median CLI wall time (11,750 examples) | **0.3015 s** | 0.4378 s | 0.3276 s |

Timing here is measured through the **same lexical CLI**, including base/sidecar model loading, so it differs from the earlier direct ByteFast `./bytefast predict` time. The sidecar file is 5,079,199 bytes, the base 2,129,882 bytes, total 7,209,081 bytes.

**Source:** [confidence gate action run](https://github.com/CommonGlot/ByteFast/actions/runs/38480834079).

The gate largely preserves lexical quality with only a small overhead over byte-only scoring through the same CLI on this benchmark. Reproducibility with multiple seeds and other domains is required before enabling the lexical branch by default.

```bash
./bytefast-lexical predict base.bin words.lex input.txt -weight 3.0 -gateMargin 0.75
```

## What remains unproven

- Stronger-than-GlotLID performance at 2,102 labels (we only tested 235 WiLI languages)
- Equal tuning budgets, model-size-matched/quantized fastText comparison
- Performance of the word branch on code-switching, noisy OCR, and no-whitespace scripts
- Open-set rejection and per-language FPR under highly imbalanced web prevalence
- How the lexicon behaves when trained on a corpus with different topic/domain distributions

For now, keep this architecture **experimental and optional**, and retain a byte-only fast path.
