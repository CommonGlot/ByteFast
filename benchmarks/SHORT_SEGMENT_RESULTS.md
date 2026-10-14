# ByteFast short-segment classification: WiLI-2018 evaluation

**Run:** https://github.com/CommonGlot/ByteFast/actions/runs/38626316891
**Result artifact:** \`bytefast-wili-short-segments\` (file: \`short-segment-results.json\`).
**Script:** [run_short_segments.py](run_short_segments.py).
**Status:** one successful exploratory run, **not** independent validation of full GlotLID coverage.

## Data and split

- 235 WiLI-2018 language labels from the official data (no additional training sources).
- Official WiLI training: 400 full paragraphs per label (94,000 total), plus 50 calibration and 50 validation paragraphs per label.
- Official WiLI test: 50 paragraphs per label (11,750 total) scored on the *same* examples in all configurations.
- Short test views: prefixes of 5, 10, 20, 40, 80 **Unicode codepoints**, plus 10, 20 and 40-codepoint spans from the middle. Crops occur before UTF-8 encoding, preserving valid multibyte characters. Each view is derived from the same original test paragraph, so views are **not independent**.
- Mixed training: the 94,000 full training paragraphs + one 20-codepoint pseudorandom internal crop of *each same train paragraph*, deterministic by document content (188,000 training lines). Both ByteFast and original fastText get this exact augmented corpus. The mixed ByteFast fusion parameters are fitted using full and cropped training-derived calibration paragraphs, with separate validation documents.
- ByteFast byte encoder: ASCII folding, 32 embedding dimensions, 16,384 hash buckets, byte 2–5-grams, 10 epochs, lr=0.3. No architectural changes across the mixed and long-only models.
- Lexical model: same 200,000-word cap and min document count 1, built from full training paragraphs; same word map for both byte encoders. A distinct trained fusion for each byte encoder; fixed ambiguity gate 0.75 (previously selected on development).
- fastText baseline: same 32 embedding dimensions, 16,384 buckets, subword character n-grams 2–5, 40 epochs, lr=0.3, two training threads. Same hyperparameters in full-only and mixed training.
- Model choice and hyperparameters were **not reselected on the official test**. That said, earlier ByteFast research has already inspected this official test split; the whole experiment is exploratory rather than an unbiased external evaluation.

## Prefix lengths: macro F1

| Unicode characters | Byte-only, full-trained | Byte + lexicon, full-trained | fastText, full-trained | Byte-only, mixed-trained | Byte + lexicon, mixed-trained | fastText, mixed-trained |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 0.212993 | 0.226777 | 0.106095 | 0.236664 | **0.264006** | 0.161155 |
| 10 | 0.318196 | 0.375026 | 0.225460 | 0.393352 | **0.458959** | 0.305558 |
| 20 | 0.474691 | 0.582340 | 0.408995 | 0.594917 | **0.697161** | 0.550613 |
| 40 | 0.675545 | 0.788807 | 0.638511 | 0.775656 | **0.853251** | 0.753933 |
| 80 | 0.827310 | 0.901475 | 0.823383 | 0.881558 | **0.926473** | 0.875626 |
| Full paragraph | 0.931137 | 0.955567 | 0.942766 | 0.941643 | **0.956840** | 0.947971 |

On 20-character prefixes, ByteFast with lexical fusion and mixed training achieves **0.697161**, compared with 0.550613 for fastText with exactly the same mixed training records (+0.146548 macro F1). The stratified paired bootstrap 95% percentile interval for this difference is **[0.137722, 0.155232]** with 250 replicates, resampling official test documents *within* their true language.

## Internal spans: macro F1

| View | Byte + lexicon, full-trained | fastText, full-trained | Byte + lexicon, mixed-trained | fastText, mixed-trained |
|---|---:|---:|---:|---:|
| 10 chars from middle | 0.376249 | 0.247253 | **0.486757** | 0.365607 |
| 20 chars from middle | 0.597482 | 0.433256 | **0.706005** | 0.581443 |
| 40 chars from middle | 0.788120 | 0.656560 | **0.855619** | 0.774480 |

The 20-character internal segment comparison is 0.706005 against 0.581443 (+0.124562 macro F1, stratified paired bootstrap 95% percentile interval **[0.116574, 0.133814]**).

## BCMS macro F1

The BCMS set on WiLI includes **bos, hbs, hrv, srp**, only 50 test items per class. It does *not* contain Montenegrin \`cnr\`. Treat these four-class averages as higher variance than the full-language macro F1.

| View | Byte + lexicon, long-only | fastText, long-only | Byte + lexicon, mixed | fastText, mixed |
|---|---:|---:|---:|---:|
| Prefix 10 | 0.163756 | 0.140826 | **0.246207** | 0.111116 |
| Prefix 20 | 0.304620 | 0.219778 | **0.482159** | 0.332761 |
| Prefix 40 | 0.488189 | 0.405924 | **0.594162** | 0.496314 |
| Middle 20 | 0.298579 | 0.247912 | **0.389726** | 0.321871 |
| Full paragraph | **0.708446** | 0.682522 | 0.704738 | 0.658532 |

## CPU and model size

Same GitHub Actions CPU runner, wall-clock seconds for **11,750 inputs including CLI startup and model loading**, median of three runs:

| Model | Full-paragraph wall time | Prefix-20 wall time | Serialized bytes |
|---|---:|---:|---:|
| Byte-only, full-trained | 0.1392 | 0.0439 | 2,129,882 |
| Byte + lexicon, full-trained | 0.2236 | 0.1075 | 8,192,495 |
| fastText, full-trained | 0.9609 | 0.8035 | 192,248,893 |
| Byte-only, mixed-trained | 0.1479 | 0.0385 | 2,129,882 |
| **Byte + lexicon, mixed-trained** | **0.2188** | **0.1013** | **8,192,495** |
| fastText, mixed-trained | 0.9928 | 0.8336 | 200,979,509 |

On prefix-20 text with mixed training, the lexical ByteFast model was ~8.2× faster including loading and its serialized model ~24.5× smaller than this specific unquantized fastText baseline. Times are tied to one runner and CLI subprocess I/O, not warm in-process inference.

## Observations and limitations

1. ByteFast substantially outperforms original fastText on short text **in this same-domain WiLI setup**, even without short augmentation. Its lexical sidecar helps at 10–40 characters, despite partial words often missing from the vocabulary.
2. Both architectures improve when short crops are included in training. ByteFast mixed lexical gains on prefix-20 by ~11.5 macro F1 percentage points, and fastText mixed gains by ~14.2 points versus their respective long-only controls. Full-paragraph macro F1 is not damaged in this particular dataset.
3. Extremely short inputs remain difficult: even the strongest model reaches only 0.264 macro F1 at five Unicode characters and 0.459 at ten.
4. The cropped views are not natural short user messages and remain Wikipedia-derived. Evaluate independently on social-media text, short queries, code-switching, unknown languages and noise categories.
5. These models predict among only **235 known labels**. They provide **no evidence of reliable unknown-language rejection** or performance for the 2,000+ language label set.
6. Training compute was not exactly matched: ByteFast used 10 epochs and ordinary fastText 40, as in the earlier tuned baseline. The augmentation was identical within each architecture.

The source and complete machine-readable metrics are in the Actions artifact attached to the run linked above.
