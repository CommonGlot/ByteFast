# Learned lexical fusion: removing the two magic 3s

**Status:** experimental, successful CPU CI on 235 WiLI-2018 languages. Not a full GlotLID or web-domain evaluation.

## Why

Old ByteFast combines a word association lexicon with byte predictions using a manually selected lexical weight of **3.0**, and builds that lexicon with a minimum word document frequency of **3**.

This experiment removes the fixed lexical weight, fitting byte and lexical weights by supervised cross-entropy with validation-based early stopping. It also allows vocabulary items appearing just once, under a 200,000-word budget, and compares minCount=1 and minCount=3 on validation.

The architecture remains raw UTF-8 -> continuous byte 2–5-grams -> one hashed embedding table -> mean pooling -> linear classifier. Complete-word evidence is a **separate optional sidecar**, not the primary tokenizer.

## Learned score

    S(language, text) =
       a * log P_byte(language | text)
       + (b + u[language]) * lexical_evidence(language, text)
       + bias[language]

The shared byte weight a, global word weight b, language-specific word corrections u and per-language bias are fitted on **50 training-derived calibration examples per language**. An additional 50 train-derived examples per language are used for validation and gate selection, while 400 train the original byte model and lexicon. The original **50 official WiLI test examples per language are not used in fitting**.

Learned coefficients with the cutoff-one model:
- Byte evidence shared coefficient: **0.602885**
- Word evidence shared coefficient: **0.923283**
- Effective per-language word weights span **0.219952 to 3.05438**
- Best optimization epoch by validation cross-entropy: **52**

So **3.0 is no longer a universal lexical multiplier**. Each language learns its own sensitivity, and the byte coefficient is learned too.

## Development-selected configuration

- Minimum document count: **1** (no exclusion beyond occurring once in training)
- Lexicon size: **200,000 words**
- Top-two byte-score margin gate: **0.75**
- The gate is chosen on development data with a 0.0015 macro/BCMS composite F1 tolerance and a 0.002 BCMS F1 tolerance in favor of less CPU work.

The minCount=3 lexicon had only 158,487 words and performed worse on this development set.

## Official WiLI test results (single GitHub Actions runner)

| Classifier | Overall macro F1 | BCMS macro F1 | Bytes on disk | CPU wall time for 11,750 paragraphs |
|---|---:|---:|---:|---:|
| **ByteFast + learned lexical fusion** | **0.955567** | **0.708446** | 8,192,495 | 0.3493 s |
| ByteFast + old fixed weight 3, min count 3 | 0.954319 | 0.688549 | 7,209,081 | **0.3379 s** |
| ByteFast byte-only | 0.931137 | 0.445980 | 2,129,882 | 0.2450 s |
| Ordinary fastText (40 epochs) | 0.942621 | 0.677627 | 192,248,893 | 1.5536 s |

The learned variant improves BCMS F1 by 1.99 percentage points and overall F1 by 0.125 percentage points relative to the fixed-3 variant. It is ~3.4% slower and ~1 MB larger; both variants are much smaller and faster than this ordinary, unquantized fastText baseline.

Individual BCMS F1 for the learned variant: bos 0.58947; hbs 0.60417; hrv 0.66055; srp 0.97959. The hbs label did not improve relative to the old fixed-weight baseline, so this is not a win for every individual label.

[Successful learned-gate training and final test](https://github.com/CommonGlot/ByteFast/actions/runs/38486654099) · [initial always-lexical experiment](https://github.com/CommonGlot/ByteFast/actions/runs/38486199303)

## Build and train

    make bytefast bytefast-lexical bytefast-calibrated
    ./bytefast train train.fasttext byte.bin -dim 32 -bucket 16384 -epoch 10 -asciiFold 1
    ./bytefast-lexical build byte.bin train.fasttext words.lex 200000 1
    ./bytefast-calibrated fit byte.bin words.lex fit.fasttext validation.fasttext weights.fuse
    ./bytefast-calibrated predict byte.bin words.lex weights.fuse input.txt -gate 0.75

This is a new optional C++ CPU executable; existing fastText and byte-only ByteFast remain unchanged. The fusion weights are stored in a compact, versioned, label-checked sidecar.

## Limitations

These results are exploratory: prior WiLI test results were already examined. The new parameters were fitted on a disjoint training-derived calibration partition and chosen on a separate training-derived validation partition, but a genuinely independent domain is required to confirm superiority. The comparison is not against a quantized or equally tuned fastText model, full GlotLID 2,102 languages, or real noisy web corpora. No claims about false-positive rates, open-set rejection, or improved low-resource corpus precision follow from these accuracy results.

No CommonLID examples were used for training.
