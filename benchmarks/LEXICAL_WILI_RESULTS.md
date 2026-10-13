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
