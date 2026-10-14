# ByteFast: WiLI-2018 benchmark, 2026-10-13

Real 235-language Wikipedia paragraph classification on the official WiLI-2018 train/test split. This is **not** a full GlotLID 2,000+ language benchmark. No CommonLID examples were used in training.

## Settings

- Fixed test partition: 11,750 examples, 50 per language
- Training examples/label: 5, 25, 100, 500, and imbalanced 5/25/500
- Deterministically shuffled training examples, random seed 1729
- Models: 32-dimensional embeddings, 16,384 hash buckets, five epochs, learning rate 0.3
- Input: at most 512 bytes; ByteFast contiguous UTF-8 byte n-grams 2–5; fastText word-bounded subwords 2–5
- GitHub-hosted Ubuntu CPU runner (AMD EPYC 9V74)

## Variable training budgets

| Samples per language | Total train | fastText accuracy | ByteFast accuracy | fastText macro F1 | ByteFast macro F1 |
|---|---:|---:|---:|---:|---:|
| 5 | 1,175 | 14.67% | 0.43% | 0.108773 | 0.000036 |
| 25 | 5,875 | 9.81% | 7.95% | 0.070893 | 0.054015 |
| 100 | 23,500 | 0.83% | 49.09% | 0.000177 | 0.440011 |
| 500 | 117,500 | 82.02% | **91.29%** | 0.815395 | **0.911971** |
| Imbalanced 5/25/500 | 49,585 | 34.24% | **43.20%** | 0.222118 | **0.321313** |

**Warning:** The fastText results at 25 and 100 examples are abnormally unstable. Treat them as training/optimization failures until tested with independently tuned learning rates, epochs, and multiple seeds. The 500-example result is the meaningful initial comparison, not proof of general superiority.

## Model size, CPU time and peak resident memory at 500 examples/language

| Metric | fastText | ByteFast (1 hash) |
|---|---:|---:|
| Model file | 228,018,258 bytes | 2,129,878 bytes |
| Median end-to-end 11,750-line prediction time (includes load) | 1.80 s | 0.24 s |
| Peak RSS | approximately 616 MB | approximately 6 MB |

Each time is a median of 3 runs on the same runner. FastText's model includes its word vocabulary; ByteFast has only hashed byte features. This is not a comparison with a quantized/tuned fastText baseline.

## Real-data ByteFast ablations

| Feature configuration | Macro F1 (100/label) | Macro F1 (500/label) | Wall seconds at 500/label |
|---|---:|---:|---:|
| One hash, no positions | **0.440011** | **0.911971** | **0.20** |
| Two hashes | 0.196304 | 0.888759 | 0.29 |
| One hash + four position bins | 0.118413 | 0.856345 | 0.32 |
| Two hashes + positions | 0.072474 | 0.789200 | 0.58 |

All ByteFast ablation models use the same 2.13 MB embedding table and classifier. Additional features increase lookup and training costs but did not increase quality with this training recipe.

**Recommendation:** keep the current default to 1 hash, no positions. Extra hash and position features remain optional experiments, not default improvements.

## Reproduce / inspect

- [Variable budgets action run](https://github.com/CommonGlot/ByteFast/actions/runs/38474109691)
- [Hash / position ablations action run](https://github.com/CommonGlot/ByteFast/actions/runs/38474706104)
- [WiLI experiment workflow](../.github/workflows/wili-experiment.yml)
- [WiLI ablation workflow](../.github/workflows/wili-ablations.yml)

Limitations: one training/subset seed, no CommonLID evaluation, 235 labels rather than all GlotLID labels, and fastText hyperparameters not independently optimized.
