# ByteFast: CPU LID robustness and open-set experiments (2026-10-14)

These experiments were motivated by the W-NUT 2026 presentation *Rethinking Language Identification: From Noisy Text to Multilingual LLMs*, especially the need to evaluate short text, domain shift, false positives, open-set rejection, and throughput. The slide deck itself is not included.

**All results are from GitHub Actions on WiLI-2018, not from CommonLID.** The original official WiLI test examples were not used to train models or select reject thresholds. Source: https://github.com/CommonGlot/ByteFast/actions

## Architecture

ByteFast raw UTF-8 bytes -> 2–5 byte n-gram hashing -> mean-pooled embedding -> linear softmax. One hash, no positional vectors, no neural attention/convolution. Configurable ASCII case folding adds a branch inside incremental byte hashing, without changing embedding size. Inference also supports confidence and top-two margin rejection; these scores are **uncalibrated** and must not be interpreted as posterior probabilities.

Models are stored in a versioned ByteFast format. Version 2 serializes ASCII-fold status; the loader remains compatible with version 1. Training rejects configurations estimated to exceed the 1 GB downloadable file budget.

## Controlled held-out WiLI robustness (235 languages)

Original fastText was trained 40 epochs, ByteFast 10 epochs, same examples, 32 embedding dimensions and 16,384 hash buckets. The stronger fastText baseline is **not** a fully tuned or quantized GlotLID model.

| Condition | FastText macro F1 | ByteFast macro F1 |
|---|---:|---:|
| Clean (max 512 bytes) | **0.946441** | 0.937211 |
| 32-byte inputs | 0.495798 | **0.558854** |
| 64-byte inputs | 0.724334 | **0.747889** |
| 128-byte inputs | **0.872008** | 0.870263 |
| Alternating case | 0.220565 | **0.383024** |
| Synthetic OCR-like noise | **0.931686** | 0.923918 |
| Added web-header prefix | **0.936993** | 0.932361 |

Report/log: https://github.com/CommonGlot/ByteFast/actions/runs/38477580632

On 11,750 clean paragraphs ByteFast had median wall time 0.14 s, peak RSS ~6.3 MB and model file 2,129,878 bytes. FastText had median wall time 1.09 s, peak RSS ~617 MB and model file 228,018,258 bytes. Includes model loading and startup; not a quantized baseline.

## ASCII-folded byte hashing ablation (235 languages)

| Condition | Original ByteFast F1 | With -asciiFold 1 F1 |
|---|---:|---:|
| Clean | 0.937211 | 0.936984 |
| 32 bytes | 0.558854 | **0.565339** |
| 64 bytes | 0.747889 | **0.750466** |
| 128 bytes | 0.870263 | **0.872181** |
| Alternating case | 0.383024 | **0.874619** |
| OCR-like noise | **0.923918** | 0.923898 |
| Web-header prefix | **0.932361** | 0.930515 |

Both variants were 2,129,882 bytes in the new model format. Median inference wall time 0.13 s without fold versus ~0.14 s with fold. A second run and more scripts/domains are needed, but this is a strong low-cost robustness win.

Log: https://github.com/CommonGlot/ByteFast/actions/runs/38477967275

## True held-out-language rejection

The first experiment trained on 200 languages and withheld 35 completely. Reject thresholds were selected on held-out examples from the WiLI training partition, with >=90% known-language dev acceptance, then tested on the untouched official test partition. Synthetic junk tests are NOT representative of all natural web noise.

| Metric | ByteFast | fastText |
|---|---:|---:|
| Known-language acceptance | 89.83% | **90.80%** |
| Correct among accepted known examples | 98.39% | **98.59%** |
| Unseen-language false acceptance (35 languages) | 46.06% | **42.69%** |
| Synthetic-noise false acceptance | **0.40%** | 8.60% |
| Test macro per-language FPR | 0.0003906 | **0.0003762** |

Both models misidentify too many entirely unsupported languages, confirming the open-set failure mode. Confidence/margin rejection is an inexpensive *policy*, not a solution.

Log: https://github.com/CommonGlot/ByteFast/actions/runs/38477349291

## Training dedicated unknown and nonlinguistic classes

A second experiment used 15 additional languages **only** as training examples for a generic `und` label, and added distinct synthetic `zxx` training examples. A disjoint set of 20 languages remained fully unseen in training and threshold selection.

| Metric | No negative training | Trained und/zxx |
|---|---:|---:|
| Known-language acceptance | 89.83% | 90.76% |
| Correct among accepted known | 98.39% | 97.94% |
| Unseen false acceptance (20 unseen) | 51.50% | **48.90%** |
| Synthetic-noise false acceptance | 0.40% | **0.00%** |
| Model file | 2,125,006 bytes | 2,125,284 bytes |

These results are not directly comparable to the 35-language experiment above, because the unknown language inventory differs. This is only a partial improvement; real unseen-language rejection remains inadequate.

Reproducible run: https://github.com/CommonGlot/ByteFast/actions/runs/38478506925

## Decisions

1. Retain the **single-hash mean-pooled linear classifier**. Earlier WiLI tests found Bloom multi-hashes and position bins slower and less accurate under the evaluated training recipe.
2. Keep optional **ASCII folding** for robustness-sensitive deployment; confirm with more seeds and external data before changing the default.
3. Continue evaluating explicit `und_<Script>` and `zxx_*` labels inspired by GlotLID rather than relying solely on max softmax or margin. This prototype currently supports generic labels only.
4. **Do not claim that ByteFast outperforms GlotLID or all fastText configurations.** It sometimes loses to an appropriately trained fastText control on clean WiLI.
5. Before a release or full GlotLID training, run CommonLID **evaluation only**, more noisy-domain datasets, low-prevalence corpus precision simulations, multilingual/code-switch tests, seed variance, tuning-budget matched baselines, and 2,102-label CPU profiling.

No user access tokens or CommonLID examples were used in these tests.

## Combined rejection + ASCII case fold (deterministic rerun)

After sorting the negative-training language list to eliminate dependence on Python hash randomization, we trained and compared all three models using the same 200 known languages, 15 `und`-training languages, and 20 entirely unseen evaluation languages.

| Metric | ByteFast base | + und/zxx | + und/zxx and ASCII fold |
|---|---:|---:|---:|
| Known-language acceptance | 89.83% | 90.76% | **90.83%** |
| Known-language accuracy among accepted | **98.39%** | 97.94% | 97.97% |
| Completely unseen language false acceptance | 51.50% | 48.90% | **48.10%** |
| Synthetic noise false acceptance | 0.40% | **0%** | **0%** |
| Inference wall time for 10,000 known paragraphs | 0.21 s | 0.21 s | 0.21 s |
| Model size | 2,125,006 bytes | 2,125,284 bytes | 2,125,284 bytes |

Run: https://github.com/CommonGlot/ByteFast/actions/runs/38478506925

**Interpretation:** Combined generic und/zxx training and case folding modestly reduce unseen-language false acceptance without slowing this small model, but about half the texts in fully unseen languages are *still* accepted as some known language. Do not ship generic rejection as a solved feature. Further tests need real OOD scripts, per-script unknown labels, low-prevalence FPR targets, and larger multilingual coverage. The earlier und/zxx run measured 47.8% false acceptance; that result is superseded by the deterministic negative-data generator (48.9%).
