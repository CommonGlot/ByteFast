# ByteFast architecture and evidence

ByteFast is a general-purpose text classifier. Language identification is
one of the tasks it is tuned on, but nothing in the library is specific to
languages: labels, feature types and normalization are all generic. An
option stays in ByteFast only if an ablation showed that it helps.

## Model

For an input text, ByteFast collects a list of feature ids, averages their
embeddings into one `dim`-dimensional vector, and applies a linear layer and
a softmax. This is the fastText supervised model; ByteFast changes which
features go into the average.

```
text ──► tokens ──┬─► word ids (vocabulary)               ─┐
                  ├─► subword n-grams      (-minn/-maxn)   ├─► mean ─► linear ─► softmax ─► lexicon check
                  └─► " w1 w2 … wn " ──► byte n-grams (-minb/-maxb) ─┘
```

All hashed features share one table of `-bucket` rows after the vocabulary
rows. Byte n-grams use their own hash (FNV-1a over unsigned bytes with a seed
and a murmur3 finalizer), so they do not collide systematically with the word
hash used for words, subwords and word n-grams, and power-of-two bucket
counts still use well-mixed bits.

- **Byte n-grams cross word boundaries.** The tokens of a line are joined by
  single spaces and padded with one space on each side, so repeated
  whitespace and tabs do not create features. Labels are never part of the
  text. `-maxBytes` limits how much of the line they read, cut at a
  character boundary; word features always see the whole line.
- **Vocabulary control.** `-minCount` (frequency) and `-wordlist` (a given
  vocabulary) can be combined.
- **Lexicon check.** `-lexicon` stores per-label word lists as a Bloom
  filter over (label, word) pairs. At prediction the top label must have at
  least `-lexThreshold` of the line's normalized words in its list, else the
  first of the top `-lexTopK` that does moves first (Caswell et al. 2020's
  percent-threshold filter, used as a re-ranker).
- **Training-only options.** `-cropProb/-cropMin/-cropMax` (random character
  spans), `-normProb` (CommonLID-style normalization: lowercase, no digits or
  punctuation) and `-stripProb` (no optional diacritics) change what a
  training step sees, never prediction. Each is drawn again for every line
  in every epoch. The Unicode tables come from `tools/gen_*.py`.
- **File format.** Models are safetensors files (`format` = `bytefast`,
  `format_version` = `1` in the metadata). The settings are `args.*`
  metadata strings, the vocabulary is three tensors (`vocab.strings`,
  `vocab.counts`, `vocab.types`), the matrices are `input` and `output`, and
  a lexicon is `lexicon.bloom` (U64) and `lexicon.labels` (U8, whether a
  label has a list).

## Evidence

General datasets: `benchmarks/ablate.py` with `benchmarks/ablations/*.json`
on WiLI-2018, AG News, DBpedia and Yelp (macro F1 on test, mean of 3 seeds,
learning rate and epochs tuned on dev for each variant). Language
identification: 20M lines of the GlotLID corpus with the GlotLID recipe,
scored on CommonLID (full, 5-word and 20-character views), UDHR, SmolSent,
FLORES+ devtest and the GlotLID test split.

**Kept**

| Option | Evidence |
| --- | --- |
| byte n-grams 2-5 | best or within 0.1 of the best on all four general datasets, and the smallest model; ties fastText's features for language ID at equal table size |
| words (`-minCount`, `-wordlist`) | help close languages (Arabic dialects, Hindi belt); a 20k-word list matches the full vocabulary at a tenth of its size |
| subwords (`-minn/-maxn`) | with bytes and words, the most accurate language-ID mix (CommonLID 0.613 vs 0.608 for bytes alone), at 8.1k vs 11.4k lines/s |
| `-maxBytes` | a speed knob: 1.8x faster on Yelp's long reviews, no change on short texts |
| `-cropProb` (character spans) | the short-text views of the GlotLID recipe |
| `-normProb` | CommonLID's normalized input |
| `-stripProb` | UDHR 0.815 -> 0.824 (2M lines, 0.3); Ewe and Fon gain, Yoruba loses |
| `-lexicon` (threshold 0.5, top 5, 2+ words) | CommonLID 0.601 -> 0.607, 5-word 0.535 -> 0.547, SmolSent 0.738 -> 0.745, UDHR 0.755 -> 0.753; thresholds 0.6-0.8 and returning no label are worse |

**Removed**

| Option | Why |
| --- | --- |
| `-hashWords` | no effect on any dataset (within 0.0005) |
| `-loss hs`, `-loss ns` | worse (WiLI 0.939 and 0.898 vs 0.955 for softmax) |
| `-loss ova` | diverged (NaN) on 2,102 language labels at fastText learning rates |
| character n-grams over the text (`-minc/-maxc`) | worse than bytes on all four general datasets; mixed for language ID |
| crop units word and byte | byte crops: no gain; word crops: 20-character F1 0.744 vs 0.762 for character crops |
| weighted average of the input rows, a hidden layer, focal loss, logit adjustment | diverged, mixed or worse on the 20M-line screen; never added |
| product quantization (`quantize`), `-inputModel`, autotune, pretrained vectors, word vectors (skipgram/cbow) | not used by any experiment, or outside a classifier's job |
| `-fold`, `-posBins`, `-textScope` | no gain in earlier ablations |
| word n-grams (`-wordNgrams`), `-maxVocab`, `-minCountLabel` | not part of the chosen language-ID settings (word bigrams helped only AG News and Yelp; `-wordlist` covers vocabulary size) |
| `-saveEpochs`, `-validation`, `-lrUpdateRate`, `test-label` | tooling outside the chosen settings; the epoch count is fixed at 3 |
