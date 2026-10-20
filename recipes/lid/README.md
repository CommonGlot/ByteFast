# Language identification recipe (GlotLID-Byte)

`train.sh` builds a language-ID model from a GlotLID-style corpus
(`data/LABEL/LABEL_SOURCE.txt`) with the settings chosen in
[docs/architecture.md](../../docs/architecture.md):

```bash
make opt
recipes/lid/train.sh data/glotlid-corpus out/lid                    # default: fast and accurate
recipes/lid/train.sh data/glotlid-corpus out/accurate lid-accurate  # + subwords
```

| Setting | Model | Speed (1 thread) | Size |
| --- | --- | --- | --- |
| `lid` (default) | bytes 2-5 (262k buckets), the 20k most frequent words, dim 64, `-maxBytes 512` | 12.0k lines/s | 84 MB |
| `lid-accurate` | `lid` + subwords 2-5, 1M buckets | 8.8k lines/s | 285 MB |

GlotLID-Byte v4 (20M lines, 2,088 labels), macro F1:

| | CommonLID | 5 words | 20 chars | UDHR | SmolSent | FLORES+ devtest | test split |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lid` | 0.6126 | 0.5511 | 0.4884 | 0.7579 | 0.7390 | 0.9428 | 0.9839 |
| `lid-accurate` | 0.6136 | 0.5575 | 0.4952 | 0.7637 | 0.7386 | 0.9472 | 0.9858 |

Both presets train for 3 epochs at lr 0.8 on random 5-20 character spans
(30% of lines), CommonLID-normalized lines (30%) and lines without optional
diacritics (10%).

Steps (each is skipped if its output exists):

1. `prepare.py`: per-label 15% hold-out for dev and test, then GlotLID's
   upsampling to `TOTAL` (20M) training lines. `source_caps.tsv` caps a few
   source files that would otherwise dominate their label (the Dyula religious
   text in bam_Latn); `build_lexicon.py` applies the same caps so its training
   split matches.
2. `build_lexicon.py`: per-label lists of correct words from the training
   split only (at least 3 lines, the label's script, at least 2 sources when
   a second source holds 5% of the lines, 97% coverage up to 10k words, no
   strays of the 40 biggest labels), kept only if a list accepts at least 90%
   of its label's held-out dev lines. The model keeps them for the lexicon
   check (`-lexicon`; `bytefast set-lexicon` swaps them later).
3. A first model and `fit_weights.py`: a per-label bias fitted on dev,
   turned into training weights exp(b/4) (x0.25-x4, at most 20 passes over
   a label's lines), and x2 for labels the first model confuses
   (`-labelWeights`).
4. The model, with the weights and the lexicon.

`BYTEFAST`, `THREADS` and `TOTAL` set the binary, the thread count and the
number of training lines. The scripts need Python 3 with `numpy` and
`regex`.

The numbers above are for models trained before the corpus's vote-based
cleaning of high-resource lines (glotlid-corpus e1e95d9).
