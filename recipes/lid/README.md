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
   upsampling to `TOTAL` (20M; 0 keeps the full upsampled pool) training
   lines, at most `CAP` per label when set. Balance between labels comes
   from this sampling only; there are no per-label training weights.
2. `build_lexicon.py`: per-label lists of correct words from the training
   split only (at least 3 lines, the label's script, at least 2 sources when
   a second source holds 5% of the lines, 97% coverage up to 10k words, no
   strays of the 40 biggest labels), kept only if a list accepts at least 90%
   of its label's held-out dev lines. The model keeps them for the lexicon
   check (`-lexicon`; `bytefast set-lexicon` swaps them later).
3. The model, with the lexicon.

`BYTEFAST`, `THREADS`, `TOTAL` and `CAP` set the binary, the thread count, the
number of training lines and the per-label cap. The scripts need Python 3 with `numpy` and
`regex`.

The numbers above are for models trained before the corpus's vote-based
cleaning of high-resource lines (glotlid-corpus e1e95d9).
