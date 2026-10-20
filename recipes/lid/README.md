# Language identification recipe (GlotLID-Byte)

`train.sh` builds a language-ID model from a GlotLID-style corpus
(`data/LABEL/LABEL_SOURCE.txt`) with the settings chosen in
[docs/architecture.md](../../docs/architecture.md):

```bash
make opt
recipes/lid/train.sh data/glotlid-corpus out/lid                  # default: fast and accurate
recipes/lid/train.sh data/glotlid-corpus out/accurate lid-accurate          # + subwords
recipes/lid/train.sh data/glotlid-corpus out/accurate lid-accurate experts  # + experts
```

| Setting | Model | Speed (1 thread) | Size |
| --- | --- | --- | --- |
| `lid` (default) | bytes 2-5 (262k buckets), the 20k most frequent words, dim 64, `-maxBytes 512` | ~10k lines/s | ~85 MB |
| `lid-accurate` | `lid` + subwords 2-5, 1M buckets | ~8k lines/s | ~280 MB |
| `experts` | + one expert per group of `expert_groups.tsv` (Arabic dialects, Hindi group, zsm/ind, Chinese, Tibetan) | close to the above; only unsure lines of a group run an expert | +~1.2 GB |

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
   split only (at least 3 lines, the label's script, at least 2 sources,
   97% coverage up to 10k words, no strays of the 40 biggest labels). The
   model keeps them for the lexicon check (`-lexicon`).
3. A first model and `fit_weights.py`: a per-label bias fitted on dev,
   turned into training weights exp(b/4) (x0.25-x4, at most 20 passes over
   a label's lines), and x2 for labels the first model confuses
   (`-labelWeights`).
4. The model, with the weights and the lexicon.
5. With `experts`: for each group, a model on the group's unique training
   lines (at most 300k per label), stored in the model with `bytefast
   add-expert` at gate 0.8.

`BYTEFAST`, `THREADS` and `TOTAL` set the binary, the thread count and the
number of training lines. The scripts need Python 3 with `numpy` and
`regex`.

What is measured and what is not: each piece was tested on 20M lines on its
own (numbers in docs/architecture.md). The weights at exp(b/4), the experts
trained without word bigrams and the full combination have not been scored
together yet.
