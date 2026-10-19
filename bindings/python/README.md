# bytefast (Python)

```bash
python -m pip install .       # from a clone of this repository
```

## Training

```python
import bytefast

model = bytefast.train_supervised(
    input="train.txt",            # one example per line, labels first
    minb=2, maxb=5,               # byte n-grams over the text
    bucket=1_000_000, dim=64, epoch=5, lr=0.5, thread=8,
)
model.save_model("model.safetensors")
```

`train_supervised` takes keyword arguments only. Every option of
`bytefast supervised -h` is accepted under the same name, and these
snake_case aliases also work: `min_count`, `label_prefix`, `max_bytes`,
`crop_prob`, `crop_min`, `crop_max`, `norm_prob`, `strip_prob`,
`lex_threshold`, `lex_min_words` and `lex_top_k`.

The loss is softmax. A line with several labels trains on one of them, chosen
at random each time it is read.

## Prediction and evaluation

```python
model = bytefast.load_model("model.safetensors")

labels, probs = model.predict("The striker scored twice", k=3)
all_labels, all_probs = model.predict(["first line", "second line"], k=1)

n, precision, recall = model.test("valid.txt", k=1)
model.set_lexicon_threshold(0.0)   # turn off the lexicon check
```

`predict` takes one line at a time (no `\n` inside a string).
`threshold=` drops labels below that probability.

## Model information

```python
model.labels                     # label names
model.words                      # vocabulary
model.get_labels(include_freq=True)
model.get_dimension()
```
