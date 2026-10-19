#!/usr/bin/env python3
"""Per-label training weights for ByteFast's -labelWeights.

1. Score MODEL on a dev sample (top-k labels and probabilities) and fit a
   per-label bias b that raises macro F1: b += STEP * (FN - FP) / n per
   round, keeping the best round.
2. A label's weight is exp(b / --temperature) (4: about half the bias's
   effect, which keeps natural-distribution accuracy), clamped to
   [--min-weight, --max-weight] and, with --labels (prepare.py's
   labels.tsv), to --max-repeat passes over the label's unique lines.
3. Labels of confused pairs get --pair-weight times more: the pairs of
   --pairs (two labels per line), or else every pair the model confuses at
   least --min-confusions times on the dev sample.

  python recipes/lid/fit_weights.py --bin ./bytefast --model base.safetensors \\
      --dev data/glotlid/dev.txt --labels data/glotlid/labels.tsv \\
      --pairs pairs.tsv --out weights.tsv
"""
import argparse
import math
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PREFIX = "__label__"


def top_k(binary, model, path, k, threads):
    out = subprocess.run([binary, "predict-prob", model, str(path), str(k), "0.0", str(threads)],
                         check=True, stdout=subprocess.PIPE).stdout.decode("utf-8", "replace")
    rows = []
    for line in out.split("\n")[:-1]:
        parts = line.split()
        rows.append([(parts[i][len(PREFIX):], float(parts[i + 1])) for i in range(0, len(parts) - 1, 2)])
    return rows


def macro_f1(gold, pred, n):
    tp = np.bincount(pred[pred == gold], minlength=n)
    fp = np.bincount(pred[pred != gold], minlength=n)
    fn = np.bincount(gold[pred != gold], minlength=n)
    f1 = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    covered = np.bincount(gold, minlength=n) > 0
    return f1[covered].mean(), fp, fn


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--dev-lines", type=int, default=300000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--min-weight", type=float, default=0.25)
    p.add_argument("--max-weight", type=float, default=4.0)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--max-repeat", type=float, default=20.0)
    p.add_argument("--pairs", type=Path, default=None)
    p.add_argument("--pair-weight", type=float, default=2.0)
    p.add_argument("--min-confusions", type=int, default=5)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    lines = open(a.dev, encoding="utf-8", errors="replace").read().split("\n")[:-1]
    random.Random(0).shuffle(lines)
    lines = [l for l in lines if l.startswith(PREFIX)][:a.dev_lines]
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "dev.txt"
        sample.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows = top_k(a.bin, a.model, sample, a.k, a.threads)
    labels = sorted({l.split(" ", 1)[0][len(PREFIX):] for l in lines} |
                    {lab for row in rows for lab, _ in row})
    index = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    gold = np.array([index[l.split(" ", 1)[0][len(PREFIX):]] for l in lines])
    lab = np.zeros((len(rows), a.k), dtype=np.int64)
    logp = np.full((len(rows), a.k), -np.inf)
    for i, row in enumerate(rows):
        for j, (l, prob) in enumerate(row[:a.k]):
            lab[i, j], logp[i, j] = index[l], math.log(max(prob, 1e-12))

    def decide(b):
        return lab[np.arange(len(lab)), (logp + b[lab]).argmax(axis=1)]

    counts = np.maximum(np.bincount(gold, minlength=n), 1)
    bias = np.zeros(n)
    best = (macro_f1(gold, decide(bias), n)[0], bias.copy())
    start = best[0]
    for _ in range(a.rounds):
        _, fp, fn = macro_f1(gold, decide(bias), n)
        bias += a.step * (fn - fp) / counts
        f1 = macro_f1(gold, decide(bias), n)[0]
        if f1 > best[0]:
            best = (f1, bias.copy())
    bias = best[1]
    print("dev macro F1 %.4f -> %.4f with the bias" % (start, best[0]))

    cap = {}
    if a.labels:
        for line in open(a.labels, encoding="utf-8"):
            cols = line.rstrip("\n").split("\t")
            if cols[0] != "label":
                pool, sampled = int(cols[1]), int(cols[2])
                cap[cols[0]] = a.max_repeat * pool / max(sampled, 1)
    paired = set()
    if a.pairs:
        for line in open(a.pairs, encoding="utf-8"):
            paired.update(line.split()[:2])
    else:
        pred = decide(np.zeros(n))
        wrong = pred != gold
        pairs, times = np.unique(np.stack([np.minimum(gold, pred), np.maximum(gold, pred)])[:, wrong],
                                 axis=1, return_counts=True)
        for (x, y), c in zip(pairs.T, times):
            if c >= a.min_confusions:
                paired.update((labels[x], labels[y]))
    print("%d labels in confused pairs" % len(paired))
    with open(a.out, "w", encoding="utf-8") as out:
        for l in labels:
            w = min(max(math.exp(bias[index[l]] / a.temperature), a.min_weight), a.max_weight)
            w = min(w, cap.get(l, w))
            if l in paired:
                w *= a.pair_weight
            out.write("%s\t%.4f\n" % (l, w))
    print("wrote %d label weights to %s" % (n, a.out))


if __name__ == "__main__":
    main()
