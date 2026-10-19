#!/usr/bin/env python3
"""Build train/dev/test files from the GlotLID corpus, following the GlotLID paper.

For every label (merged across the version folders given with --versions):
  1. read all source files, collapse whitespace, drop empty and duplicate lines;
  2. shuffle deterministically and hold out --holdout (15%) of the lines;
     test and dev are each min(--max-eval, half the hold-out) lines of it;
  3. the rest is the training pool.
Training lines are then resampled with temperature: label l gets
N * p_l**alpha / sum(p**alpha) lines (alpha = 1/T = 0.3), repeating lines of
small labels and subsampling large ones; N is the pool size (or --total).
--max-train-per-label caps that number for quick proxy runs.

--sampling unimax uses UniMax (Chung et al., 2023) instead: the budget N is
spread as evenly as possible over the labels, but no label gets more than
--epochs times its pool; budget a small label cannot use goes to the larger
ones (labels are filled from the smallest up).

--sampling upsample is GlotLID's own rule (assets/train/upsample.py): each label
gets max(pool, temperature target at the full corpus size), i.e. small labels
are upsampled and large ones keep all their lines. With --total the resulting
mixture is scaled to that budget (same shares); --max-repeat caps any label at
that many passes over its pool.

--variants writes several samplings from one pass over the corpus, e.g.
"temp0.3=temperature:0.3 unimax5=unimax:5": OUT/train.NAME.txt and
OUT/labels.NAME.tsv per variant; dev and test are shared.

Writes OUT/train.txt (globally shuffled), OUT/dev.txt, OUT/test.txt and
OUT/labels.tsv (per-label counts). The corpus licenses only allow research
use; keep the outputs on the machine where the corpus lives.

  python recipes/lid/prepare.py --corpus data/glotlid-corpus --out data/glotlid
"""
import argparse
import math
import os
import random
import shutil
import sys
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ARGS = None


def init_worker(args):
    global ARGS
    ARGS = args


def label_files(corpus, versions):
    files = defaultdict(list)
    for version in versions:
        root = corpus / version
        if not root.is_dir():
            print("skipping missing folder", root, file=sys.stderr)
            continue
        for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            files[label_dir.name] += sorted(label_dir.glob("*.txt"))
    return files


def split(label, paths):
    """Deterministic (train, dev, test) lists for one label."""
    seen, lines = set(), []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                text = " ".join(line.split())
                if text and text not in seen:
                    seen.add(text)
                    lines.append(text)
    rng = random.Random(ARGS.seed ^ zlib.crc32(label.encode()))
    rng.shuffle(lines)
    n = len(lines)
    hold = min(round(n * ARGS.holdout), n - 1) if n > 1 else 0
    n_test = min(ARGS.max_eval, math.ceil(hold / 2))
    n_dev = min(ARGS.max_eval, hold - n_test)
    return lines[hold:], lines[n_test:n_test + n_dev], lines[:n_test]


def temperature_targets(pools, total, alpha):
    total_pool = sum(pools.values())
    weights = {l: (n / total_pool) ** alpha for l, n in pools.items()}
    norm = sum(weights.values())
    return {l: max(1, round(total * w / norm)) for l, w in weights.items()}


def upsample_targets(pools, total, alpha, max_repeat=0):
    """GlotLID's rule: max(pool, temperature target); scaled to `total` lines if given."""
    full = sum(pools.values())
    temp = temperature_targets(pools, full, alpha)
    targets = {l: max(n, temp[l]) for l, n in pools.items()}
    if total:
        scale = total / sum(targets.values())
        targets = {l: max(1, round(t * scale)) for l, t in targets.items()}
    if max_repeat:
        targets = {l: min(t, max(1, round(pools[l] * max_repeat))) for l, t in targets.items()}
    return targets


def unimax_targets(pools, total, epochs):
    """UniMax: uniform budget per label, capped at `epochs` passes over its pool."""
    targets, left = {}, total
    order = sorted(pools, key=lambda l: (pools[l], l))
    for i, label in enumerate(order):
        share = left / (len(order) - i)
        targets[label] = max(1, min(round(pools[label] * epochs), round(share)))
        left -= targets[label]
    return targets


def pass_one(item):
    label, paths = item
    train, dev, test = split(label, paths)
    return label, len(train), dev, test


def pass_two(job):
    label, paths, targets = job
    train, _, _ = split(label, paths)
    for name, target in targets.items():
        rng = random.Random(ARGS.seed * 7919 ^ zlib.crc32(label.encode()))
        if target <= len(train):
            sample = rng.sample(train, target)
        else:
            sample = train * (target // len(train)) + rng.sample(train, target % len(train))
        # Each worker process appends to its own shard files, so no two processes
        # write the same file and the number of files stays workers * shards.
        by_shard = defaultdict(list)
        for text in sample:
            by_shard[rng.randrange(ARGS.shards)].append("__label__%s %s\n" % (label, text))
        shard_dir = ARGS.out / ("shards" + name)
        for s, lines in by_shard.items():
            with open(shard_dir / ("%03d-%d.txt" % (s, os.getpid())), "a", encoding="utf-8") as f:
                f.writelines(lines)
    return label


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--versions", nargs="+", default=["v3.1", "v3.2-ext", "v4.0-ext", "zxx"])
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument("--max-eval", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--sampling", choices=["temperature", "unimax", "upsample"], default="temperature")
    p.add_argument("--max-repeat", type=float, default=0, help="upsample: cap on passes over a pool")
    p.add_argument("--epochs", type=float, default=5.0, help="UniMax: max passes over a label's pool")
    p.add_argument("--variants", nargs="*", default=[],
                   help="NAME=temperature:ALPHA, NAME=unimax:EPOCHS or NAME=upsample:ALPHA[:MAX_REPEAT]; "
                        "one train file per variant")
    p.add_argument("--total", type=int, default=0, help="training lines (0 = pool size)")
    p.add_argument("--max-train-per-label", type=int, default=0)
    p.add_argument("--min-train", type=int, default=1,
                   help="drop labels with fewer training lines than this")
    p.add_argument("--shards", type=int, default=64)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--seed", type=int, default=1)
    ARGS = p.parse_args()
    ARGS.out.mkdir(parents=True, exist_ok=True)

    files = label_files(ARGS.corpus, ARGS.versions)
    print("labels found:", len(files), file=sys.stderr, flush=True)
    pool_sizes = {}
    with Pool(ARGS.workers, init_worker, (ARGS,)) as pool, \
            open(ARGS.out / "dev.txt", "w", encoding="utf-8") as dev_out, \
            open(ARGS.out / "test.txt", "w", encoding="utf-8") as test_out:
        for label, n_train, dev, test in pool.imap_unordered(pass_one, sorted(files.items())):
            if n_train < ARGS.min_train:
                continue
            pool_sizes[label] = (n_train, len(dev), len(test))
            dev_out.writelines("__label__%s %s\n" % (label, t) for t in dev)
            test_out.writelines("__label__%s %s\n" % (label, t) for t in test)

    pools = {l: v[0] for l, v in pool_sizes.items()}
    total_pool = sum(pools.values())
    total = ARGS.total or total_pool
    specs = {}  # "" = the single default variant (train.txt, labels.tsv)
    for v in ARGS.variants:
        name, spec = v.split("=")
        specs["." + name] = tuple(spec.split(":"))
    if not specs:
        specs[""] = {"temperature": ("temperature", str(ARGS.alpha)),
                     "unimax": ("unimax", str(ARGS.epochs)),
                     "upsample": ("upsample", str(ARGS.alpha), str(ARGS.max_repeat))}[ARGS.sampling]
    plans = {}
    for name, (kind, value, *rest) in specs.items():
        if kind == "temperature":
            targets = temperature_targets(pools, total, float(value))
        elif kind == "unimax":
            targets = unimax_targets(pools, total, float(value))
        else:
            targets = upsample_targets(pools, ARGS.total, float(value), float(rest[0]) if rest else 0)
        if ARGS.max_train_per_label:
            targets = {l: min(t, ARGS.max_train_per_label) for l, t in targets.items()}
        plans[name] = targets
        print("%s %s:%s pool lines: %d, sampled lines: %d, labels: %d" % (
            name or "train", kind, value, total_pool, sum(targets.values()), len(targets)),
            file=sys.stderr, flush=True)

    for name in plans:
        shard_dir = ARGS.out / ("shards" + name)
        shutil.rmtree(shard_dir, ignore_errors=True)
        shard_dir.mkdir()
    # Largest labels first, so that no long job starts last.
    jobs = sorted(((l, files[l], {n: t[l] for n, t in plans.items()}) for l in pools),
                  key=lambda j: -max(j[2].values()))
    with Pool(ARGS.workers, init_worker, (ARGS,)) as pool:
        for _ in pool.imap_unordered(pass_two, jobs):
            pass

    for name, targets in plans.items():
        shard_dir = ARGS.out / ("shards" + name)
        rng = random.Random(ARGS.seed)
        with open(ARGS.out / ("train%s.txt" % name), "w", encoding="utf-8") as out:
            for s in range(ARGS.shards):
                parts = sorted(shard_dir.glob("%03d-*.txt" % s))
                lines = []
                for part in parts:
                    with open(part, encoding="utf-8") as f:
                        lines += f.readlines()
                    part.unlink()
                rng.shuffle(lines)
                out.writelines(lines)
        shard_dir.rmdir()
        with open(ARGS.out / ("labels%s.tsv" % name), "w", encoding="utf-8") as f:
            f.write("label\tpool\tsampled\tdev\ttest\n")
            for label in sorted(targets):
                n_train, n_dev, n_test = pool_sizes[label]
                f.write("%s\t%d\t%d\t%d\t%d\n" % (label, n_train, targets[label], n_dev, n_test))
    print("wrote", ARGS.out, file=sys.stderr)


if __name__ == "__main__":
    main()
