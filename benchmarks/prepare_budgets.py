#!/usr/bin/env python3
"""Create deterministic, nested, per-language training budgets from WiLI.

Only consumes the official training partition. It does not touch test data.
"""
import argparse
import collections
import hashlib
import pathlib
import random


def make_budget(lines, cap, seed, skew=False):
    groups = collections.defaultdict(list)
    for line in lines:
        if not line.startswith("__label__") or " " not in line:
            continue
        groups[line.split(" ", 1)[0]].append(line)
    if not groups:
        raise ValueError("No labeled examples")
    selected = []
    counts = {}
    for rank, label in enumerate(sorted(groups)):
        salt = int.from_bytes(
            hashlib.blake2b(label.encode(), digest_size=8).digest(), "little")
        local = list(groups[label])
        random.Random(seed ^ salt).shuffle(local)
        budget = cap if not skew else (
            5 if rank % 5 == 0 else 25 if rank % 5 in (1, 2) else cap)
        take = min(budget, len(local))
        counts[label] = take
        selected.extend(local[:take])
    random.Random(seed).shuffle(selected)
    return selected, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--per-language", type=int, required=True)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--skew", action="store_true")
    args = p.parse_args()
    if args.per_language <= 0:
        p.error("per-language must be positive")
    text = pathlib.Path(args.input).read_text(encoding="utf-8").splitlines(keepends=True)
    selected, counts = make_budget(text, args.per_language, args.seed, args.skew)
    pathlib.Path(args.output).write_text("".join(selected), encoding="utf-8")
    print("budget=", args.per_language, "skew=", args.skew,
          "train=", len(selected), "labels=", len(counts),
          "min=", min(counts.values()), "max=", max(counts.values()),
          flush=True)


if __name__ == "__main__":
    main()
