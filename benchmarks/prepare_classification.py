#!/usr/bin/env python3
# Copyright (c) 2026 CommonGlot contributors. MIT License.
"""Download a classification dataset and write train/dev/test in ByteFast format.

Datasets:
  wili     WiLI-2018 language identification (235 labels, Zenodo, ODbL)
  dbpedia  DBpedia ontology topics (14 labels)
  ag_news  AG News topics (4 labels)
  yelp     Yelp review polarity (2 labels)

The text is kept as published (no lowercasing or tokenization) so that case
folding and tokenization remain ablation choices. dev is a deterministic,
class-stratified slice of the official training split; test is the official
test split.

  python3 benchmarks/prepare_classification.py --dataset dbpedia --out data/dbpedia
"""
import argparse
import csv
import io
import random
import sys
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

FASTTEXT_DATA = "https://dl.fbaipublicfiles.com/fasttext/data/%s_csv.tar.gz"
WILI = "https://zenodo.org/records/841984/files/wili-2018.zip?download=1"
CSV_DATASETS = {"dbpedia": "dbpedia", "ag_news": "ag_news",
                "yelp": "yelp_review_polarity"}


def clean(text):
    return " ".join(text.replace("\\n", " ").split())


def download(url, path):
    if not path.exists():
        print("downloading", url, file=sys.stderr)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def read_csv_dataset(name, cache):
    archive = download(FASTTEXT_DATA % CSV_DATASETS[name],
                       cache / ("%s_csv.tar.gz" % name))
    splits = {}
    csv.field_size_limit(1 << 24)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            stem = Path(member.name).name
            if stem not in ("train.csv", "test.csv"):
                continue
            rows = []
            data = io.TextIOWrapper(tar.extractfile(member), encoding="utf-8")
            for row in csv.reader(data):
                text = clean(" ".join(row[1:]))
                if text:
                    rows.append((row[0], text))
            splits[stem[:-4]] = rows
    return splits["train"], splits["test"]


def read_wili(cache):
    archive = download(WILI, cache / "wili-2018.zip")
    splits = {}
    with zipfile.ZipFile(archive) as z:
        names = {Path(n).name: n for n in z.namelist()}
        for split in ("train", "test"):
            texts = z.read(names["x_%s.txt" % split]).decode("utf-8").splitlines()
            labels = z.read(names["y_%s.txt" % split]).decode("utf-8").splitlines()
            splits[split] = [(y.strip(), clean(x)) for x, y in zip(texts, labels)
                             if y.strip() and clean(x)]
    return splits["train"], splits["test"]


def write(path, rows):
    with path.open("w", encoding="utf-8") as out:
        for label, text in rows:
            out.write("__label__%s %s\n" % (label, text))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["wili"] + sorted(CSV_DATASETS))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--dev-fraction", type=float, default=0.1)
    p.add_argument("--max-train-per-label", type=int, default=0,
                   help="cap training examples per label (0 = all)")
    p.add_argument("--seed", type=int, default=1729)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.cache or a.out
    cache.mkdir(parents=True, exist_ok=True)

    train, test = read_wili(cache) if a.dataset == "wili" else read_csv_dataset(a.dataset, cache)
    by_label = defaultdict(list)
    for row in train:
        by_label[row[0]].append(row)
    rng = random.Random(a.seed)
    fit, dev = [], []
    for label in sorted(by_label):
        rows = by_label[label]
        rng.shuffle(rows)
        n_dev = max(1, round(len(rows) * a.dev_fraction))
        dev += rows[:n_dev]
        rest = rows[n_dev:]
        if a.max_train_per_label:
            rest = rest[:a.max_train_per_label]
        fit += rest
    # Shuffle once so that streaming trainers see mixed labels.
    rng.shuffle(fit)
    write(a.out / "train.txt", fit)
    write(a.out / "dev.txt", dev)
    write(a.out / "test.txt", test)
    print("%s: train=%d dev=%d test=%d labels=%d -> %s" % (
        a.dataset, len(fit), len(dev), len(test), len(by_label), a.out))


if __name__ == "__main__":
    main()
