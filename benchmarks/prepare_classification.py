#!/usr/bin/env python3
# Copyright (c) 2026 CommonGlot contributors. MIT License.
"""Download a classification dataset and write train/dev/test in ByteFast format.

Datasets:
  wili     WiLI-2018 language identification (235 labels, Zenodo, ODbL)
  dbpedia  DBpedia ontology topics (14 labels, fancyzhx/dbpedia_14)
  ag_news  AG News topics (4 labels, fancyzhx/ag_news)
  yelp     Yelp review polarity (2 labels, fancyzhx/yelp_polarity)

The topic and sentiment sets are the Zhang et al. (2015) splits used in the
fastText paper; their original download URL is no longer served, so they
are read from the Hugging Face copies (needs huggingface_hub and pyarrow).
Labels are the class indices of those copies.

The text is kept as published (no lowercasing or tokenization) so that case
folding and tokenization remain ablation choices. dev is a deterministic,
class-stratified slice of the official training split; test is the official
test split.

  python3 benchmarks/prepare_classification.py --dataset dbpedia --out data/dbpedia
"""
import argparse
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

WILI = "https://zenodo.org/records/841984/files/wili-2018.zip?download=1"
HF_DATASETS = {
    "ag_news": ("fancyzhx/ag_news", ["text"]),
    "dbpedia": ("fancyzhx/dbpedia_14", ["title", "content"]),
    "yelp": ("fancyzhx/yelp_polarity", ["text"]),
}


def clean(text):
    return " ".join(text.replace("\\n", " ").split())


def download(url, path):
    if not path.exists():
        print("downloading", url, file=sys.stderr)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def read_hf_dataset(name, cache):
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq

    repo, columns = HF_DATASETS[name]
    root = Path(snapshot_download(repo_id=repo, repo_type="dataset",
                                  allow_patterns=["*.parquet"],
                                  local_dir=cache / repo.replace("/", "__")))
    splits = {}
    for split in ("train", "test"):
        rows = []
        for path in sorted(root.rglob("*.parquet")):
            if not path.name.startswith(split):
                continue
            table = pq.read_table(path, columns=["label"] + columns).to_pydict()
            for i, label in enumerate(table["label"]):
                text = clean(" ".join(str(table[c][i]) for c in columns))
                if text:
                    rows.append((str(label), text))
        if not rows:
            raise RuntimeError("no %s parquet files for %s" % (split, repo))
        splits[split] = rows
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
    p.add_argument("--dataset", required=True, choices=["wili"] + sorted(HF_DATASETS))
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

    train, test = read_wili(cache) if a.dataset == "wili" else read_hf_dataset(a.dataset, cache)
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
