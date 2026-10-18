#!/usr/bin/env python3
# Copyright (c) 2026 CommonGlot contributors. MIT License.
"""Run a ByteFast ablation: the same data and tuning budget for every variant.

Every variant (including the fastText recipes) is trained with the same
bytefast binary, so differences come only from the flags. For each variant:

1. grid search over --grid on dev.txt (first seed), select by dev macro F1;
2. retrain the selected setting with every seed and score test.txt once;
3. for the first seed, measure model size, model load time and warm
   single-thread prediction throughput (load time is subtracted, see
   --throughput-lines). Scoring dev and test uses --threads.

DATA must contain train.txt, dev.txt and test.txt in ByteFast format.
VARIANTS is a JSON file:

  {"common": "-dim 64 -bucket 1048576",
   "variants": [{"name": "words", "flags": "-minCount 1"},
                {"name": "bytes", "flags": "-minb 2 -maxb 5 -minCount 1000000000"},
                {"name": "recipe", "flags": "-minn 2 -maxn 5",
                 "grid": {"lr": ["0.8"], "epoch": ["2"]}}]}

A variant's own "grid" replaces --grid for that variant (for published
recipes that must run with their exact settings).

  python3 benchmarks/ablate.py --data data/wili --variants benchmarks/ablations/units.json \\
      --out results/wili-units.jsonl --threads 32 --grid lr=0.1,0.5,1.0 epoch=5,15
"""
import argparse
import itertools
import json
import os
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


def labels_of(path, prefix="__label__"):
    gold = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            first = next((t for t in line.split() if t.startswith(prefix)), "")
            gold.append(first[len(prefix):])
    return gold


def score(gold, pred):
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            if p:
                fp[p] += 1
    per_label = {}
    for label in sorted(set(gold)):
        prec = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        rec = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        per_label[label] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "accuracy": sum(tp.values()) / len(gold),
        "macro_f1": sum(per_label.values()) / len(per_label),
        "per_label_f1": per_label,
    }


def run(cmd, stdout=subprocess.DEVNULL):
    start = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=stdout, stderr=subprocess.PIPE)
    return time.perf_counter() - start


def predict(binary, model, path, threads=1):
    out = subprocess.run([binary, "predict", model, str(path), "1", "0.0", str(threads)],
                         check=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = out.stdout.decode("utf-8", "replace").split("\n")[:-1]
    return [l.split()[0][len("__label__"):] if l.strip() else "" for l in lines]


def evaluate(binary, model, path, threads=1):
    gold = labels_of(path)
    pred = predict(binary, model, path, threads)
    if len(pred) != len(gold):
        raise RuntimeError("prediction count %d != %d for %s" % (len(pred), len(gold), path))
    return score(gold, pred)


def throughput(binary, model, test, lines, workdir, repeats=3):
    texts = [" ".join(t for t in l.split() if not t.startswith("__label__"))
             for l in open(test, encoding="utf-8", errors="replace")]
    big = Path(workdir) / "throughput.txt"
    with big.open("w", encoding="utf-8") as out:
        for i in range(lines):
            out.write(texts[i % len(texts)] + "\n")
    empty = Path(workdir) / "empty.txt"
    empty.write_text("")
    load = statistics.median(run([binary, "predict", model, str(empty)]) for _ in range(repeats))
    full = statistics.median(run([binary, "predict", model, str(big)]) for _ in range(repeats))
    big.unlink()
    return load, lines / max(full - load, 1e-9)


def train(binary, data, model, flags, threads, seed):
    """Return (seconds, None), or (None, error) if training failed (e.g. NaN)."""
    cmd = [binary, "supervised", "-input", str(Path(data) / "train.txt"),
           "-output", model, "-thread", str(threads), "-seed", str(seed),
           "-verbose", "0"] + flags
    try:
        return run(cmd), None
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
        return None, tail[0] if tail else "exit status %d" % e.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", default="./bytefast")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--variants", required=True, type=Path)
    p.add_argument("--only", nargs="*", help="run only these variant names")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--grid", nargs="*", default=["lr=0.5", "epoch=5"],
                   help="NAME=v1,v2 lists; every combination is tried on dev")
    p.add_argument("--seeds", default="1", help="comma-separated seeds")
    p.add_argument("--throughput-lines", type=int, default=200000)
    p.add_argument("--keep-models", type=Path, default=None)
    a = p.parse_args()

    spec = json.loads(a.variants.read_text())
    common = shlex.split(spec.get("common", ""))
    variants = [v for v in spec["variants"] if not a.only or v["name"] in a.only]
    grid_names, grid_values = [], []
    for item in a.grid:
        name, values = item.split("=", 1)
        grid_names.append(name)
        grid_values.append(values.split(","))
    seeds = [int(s) for s in a.seeds.split(",")]
    a.out.parent.mkdir(parents=True, exist_ok=True)

    summary = []
    with tempfile.TemporaryDirectory(prefix="bytefast-ablate-") as work, \
            a.out.open("a", encoding="utf-8") as log:
        model = str(Path(work) / "model")
        for v in variants:
            flags = common + shlex.split(v["flags"])
            names, values = grid_names, grid_values
            if "grid" in v:
                names, values = list(v["grid"]), [list(map(str, x)) for x in v["grid"].values()]
            best = None
            for combo in itertools.product(*values):
                grid = [x for name, value in zip(names, combo)
                        for x in ("-" + name, value)]
                seconds, error = train(a.bin, a.data, model, flags + grid, a.threads, seeds[0])
                if error:
                    record = {"stage": "dev", "variant": v["name"],
                              "grid": dict(zip(names, combo)), "error": error}
                    log.write(json.dumps(record) + "\n"); log.flush()
                    print("%-24s dev %s FAILED: %s" % (v["name"], record["grid"], error), flush=True)
                    continue
                dev = evaluate(a.bin, model + ".safetensors", a.data / "dev.txt", a.threads)
                record = {"stage": "dev", "variant": v["name"], "grid": dict(zip(names, combo)),
                          "dev_macro_f1": dev["macro_f1"], "dev_accuracy": dev["accuracy"],
                          "train_seconds": seconds}
                log.write(json.dumps(record) + "\n"); log.flush()
                print("%-24s dev %s macroF1=%.4f (%.1fs)" % (v["name"], record["grid"],
                                                              dev["macro_f1"], seconds), flush=True)
                if best is None or dev["macro_f1"] > best[0]:
                    best = (dev["macro_f1"], grid, record["grid"])
            if best is None:
                print("%-24s every grid setting failed; skipped" % v["name"], flush=True)
                continue
            tests = []
            for seed in seeds:
                seconds, error = train(a.bin, a.data, model, flags + best[1], a.threads, seed)
                if error:
                    log.write(json.dumps({"stage": "test", "variant": v["name"], "grid": best[2],
                                          "seed": seed, "error": error}) + "\n"); log.flush()
                    continue
                test = evaluate(a.bin, model + ".safetensors", a.data / "test.txt", a.threads)
                record = {"stage": "test", "variant": v["name"], "flags": " ".join(flags),
                          "grid": best[2], "seed": seed, "dev_macro_f1": best[0],
                          "test_macro_f1": test["macro_f1"], "test_accuracy": test["accuracy"],
                          "train_seconds": seconds, "threads": a.threads,
                          "per_label_f1": test["per_label_f1"]}
                if not tests:  # first successful seed: size and speed
                    record["model_bytes"] = os.path.getsize(model + ".safetensors")
                    load, lps = throughput(a.bin, model + ".safetensors", a.data / "test.txt",
                                           a.throughput_lines, work)
                    record["load_seconds"] = load
                    record["predict_lines_per_second"] = lps
                    if a.keep_models:
                        a.keep_models.mkdir(parents=True, exist_ok=True)
                        os.replace(model + ".safetensors", a.keep_models / ("%s.safetensors" % v["name"]))
                log.write(json.dumps(record) + "\n"); log.flush()
                tests.append(record)
            if not tests:
                continue
            f1 = [t["test_macro_f1"] for t in tests]
            summary.append((v["name"], best[2], statistics.mean(f1),
                            statistics.stdev(f1) if len(f1) > 1 else 0.0,
                            statistics.mean(t["test_accuracy"] for t in tests),
                            tests[0]["model_bytes"], tests[0]["load_seconds"],
                            tests[0]["predict_lines_per_second"],
                            statistics.mean(t["train_seconds"] for t in tests)))

    print("\n| variant | selected | test macro F1 | test acc | size MB | load s | lines/s | train s |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, grid, f1, sd, acc, size, load, lps, secs in summary:
        print("| %s | %s | %.4f ± %.4f | %.4f | %.1f | %.2f | %.0f | %.1f |" % (
            name, " ".join("%s=%s" % kv for kv in grid.items()), f1, sd, acc,
            size / 1e6, load, lps, secs))


if __name__ == "__main__":
    main()
