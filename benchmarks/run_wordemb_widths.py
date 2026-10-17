#!/usr/bin/env python3
"""Controlled word-vector width sweep: 16, 32, 64, 128, 256, 512.

All widths use identical WiLI training data, selected word vocabulary size,
frequency cutoff, loss, optimizer, epochs and learning rate. Select based
on the train-derived development split. The official WiLI test split is
evaluated only for the development-selected model and reference controls.

Exploratory only: previous WiLI official-test results have been observed.
Confirm any claimed gain using an independent held-out data source.
"""
import argparse
import json
import subprocess
from pathlib import Path

from run_wordemb_experiment import load_labels, scores, walltime


def run(argv, output=None):
    if output is None:
        subprocess.run(argv, check=True)
    else:
        with open(output, "w", encoding="utf-8") as f:
            subprocess.run(argv, stdout=f, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/tmp/wili")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--vocab", type=int, default=40000)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    args = parser.parse_args()

    root = Path(args.workdir)
    base = root / "word-byte.bin"
    train = root / "lex-train.fasttext"
    devtxt = root / "lex-dev.txt"
    devgold = load_labels(root / "lex-dev.fasttext")
    testtxt = root / "test.txt"
    testgold = load_labels(root / "test.fasttext")
    widths = (16, 32, 64, 128, 256, 512)

    evaluations = []
    for dim in widths:
        model = root / f"width-{dim}.bin"
        print(f"WIDTH_TRAIN dim={dim} epochs={args.epochs} words={args.vocab}",
              flush=True)
        run([
            "./bytefast-wordemb", "train", str(base), str(train), str(model),
            "-vocab", str(args.vocab), "-dim", str(dim),
            "-minCount", "3", "-epoch", str(args.epochs),
            "-lr", str(args.learning_rate), "-selection", "confusion",
        ])
        predict = [
            "./bytefast-wordemb", "predict", str(base),
            str(model), str(devtxt),
        ]
        output = root / f"width-{dim}-dev.pred"
        run(predict, output)
        result = scores(devgold, load_labels(output))
        entry = {
            "dimension": dim,
            "epochs": args.epochs,
            "vocab_cap": args.vocab,
            "dev": result,
            "dev_score": round(
                result["macro_f1"] + 0.12 * (result["bcms_f1"] or 0), 9),
            "model_bytes": base.stat().st_size + model.stat().st_size,
            "dev_wall_seconds_median_3": walltime(predict),
            "word_model": str(model),
        }
        evaluations.append(entry)
        print("WIDTH_DEV", json.dumps(entry, sort_keys=True), flush=True)

    selected = max(
        evaluations,
        key=lambda e: (e["dev_score"], e["dev"]["macro_f1"],
                       -e["model_bytes"]),
    )
    print("WIDTH_SELECTED_ON_DEV",
          json.dumps(selected, sort_keys=True), flush=True)

    reference = [
        ("base_byte", [
            "./bytefast-proto", "predict", str(base), str(testtxt),
        ], base.stat().st_size),
        ("selected_learned_words", [
            "./bytefast-wordemb", "predict", str(base),
            selected["word_model"], str(testtxt),
        ], selected["model_bytes"]),
    ]
    lexicon = root / "word-sparse.lex"
    if lexicon.is_file():
        reference.append(("gated_statistical_lexicon", [
            "./bytefast-lexical", "predict", str(base), str(lexicon),
            str(testtxt), "-weight", "3", "-gateMargin", "0.75",
        ], base.stat().st_size + lexicon.stat().st_size))
    fasttext = root / "word-fasttext.bin"
    if fasttext.is_file():
        reference.append(("ordinary_fasttext", [
            "./bytefast", "predict", str(fasttext), str(testtxt),
        ], fasttext.stat().st_size))

    test = []
    for name, argv, size in reference:
        filename = root / f"width-{name}-test.pred"
        run(argv, filename)
        result = scores(testgold, load_labels(filename))
        result.update({
            "name": name, "model_bytes": size,
            "wall_seconds_median_3": walltime(argv),
        })
        test.append(result)
        print("WIDTH_TEST", json.dumps(result, sort_keys=True), flush=True)

    report = {
        "selection_only_uses_train_derived_development": True,
        "external_validation_still_required": True,
        "note": "Prior WiLI official-test evaluations have already been viewed; "
                "treat this follow-up as exploratory.",
        "dev_candidates": evaluations,
        "selected_on_dev": selected,
        "official_test_comparison": test,
    }
    (root / "wordemb-width-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
