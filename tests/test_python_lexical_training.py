#!/usr/bin/env python3
"""Regression tests for the prototype lexical training API (bytefast.proto)."""
from pathlib import Path
import tempfile

import bytefast
from bytefast.proto import _split_lexical_training


def make_input(root):
    path = root / "classes.fasttext"
    examples = {
        "sports": (
            "The tournament team won with a late goal in the match.",
            "A defender stopped the player before the championship.",
            "The coach watched the team practice for the game.",
        ),
        "finance": (
            "The stock market earnings report increased investor interest.",
            "Banks raised the interest rate on new loans.",
            "The quarterly financial results affected share prices.",
        ),
        "technology": (
            "The computer processor supports the new software.",
            "An engineer designed a semiconductor with new transistors.",
            "The developer fixed the program using a faster algorithm.",
        ),
    }
    rows = [
        "__label__%s %s Example %d" % (label, texts[i % len(texts)], i)
        for label, texts in examples.items()
        for i in range(40)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = make_input(root)
        split = root / "split"
        split.mkdir()
        fit, cal, dev = _split_lexical_training(source, split, seed=42)
        assert all(p.exists() for p in (fit, cal, dev))
        lines = [set(p.read_text().splitlines()) for p in (fit, cal, dev)]
        assert not lines[0].intersection(lines[1])
        assert not lines[0].intersection(lines[2])
        assert not lines[1].intersection(lines[2])
        assert tuple(map(len, lines)) == (96, 12, 12)
        assert len(set.union(*lines)) == 120

        bare = bytefast.proto.train_supervised(
            source, dim=16, bucket=2048, epoch=5, lr=0.2)
        assert bare.model_type == "bytefast"

        output = root / "combined.bin"
        trained = bytefast.proto.train_supervised(
            source,
            output=output,
            dim=16,
            bucket=2048,
            epoch=5,
            lr=0.2,
            lexical=True,
            lexical_vocab=100,
            lexical_min_df=1,
            lexical_selection="global",
            lexical_seed=42,
        )
        assert output.read_bytes().startswith(b"BFBN")
        assert trained.model_type == "bytefast-calibrated"
        assert set(trained.get_labels()) == {
            "__label__sports", "__label__finance",
            "__label__technology",
        }
        loaded = bytefast.load_model(output)
        for text in (
            "The match ended in a championship goal",
            "Banks announced increased stock market earnings",
            "The software developer designed a processor",
        ):
            a = trained.predict(text, k=2)
            b = loaded.predict(text, k=2)
            assert a[0] == b[0], (a, b)
            assert len(a[0]) == len(a[1]) == 2

        # With no output file, the returned model owns its backing directory
        # and remains usable after training exits.
        memory_model = bytefast.proto.train_supervised(
            source, dim=16, bucket=2048, epoch=3, lr=0.2,
            lexical=True, lexical_vocab=80, lexical_min_df=1)
        assert memory_model.model_type == "bytefast-calibrated"
        assert memory_model.predict("The coach and team scored")[0]

        saved_again = root / "saved-from-memory.bin"
        memory_model.save_model(saved_again)
        assert saved_again.read_bytes().startswith(b"BFBN")
        assert bytefast.load_model(saved_again).get_labels() == (
            memory_model.get_labels())

        external = bytefast.proto.train_supervised(
            fit, lexical=True, lexical_calibration_data=cal,
            lexical_validation_data=dev,
            lexical_vocab=100, lexical_min_df=1,
            dim=16, bucket=2048, epoch=3, lr=0.2,
        )
        assert external.model_type == "bytefast-calibrated"
        assert external.predict("A computer processor")[0]

        # Every training setting is passed through, including the byte
        # encoder ablations, the automatic split and the fusion optimizer.
        custom_split = root / "custom-split"
        custom_split.mkdir()
        sizes = tuple(
            len(p.read_text().splitlines()) for p in _split_lexical_training(
                source, custom_split, seed=7, split=(0.6, 0.2, 0.2)))
        assert sizes == (72, 24, 24), sizes
        configured = bytefast.proto.train_supervised(
            source, dim=16, bucket=2048, epoch=3, lr=0.2,
            lexical=True, lexical_vocab=80, lexical_min_df=1,
            lexical_split=(0.6, 0.2, 0.2), lexical_fusion_epochs=5,
            lexical_fusion_lr=0.05, lexical_fusion_l2=0.0,
            lexical_fusion_patience=2,
        )
        assert configured.model_type == "bytefast-calibrated"
        assert configured.predict("Banks and the stock market")[0]

        for bad_params in (
            {"lexical_selection": "confusion"},
            {"lexical_selection": "balanced"},
            {"lexical_vocab": 0},
            {"lexical_min_df": 0},
            {"lexical_calibration_data": cal},
            {"lexical_split": (0.8, 0.2)},
            {"lexical_split": (0.9, 0.1, 0.0)},
            {"lexical_split": (0.5, 0.1, 0.1)},
            {"lexical_fusion_epochs": 0},
        ):
            try:
                bytefast.proto.train_supervised(source, lexical=True, **bad_params)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid lexical option accepted: %r" %
                                     bad_params)

        try:
            bytefast.proto.train_supervised(source, lexical=False,
                                     lexical_validation_data=dev)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted calibration data with lexical=False")

    print("BYTEFAST_PYTHON_LEXICAL_TRAINING_PASS")


if __name__ == "__main__":
    main()
