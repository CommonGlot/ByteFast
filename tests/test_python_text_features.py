#!/usr/bin/env python3
# Copyright (c) 2026 CommonGlot contributors. MIT License.
"""Regression tests for ByteFast text features through the Python API."""
from pathlib import Path
import tempfile

import bytefast

EXAMPLES = {
    "sports": ["The team won the match with a late goal.",
               "A defender stopped the striker before the final whistle.",
               "The coach praised the players after the tournament."],
    "finance": ["The bank raised interest rates on new loans.",
                "Quarterly earnings lifted the company's share price.",
                "Investors sold bonds after the inflation report."],
    "science": ["The telescope observed light from a distant galaxy.",
                "Researchers sequenced the genome of a rare plant.",
                "The experiment measured the mass of the particle."],
}


def write_data(root):
    rows = ["__label__%s %s Story %d" % (label, texts[i % len(texts)], i)
            for label, texts in EXAMPLES.items() for i in range(60)]
    train = root / "train.txt"
    train.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return train


def train(train, **kwargs):
    params = dict(dim=16, epoch=15, lr=0.5, thread=1, verbose=0, seed=3)
    params.update(kwargs)
    return bytefast.train_supervised(input=str(train), **params)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = write_data(root)
        query = "The striker scored a goal in the final match"

        configs = {
            "subwords": dict(minn=2, maxn=5),
            "bytes": dict(minb=2, maxb=5, bucket=50000, minCount=10 ** 9),
            "chars": dict(minc=1, maxc=4, bucket=50000, minCount=10 ** 9),
            "bytes+words": dict(minb=2, maxb=5, bucket=50000, hash_words=True,
                                fold="unicode", max_bytes=64),
        }
        for name, cfg in configs.items():
            model = train(data, **cfg)
            assert model.predict(query)[0] == ("__label__sports",), name
            path = root / ("%s.bin" % name)
            model.save_model(str(path))
            magic = path.read_bytes()[:4]
            # Only models with text features need the ByteFast format.
            expected = b"\xba\x16\x4f\x2f" if name == "subwords" else b"BFCM"
            assert magic == expected, (name, magic)
            loaded = bytefast.load_model(str(path))
            assert loaded.predict(query)[0] == model.predict(query)[0], name
            args = loaded.f.getArgs()
            for key, value in cfg.items():
                attr = {"hash_words": "hashWords", "max_bytes": "maxBytes"}.get(key, key)
                if attr == "minCount":
                    continue  # a training-time setting
                got = getattr(args, attr)
                if attr == "fold":
                    got = str(got).split(".")[-1]
                assert got == value, (name, attr, got, value)

        # Case folding makes predictions independent of letter case.
        folded = train(data, minb=2, maxb=5, bucket=50000, fold="unicode")
        upper = folded.predict(query.upper(), k=3)
        lower = folded.predict(query.lower(), k=3)
        assert upper[0] == lower[0] and abs(upper[1][0] - lower[1][0]) < 1e-6

        # Vocabulary control: wordlist and maxVocab.
        wordlist = root / "words.txt"
        # Entries are folded like the input; text after the word is ignored.
        wordlist.write_text("Telescope 10\nbank\nINVESTORS\n", encoding="utf-8")
        listed = train(data, wordlist=str(wordlist), fold="unicode",
                       minb=2, maxb=4, bucket=20000)
        assert sorted(listed.get_words()) == [
            "</s>", "bank", "investors", "telescope"], listed.get_words()
        capped = train(data, max_vocab=5, minn=2, maxn=4)
        assert len(capped.get_words()) == 5

        for bad in ({"fold": "upper"}, {"minb": 3, "maxb": 2},
                    {"minc": 0, "maxc": 3}, {"maxb": 4, "minb": 1, "bucket": 0},
                    {"max_bytes": -1}):
            try:
                train(data, **bad)
            except ValueError:
                pass
            else:
                raise AssertionError("accepted invalid settings %r" % (bad,))

    print("BYTEFAST_TEXT_FEATURES_PASS")


if __name__ == "__main__":
    main()
