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
            "bytes+words": dict(minb=2, maxb=5, bucket=50000, max_bytes=64),
        }
        for name, cfg in configs.items():
            model = train(data, **cfg)
            assert model.predict(query)[0] == ("__label__sports",), name
            path = root / ("%s.safetensors" % name)
            model.save_model(str(path))
            # A safetensors file: header length, then the JSON header.
            assert path.read_bytes()[8:9] == b"{", name
            loaded = bytefast.load_model(str(path))
            assert loaded.predict(query)[0] == model.predict(query)[0], name
            args = loaded.f.getArgs()
            for key, value in cfg.items():
                attr = {"max_bytes": "maxBytes"}.get(key, key)
                if attr == "minCount":
                    continue  # a training-time setting
                got = getattr(args, attr)
                assert got == value, (name, attr, got, value)

        # Vocabulary control: a wordlist and the most frequent words.
        wordlist = root / "words.txt"
        # Text after the word is ignored; entries match words exactly, so
        # "investors" does not match "Investors" and "Telescope" is not in
        # the data.
        wordlist.write_text("telescope 10\nbank\ninvestors\nTelescope\n", encoding="utf-8")
        listed = train(data, wordlist=str(wordlist), minb=2, maxb=4, bucket=20000)
        assert sorted(listed.get_words()) == [
            "</s>", "bank", "telescope"], listed.get_words()
        capped = train(data, max_vocab=5, minn=2, maxn=4)
        assert len(capped.get_words()) == 5

        # The lexicon check: sports and science have lexicons without the
        # query's words and finance one with all of them, so finance comes
        # first (a label without a lexicon would always pass); it is kept in
        # the model file, and threshold 0 turns it off.
        lexicon = root / "lexicon.tsv"
        lexicon.write_text("sports\tzzz\nscience\tzzz\n" + "".join(
            "__label__finance\t%s\n" % w for w in query.split()), encoding="utf-8")
        checked = train(data, minb=2, maxb=5, bucket=50000, lexicon=str(lexicon))
        assert checked.predict(query)[0] == ("__label__finance",)
        assert checked.predict("goal")[0] == ("__label__sports",)  # one word: not checked
        path = root / "lexicon.safetensors"
        checked.save_model(str(path))
        loaded = bytefast.load_model(str(path))
        assert loaded.predict(query)[0] == ("__label__finance",)
        loaded.set_lexicon_threshold(0.0)
        assert loaded.predict(query)[0] == ("__label__sports",)
        # A label without a lexicon (science here) is never moved up.
        lexicon.write_text("sports\tzzz\n" + "".join(
            "__label__finance\t%s\n" % w for w in query.split()), encoding="utf-8")
        partial = train(data, minb=2, maxb=5, bucket=50000, lexicon=str(lexicon))
        assert partial.predict(query)[0] == ("__label__finance",)

        # Presets: the language-ID settings, overridable.
        lid = train(data, preset="lid", minCount=1, bucket=50000, epoch=15, lr=0.5)
        args = lid.f.getArgs()
        assert (args.minb, args.maxb, args.maxBytes, args.minn, args.maxVocab) == (2, 5, 512, 0, 20000)
        assert args.bucket == 50000
        accurate = train(data, preset="lid-accurate", minCount=1, bucket=50000)
        assert (accurate.f.getArgs().minn, accurate.f.getArgs().maxn) == (2, 5)
        assert lid.predict(query)[0] == ("__label__sports",)

        # Label weights: a label of weight 0 is never trained.
        weights = root / "weights.tsv"
        weights.write_text("sports\t0\n__label__science\t2.5\n", encoding="utf-8")
        weighted = train(data, minb=2, maxb=5, bucket=50000, label_weights=str(weights))
        assert weighted.predict(query)[0] != ("__label__sports",)

        # Training on random character spans, normalized and stripped lines.
        cropped = train(data, minb=2, maxb=5, bucket=50000, minCount=10 ** 9,
                        crop_prob=0.5, crop_min=5, crop_max=30, norm_prob=0.3,
                        strip_prob=0.3)
        assert cropped.predict("telescope")[0][0].startswith("__label__")

        for bad in ({"minb": 3, "maxb": 2}, {"maxb": 4, "minb": 1, "bucket": 0},
                    {"max_bytes": -1}, {"crop_prob": 1.5},
                    {"crop_prob": 0.5, "crop_max": 0}, {"strip_prob": -0.1},
                    {"lex_threshold": 1.5}, {"lex_top_k": 0}):
            try:
                train(data, **bad)
            except ValueError:
                pass
            else:
                raise AssertionError("accepted invalid settings %r" % (bad,))

    print("BYTEFAST_TEXT_FEATURES_PASS")


if __name__ == "__main__":
    main()
