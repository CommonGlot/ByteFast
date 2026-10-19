# Copyright (c) 2017-present, Facebook, Inc., and 2026 CommonGlot contributors.
# MIT License.
"""Python API: train_supervised, load_model and the model object."""

import multiprocessing

import numpy as np

from . import _core_pybind as _core

LOSSES = {
    "softmax": _core.loss_name.softmax,
}

# Training arguments and their defaults. The snake_case aliases below are
# accepted as keyword arguments too.
DEFAULTS = {
    "input": "",
    "lr": 0.1,
    "dim": 100,
    "epoch": 5,
    "minCount": 1,
    "minCountLabel": 0,
    "minn": 0,
    "maxn": 0,
    "wordNgrams": 1,
    "loss": "softmax",
    "bucket": 2000000,
    "thread": max(1, multiprocessing.cpu_count() - 1),
    "lrUpdateRate": 100,
    "label": "__label__",
    "verbose": 2,
    "seed": 0,
    "minb": 0,
    "maxb": 0,
    "maxBytes": 0,
    "wordlist": "",
    "maxVocab": 0,
    "cropProb": 0.0,
    "cropMin": 1,
    "cropMax": 0,
    "normProb": 0.0,
    "stripProb": 0.0,
    "saveEpochs": False,
    "validation": "",
}

ALIASES = {
    "min_count": "minCount",
    "min_count_label": "minCountLabel",
    "word_ngrams": "wordNgrams",
    "lr_update_rate": "lrUpdateRate",
    "label_prefix": "label",
    "max_bytes": "maxBytes",
    "max_vocab": "maxVocab",
    "crop_prob": "cropProb",
    "crop_min": "cropMin",
    "crop_max": "cropMax",
    "norm_prob": "normProb",
    "strip_prob": "stripProb",
    "save_epochs": "saveEpochs",
}


def _line(text):
    if "\n" in text:
        raise ValueError("predict processes one line at a time (remove '\\n')")
    return text + "\n"


class _Model:
    """A trained classifier, returned by train_supervised and load_model."""

    def __init__(self, f):
        self.f = f
        self._labels = None
        self._words = None

    def predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
        """Top-k labels and probabilities of one line, or of a list of lines.

        A string gives (labels, probabilities); a list gives a list of labels
        per line and a list of probability arrays.
        """
        if isinstance(text, list):
            return self.f.multilinePredict(
                [_line(t) for t in text], k, threshold, on_unicode_error)
        predictions = self.f.predict(_line(text), k, threshold, on_unicode_error)
        probs, labels = zip(*predictions) if predictions else ((), ())
        return labels, np.asarray(probs)

    def test(self, path, k=1, threshold=0.0):
        """(number of examples, precision@k, recall@k) on a labeled file."""
        return self.f.test(path, k, threshold)

    def test_label(self, path, k=1, threshold=0.0):
        """Precision, recall and F1 of every label on a labeled file."""
        return self.f.testLabel(path, k, threshold)

    def save_model(self, path):
        self.f.saveModel(path)

    def get_dimension(self):
        return self.f.getArgs().dim

    def get_labels(self, include_freq=False, on_unicode_error="strict"):
        labels, freq = self.f.getLabels(on_unicode_error)
        return (labels, np.array(freq)) if include_freq else labels

    def get_words(self, include_freq=False, on_unicode_error="strict"):
        words, freq = self.f.getVocab(on_unicode_error)
        return (words, np.array(freq)) if include_freq else words

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.get_labels()
        return self._labels

    @property
    def words(self):
        if self._words is None:
            self._words = self.get_words()
        return self._words


def load_model(path):
    """Load a model file and return the model object."""
    f = _core.bytefast()
    f.loadModel(str(path))
    return _Model(f)


def train_supervised(**kwargs):
    """Train a classifier and return the model object.

    input is a UTF-8 file with one example per line and at least one label
    (``__label__NAME`` by default) per line. Each example is represented by
    the average of its feature vectors, and the features combine freely:

    - words (minCount, wordlist, maxVocab)
    - minn/maxn: character n-grams inside each word
    - wordNgrams: hashed word n-grams
    - minb/maxb: byte n-grams over the whole text; maxBytes limits how much
      of the text they read

    crop_prob/crop_min/crop_max train on random character spans of lines,
    norm_prob on normalized lines and strip_prob on lines without optional
    diacritics.
    """
    args = dict(DEFAULTS)
    for name, value in kwargs.items():
        name = ALIASES.get(name, name)
        if name not in DEFAULTS:
            raise TypeError("unexpected keyword argument '%s'" % name)
        args[name] = value
    if args["loss"] not in LOSSES:
        raise ValueError("Unrecognized loss name: %s" % args["loss"])
    args["loss"] = LOSSES[args["loss"]]
    a = _core.args()
    for name, value in args.items():
        setattr(a, name, value)
    # Without hashed features the bucket table is unused; keep it empty.
    if a.wordNgrams <= 1 and a.maxn == 0 and a.maxb == 0:
        a.bucket = 0
    f = _core.bytefast()
    _core.train(f, a)
    return _Model(f)
