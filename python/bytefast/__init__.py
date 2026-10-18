# ByteFast: text classification with byte, character and word features.
# Built on the original fastText sources (MIT license).
# Copyright (c) 2017-present, Facebook, Inc., and 2026 CommonGlot contributors.
from . import core
from .core import (
    train_supervised,
    train_unsupervised,
    tokenize,
    cbow,
    skipgram,
    supervised,
    EOS,
    BOW,
    EOW,
)
from .api import load_model

# Names kept for code written against earlier releases.
legacy = core
train_fasttext_supervised = core.train_supervised

__version__ = "0.2.0"
__all__ = [
    "train_supervised",
    "train_unsupervised",
    "load_model",
    "tokenize",
    "core",
    "cbow",
    "skipgram",
    "supervised",
    "EOS",
    "BOW",
    "EOW",
]
