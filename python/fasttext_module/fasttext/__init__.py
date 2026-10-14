# ByteFast Python package; original fastText algorithms are preserved under
# bytefast.legacy, using the original Facebook Research sources and MIT license.
# Copyright (c) 2017-present, Facebook, Inc., and 2026 CommonGlot contributors.
from . import FastText as legacy
from .FastText import cbow, skipgram, supervised, tokenize, EOS, BOW, EOW
from .api import (
    ByteFastModel,
    train_supervised,
    train_fasttext_supervised,
    train_unsupervised,
    load_model,
    bundle_model,
)

__version__ = "0.1.0"
__all__ = [
    "ByteFastModel",
    "train_supervised",
    "train_fasttext_supervised",
    "train_unsupervised",
    "load_model",
    "bundle_model",
    "legacy",
    "tokenize",
    "cbow",
    "skipgram",
    "supervised",
    "EOS",
    "BOW",
    "EOW",
]
