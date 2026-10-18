# ByteFast: text classification with byte, character and word features.
# Built on the original fastText sources (MIT license).
# Copyright (c) 2017-present, Facebook, Inc., and 2026 CommonGlot contributors.
from .core import load_model, train_supervised

__version__ = "0.2.0"
__all__ = ["train_supervised", "load_model"]
