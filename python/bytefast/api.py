"""Load any ByteFast model file.

The magic header, not the file name or extension, selects the reader:
ByteFast models and original fastText .bin/.ftz files load with the core
reader, and prototype byte models and bundles with bytefast.proto.
Copyright (c) 2026 CommonGlot contributors. MIT License.
"""
from __future__ import annotations

from pathlib import Path

from . import core
from . import proto

FASTTEXT_MAGIC = 793712314
BYTEFAST_MAGIC = int.from_bytes(b"BFCM", "little")


def load_model(path, *, lexicon=None, fusion=None):
    """Load a ByteFast model, an original fastText model or a prototype model.

    lexicon= and fusion= only apply to prototype byte models; see
    bytefast.proto.load_model.
    """
    path = Path(path)
    if proto._magic(path) in (FASTTEXT_MAGIC, BYTEFAST_MAGIC):
        if lexicon is not None or fusion is not None:
            raise ValueError("lexicon= and fusion= only apply to prototype models")
        return core.load_model(str(path))
    return proto.load_model(path, lexicon=lexicon, fusion=fusion)
