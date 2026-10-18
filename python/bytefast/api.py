"""Load any ByteFast model file.

The magic header, not the file name or extension, selects the reader:
ByteFast models and original fastText .bin/.ftz files load with the core
reader.
Copyright (c) 2026 CommonGlot contributors. MIT License.
"""
from __future__ import annotations

from pathlib import Path

from . import core

FASTTEXT_MAGIC = 793712314
BYTEFAST_MAGIC = int.from_bytes(b"BFCM", "little")


def _magic(path):
    with open(path, "rb") as f:
        head = f.read(4)
    return int.from_bytes(head, "little") if len(head) == 4 else None


def load_model(path):
    """Load a ByteFast model or an original fastText model."""
    path = Path(path)
    if _magic(path) not in (FASTTEXT_MAGIC, BYTEFAST_MAGIC):
        raise ValueError("%s is not a ByteFast or fastText model" % path)
    return core.load_model(str(path))
