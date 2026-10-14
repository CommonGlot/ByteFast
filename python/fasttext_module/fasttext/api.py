"""ByteFast Python API: native byte models and original fastText models.

The model's magic header, rather than its filename or extension, determines
which backend loads it. Legacy fastText is preserved as bytefast.legacy.
Copyright (c) 2026 CommonGlot contributors. MIT License.
"""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from . import _byte_pybind
from . import FastText as legacy

BYTEFAST_MAGIC = 0x42594654
FASTTEXT_MAGIC = 793712314
BUNDLE_MAGIC = b"BFBN"
BUNDLE_VERSION = 1
MAX_BUNDLE_BYTES = 1_000_000_000


class ByteFastModel:
    """Adapter exposing fastText-style predict/get_labels for native ByteFast."""

    def __init__(self, native, model_type="bytefast", backing=None):
        self._native = native
        self.model_type = model_type
        # Retain extracted bundle until this model object is discarded.
        self._backing = backing

    def predict(self, text, k=1, threshold=0.0, gate=0.75):
        """Return (labels, scores) or a list of such pairs for text batches."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if not (0 <= threshold <= 1):
            raise ValueError("threshold must be between 0 and 1")

        def one(value):
            if not isinstance(value, str):
                raise TypeError("predict expects a string or list of strings")
            if self.model_type == "bytefast-calibrated":
                labels, scores = self._native.predict(
                    value, k, float(threshold), float(gate))
            else:
                labels, scores = self._native.predict(
                    value, k, float(threshold))
            return tuple(labels), np.asarray(scores, dtype=np.float32)

        if isinstance(text, str):
            return one(text)
        if isinstance(text, (list, tuple)):
            return [one(value) for value in text]
        raise TypeError("predict expects a string or list of strings")

    def save_model(self, path):
        if self.model_type != "bytefast":
            raise ValueError(
                "A calibrated model consists of base, lexicon and fusion; "
                "use bundle_model() to create a single portable .bin file")
        self._native.save_model(os.fspath(path))

    def get_labels(self):
        return self._native.get_labels()

    def get_dimension(self):
        return self._native.get_dimension()


def _magic(path):
    with open(path, "rb") as stream:
        data = stream.read(4)
    if len(data) != 4:
        raise ValueError("Model file is truncated")
    return struct.unpack("<I", data)[0]


def _calibrated_model(base_path, words_path, fusion_path, backing=None):
    native = _byte_pybind.CalibratedByteFastModel()
    native.load_model(
        os.fspath(base_path), os.fspath(words_path),
        os.fspath(fusion_path))
    return ByteFastModel(native, model_type="bytefast-calibrated",
                         backing=backing)


def load_model(path, *, lexicon=None, fusion=None):
    """Load a ByteFast .bin, bundled ByteFast .bin, or original fastText .bin/.ftz.

    For a ByteFast model with separate lexical sidecars, pass both lexicon=
    and fusion=. A self-contained bundle created with bundle_model() needs
    just one file and automatically loads the calibrated classifier.
    """
    path = Path(path)
    signature = _magic(path)

    if signature == FASTTEXT_MAGIC:
        if lexicon is not None or fusion is not None:
            raise ValueError("Legacy fastText does not support ByteFast sidecars")
        return legacy.load_model(str(path))

    if signature == BYTEFAST_MAGIC:
        if (lexicon is None) != (fusion is None):
            raise ValueError("Pass both lexicon= and fusion=, or neither")
        if lexicon is not None:
            return _calibrated_model(path, lexicon, fusion)
        native = _byte_pybind.ByteFastModel()
        native.load_model(str(path))
        return ByteFastModel(native)

    if signature == int.from_bytes(BUNDLE_MAGIC, "little"):
        if lexicon is not None or fusion is not None:
            raise ValueError("A bundled model already contains all its sidecars")
        # A portable single-file model. Unpack only strictly bounded byte
        # lengths to temporary files; never trust paths supplied by bundles.
        backing = tempfile.TemporaryDirectory(prefix="bytefast-")
        try:
            with open(path, "rb") as src:
                if src.read(4) != BUNDLE_MAGIC:
                    raise ValueError("Invalid bundle header")
                version = struct.unpack("<I", src.read(4))[0]
                if version != BUNDLE_VERSION:
                    raise ValueError("Unsupported ByteFast bundle version")
                sizes_bytes = src.read(24)
                if len(sizes_bytes) != 24:
                    raise ValueError("Truncated ByteFast bundle")
                sizes = struct.unpack("<QQQ", sizes_bytes)
                if not all(sizes) or sum(sizes) > MAX_BUNDLE_BYTES:
                    raise ValueError("Invalid ByteFast bundle sizes")
                if path.stat().st_size != 32 + sum(sizes):
                    raise ValueError("Unexpected ByteFast bundle length")
                outputs = []
                for number, size in enumerate(sizes):
                    destination = Path(backing.name) / ("part%d.bin" % number)
                    with destination.open("wb") as out:
                        remaining = size
                        while remaining:
                            chunk = src.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("Truncated ByteFast bundle")
                            out.write(chunk)
                            remaining -= len(chunk)
                    outputs.append(destination)
            if _magic(outputs[0]) != BYTEFAST_MAGIC:
                raise ValueError("Bundle does not contain a ByteFast base model")
            return _calibrated_model(*outputs, backing=backing)
        except Exception:
            backing.cleanup()
            raise

    raise ValueError(
        "Unrecognized model format: expected ByteFast, ByteFast bundle, "
        "or original fastText .bin/.ftz")


def bundle_model(base, lexicon, fusion, output):
    """Pack byte classifier + lexical table + learned weights in one .bin file.

    These are ByteFast model formats, not files readable by fasttext.load_model.
    """
    paths = [Path(base), Path(lexicon), Path(fusion)]
    if _magic(paths[0]) != BYTEFAST_MAGIC:
        raise ValueError("Expected a ByteFast base model, not legacy fastText")
    sizes = [p.stat().st_size for p in paths]
    if not all(sizes) or sum(sizes) > MAX_BUNDLE_BYTES:
        raise ValueError("Combined model exceeds the 1 GB file budget")
    # Validate all model/lexicon label relationships before bundling.
    _calibrated_model(*paths)
    destination = Path(output)
    with destination.open("wb") as out:
        out.write(BUNDLE_MAGIC)
        out.write(struct.pack("<I", BUNDLE_VERSION))
        out.write(struct.pack("<QQQ", *sizes))
        for part in paths:
            with part.open("rb") as src:
                while True:
                    data = src.read(1024 * 1024)
                    if not data:
                        break
                    out.write(data)
    return destination


def train_supervised(
    input, *, output=None, dim=64, bucket=65536, minn=2, maxn=5,
    epoch=5, lr=0.15, ascii_fold=False, max_bytes=512,
):
    """Train the *ByteFast* byte-level classifier.

    For the original whitespace-tokenized fastText algorithm instead call
    bytefast.legacy.train_supervised(input=...).
    """
    native = _byte_pybind.ByteFastModel()
    native.train(os.fspath(input), dim, bucket, minn, maxn, epoch, lr,
                 bool(ascii_fold), max_bytes)
    model = ByteFastModel(native)
    if output is not None:
        model.save_model(output)
    return model


def train_fasttext_supervised(*args, **kwargs):
    """Train the original fastText algorithm, preserving its word vocabulary."""
    return legacy.train_supervised(*args, **kwargs)


def train_unsupervised(*args, **kwargs):
    """Use the original fastText unsupervised word-embedding implementation."""
    return legacy.train_unsupervised(*args, **kwargs)
