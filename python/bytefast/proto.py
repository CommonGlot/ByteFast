"""Prototype byte-only classifier with an optional lexical sidecar.

This is the experimental model from src/proto: a byte n-gram classifier, a
statistical word-to-class table and learned fusion weights, stored in its own
file formats. The main ByteFast model (bytefast.train_supervised) combines
byte n-grams and words in one network; this module is kept for comparison.
Copyright (c) 2026 CommonGlot contributors. MIT License.
"""
from __future__ import annotations

import hashlib
import os
import random
import struct
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import _proto_pybind

BYTEFAST_MAGIC = 0x42594654
BUNDLE_MAGIC = b"BFBN"
BUNDLE_VERSION = 1
MAX_BUNDLE_BYTES = 1_000_000_000


class ProtoModel:
    """predict/get_labels adapter for prototype byte and lexical models."""

    def __init__(self, native, model_type="bytefast", backing=None,
                 parts=None):
        self._native = native
        self.model_type = model_type
        # Retain extracted bundle until this model object is discarded.
        self._backing = backing
        self._parts = parts

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
        if self.model_type == "bytefast-calibrated":
            if self._parts is None:
                raise ValueError("No lexical components available to save")
            bundle_model(*self._parts, path)
        else:
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
    native = _proto_pybind.CalibratedByteFastModel()
    native.load_model(
        os.fspath(base_path), os.fspath(words_path),
        os.fspath(fusion_path))
    return ProtoModel(native, model_type="bytefast-calibrated",
                         backing=backing,
                         parts=(base_path, words_path, fusion_path))


def load_model(path, *, lexicon=None, fusion=None):
    """Load a prototype byte model or a bundled prototype lexical model.

    For a byte model with separate lexical sidecars, pass both lexicon= and
    fusion=. A self-contained bundle created with bundle_model() needs just
    one file and loads the calibrated classifier.
    """
    path = Path(path)
    signature = _magic(path)

    if signature == BYTEFAST_MAGIC:
        if (lexicon is None) != (fusion is None):
            raise ValueError("Pass both lexicon= and fusion=, or neither")
        if lexicon is not None:
            return _calibrated_model(path, lexicon, fusion)
        native = _proto_pybind.ByteFastModel()
        native.load_model(str(path))
        return ProtoModel(native)

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
        "Unrecognized model format: expected a prototype byte model or "
        "prototype bundle")


def bundle_model(base, lexicon, fusion, output):
    """Pack byte classifier + lexical table + learned weights in one .bin file.

    These are ByteFast model formats, not files readable by fasttext.load_model.
    """
    paths = [Path(base), Path(lexicon), Path(fusion)]
    if _magic(paths[0]) != BYTEFAST_MAGIC:
        raise ValueError("Expected a prototype byte model as the base")
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


def _split_lexical_training(source, directory, seed, split=(0.8, 0.1, 0.1)):
    """Make class-stratified, mutually disjoint fit/calibration/validation
    partitions (80/10/10 by default).

    Base training and word statistics use the 80% fit partition; the
    other partitions are reserved for supervised fusion and early stopping.
    """
    lines = []
    groups = defaultdict(list)
    with open(source, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split(maxsplit=1)
            if (len(fields) != 2 or
                    not fields[0].startswith("__label__") or
                    len(fields[0]) <= len("__label__")):
                raise ValueError("Expected '__label__CLASS text' on every line")
            groups[fields[0]].append(len(lines))
            lines.append(line)

    fit_indices = set()
    calibration_indices = set()
    validation_indices = set()
    for label, indices in groups.items():
        if len(indices) < 10:
            raise ValueError(
                "lexical=True requires at least 10 documents per class "
                "for the automatic lexical_split; class %s has %d"
                % (label, len(indices)))
        # Avoid Python's process-dependent string hash.
        label_seed = int.from_bytes(
            hashlib.blake2b(label.encode("utf-8"),
                            digest_size=8).digest(), "little")
        shuffled = list(indices)
        random.Random(int(seed) ^ label_seed).shuffle(shuffled)
        n_cal = max(1, round(len(indices) * split[1]))
        n_val = max(1, round(len(indices) * split[2]))
        calibration_indices.update(shuffled[:n_cal])
        validation_indices.update(shuffled[n_cal:n_cal + n_val])
        fit_indices.update(shuffled[n_cal + n_val:])

    partitions = (
        ("fit", fit_indices),
        ("calibration", calibration_indices),
        ("validation", validation_indices),
    )
    results = []
    for name, indices in partitions:
        path = Path(directory) / ("%s.fasttext" % name)
        with path.open("w", encoding="utf-8") as output_file:
            for index, line in enumerate(lines):
                if index in indices:
                    output_file.write(line + "\n")
        results.append(path)
    return results


def train_supervised(
    input, *, output=None, dim=64, bucket=65536, minn=2, maxn=5,
    epoch=5, lr=0.15, ascii_fold=False, max_bytes=512,
    hash_count=1, position_bins=0,
    lexical=False, lexical_vocab=200000, lexical_min_df=1,
    lexical_selection="global", lexical_seed=1337,
    lexical_calibration_data=None, lexical_validation_data=None,
    lexical_split=(0.8, 0.1, 0.1), lexical_fusion_epochs=90,
    lexical_fusion_lr=0.07, lexical_fusion_l2=0.01,
    lexical_fusion_patience=16,
):
    """Train the prototype byte classifier (optionally with a lexical sidecar).

    By default, train only the byte-level classifier (backward compatible).

    With lexical=True, select word-to-class statistics from training data,
    fit lexical fusion weights from separate calibration examples, and
    return a combined lexical model. If no calibration/validation files
    are supplied, make reproducible 80/10/10 splits *per class* from input;
    all three sets are disjoint. Output is one ByteFast bundle .bin.

    lexical_selection="global" is the current supported word-ranking method.
    Class-balanced and confusion-aware selection are not implemented yet.

    Every training setting is an argument: the byte encoder (dim, bucket,
    minn, maxn, epoch, lr, ascii_fold, max_bytes, hash_count,
    position_bins), the word table (lexical_vocab, lexical_min_df), the
    automatic split (lexical_split, lexical_seed) and the fusion optimizer
    (lexical_fusion_epochs, lexical_fusion_lr, lexical_fusion_l2,
    lexical_fusion_patience).

    The main ByteFast model is bytefast.train_supervised(...).
    """
    if not lexical:
        if (lexical_calibration_data is not None or
                lexical_validation_data is not None):
            raise ValueError("lexical=True is required for calibration data")
        native = _proto_pybind.ByteFastModel()
        native.train(os.fspath(input), dim, bucket, minn, maxn, epoch, lr,
                     bool(ascii_fold), max_bytes, hash_count, position_bins)
        model = ProtoModel(native)
        if output is not None:
            model.save_model(output)
        return model

    if lexical_selection != "global":
        raise ValueError(
            "Only lexical_selection='global' is implemented; "
            "class-balanced and confusion-aware selection require "
            "additional training support")
    if (isinstance(lexical_vocab, bool) or
            not isinstance(lexical_vocab, int) or
            not 1 <= lexical_vocab <= 1000000):
        raise ValueError("lexical_vocab must be an integer in [1, 1000000]")
    if (isinstance(lexical_min_df, bool) or
            not isinstance(lexical_min_df, int) or
            lexical_min_df < 1):
        raise ValueError("lexical_min_df must be a positive integer")
    if (lexical_calibration_data is None) != (lexical_validation_data is None):
        raise ValueError(
            "Pass both lexical_calibration_data and "
            "lexical_validation_data, or neither")
    if (lexical_calibration_data is not None and
            len({os.path.abspath(os.fspath(path)) for path in
                 (input, lexical_calibration_data,
                  lexical_validation_data)}) != 3):
        raise ValueError(
            "Training, calibration and validation paths must be distinct")

    split = tuple(float(x) for x in lexical_split)
    if (len(split) != 3 or min(split) <= 0 or
            abs(sum(split) - 1.0) > 1e-6):
        raise ValueError(
            "lexical_split must be three positive fractions summing to 1")

    workdir = tempfile.TemporaryDirectory(prefix="bytefast-train-")
    try:
        if lexical_calibration_data is None:
            fit_path, calibration_path, validation_path = (
                _split_lexical_training(input, workdir.name, lexical_seed,
                                        split))
        else:
            fit_path = Path(input)
            calibration_path = Path(lexical_calibration_data)
            validation_path = Path(lexical_validation_data)

        base = Path(workdir.name) / "base.bin"
        lexicon = Path(workdir.name) / "words.lex"
        fusion = Path(workdir.name) / "weights.fuse"
        native = _proto_pybind.ByteFastModel()
        native.train(os.fspath(fit_path), dim, bucket, minn, maxn,
                     epoch, lr, bool(ascii_fold), max_bytes,
                     hash_count, position_bins)
        native.save_model(str(base))
        _proto_pybind.fit_lexical_sidecars(
            str(base), os.fspath(fit_path),
            os.fspath(calibration_path), os.fspath(validation_path),
            str(lexicon), str(fusion), lexical_vocab, lexical_min_df,
            lexical_fusion_epochs, lexical_fusion_lr, lexical_fusion_l2,
            lexical_fusion_patience)

        if output is None:
            return _calibrated_model(
                base, lexicon, fusion, backing=workdir)

        # All lexical parameters, including classwise fusion coefficients,
        # are stored in the same model file.
        bundle_model(base, lexicon, fusion, output)
        model = load_model(output)
        return model
    finally:
        # The no-output model retains workdir; the loader does not need
        # the intermediate training files once weights have been loaded.
        if output is not None:
            workdir.cleanup()


# Kept for code written against earlier releases.
ByteFastModel = ProtoModel
