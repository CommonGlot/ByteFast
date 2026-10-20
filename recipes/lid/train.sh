#!/bin/bash
# GlotLID-Byte: the language-ID settings of docs/architecture.md, end to end.
#
#   recipes/lid/train.sh CORPUS OUT [lid|lid-accurate]
#
# CORPUS is a GlotLID-style corpus (data/LABEL/LABEL_SOURCE.txt). Writes
# OUT/model.safetensors: the PRESET model trained with per-label weights
# (fitted bias, confused pairs) and a lexicon built from its training
# split.
# BYTEFAST (./bytefast), THREADS (32) and TOTAL (20000000 lines) can be set.
set -euo pipefail
CORPUS=$1
OUT=$2
PRESET=${3:-lid}
BF=${BYTEFAST:-./bytefast}
T=${THREADS:-32}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$OUT/data
mkdir -p "$D"

# 1. Train/dev/test split, with the source caps of source_caps.tsv, and
# GlotLID's upsampling.
[ -f "$D/train.txt" ] || python3 "$HERE/prepare.py" --corpus "$CORPUS" --out "$D" \
  --sampling upsample --alpha 0.3 --max-repeat 20 --total "${TOTAL:-20000000}" --workers "$T" \
  --caps "$HERE/source_caps.tsv"
# 2. Per-label lexicons from the training split (same caps, so the split matches).
[ -f "$OUT/lexicon/lexicon.train.tsv" ] || python3 "$HERE/build_lexicon.py" --corpus "$CORPUS" \
  --out "$OUT/lexicon" --workers "$T" --caps "$HERE/source_caps.tsv"
# 3. A first model, to fit the label weights on dev.
[ -f "$OUT/base.safetensors" ] || "$BF" supervised -preset "$PRESET" -input "$D/train.txt" \
  -output "$OUT/base" -thread "$T" -verbose 1
[ -f "$OUT/weights.tsv" ] || python3 "$HERE/fit_weights.py" --bin "$BF" --model "$OUT/base.safetensors" \
  --dev "$D/dev.txt" --labels "$D/labels.tsv" --threads "$T" --out "$OUT/weights.tsv"
# 4. The model.
[ -f "$OUT/model.safetensors" ] || "$BF" supervised -preset "$PRESET" -input "$D/train.txt" \
  -output "$OUT/model" -labelWeights "$OUT/weights.tsv" -lexicon "$OUT/lexicon/lexicon.train.tsv" \
  -thread "$T" -verbose 1
echo "wrote $OUT/model.safetensors"
