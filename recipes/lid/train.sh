#!/bin/bash
# GlotLID-Byte: the language-ID settings of docs/architecture.md, end to end.
#
#   recipes/lid/train.sh CORPUS OUT [lid|lid-accurate]
#
# CORPUS is a GlotLID-style corpus (data/LABEL/LABEL_SOURCE.txt). Writes
# OUT/model.safetensors: the PRESET model with a lexicon built from its
# training split.
# BYTEFAST (./bytefast), THREADS (32), TOTAL (training lines, 20000000; 0 = the
# full upsampled pool) and CAP (at most this many lines per label) can be set.
set -euo pipefail
CORPUS=$1
OUT=$2
PRESET=${3:-lid}
BF=${BYTEFAST:-./bytefast}
T=${THREADS:-32}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$OUT/data
mkdir -p "$D"

# 1. Train/dev/test split with GlotLID's upsampling.
[ -f "$D/train.txt" ] || python3 "$HERE/prepare.py" --corpus "$CORPUS" --out "$D" \
  --sampling upsample --alpha 0.3 --max-repeat 20 --total "${TOTAL:-20000000}" --workers "$T" \
  ${CAP:+--max-train-per-label "$CAP"}
# 2. Per-label lexicons from the training split.
[ -f "$OUT/lexicon/lexicon.train.tsv" ] || python3 "$HERE/build_lexicon.py" --corpus "$CORPUS" \
  --out "$OUT/lexicon" --workers "$T"
# 3. The model, with the lexicon.
[ -f "$OUT/model.safetensors" ] || "$BF" supervised -preset "$PRESET" -input "$D/train.txt" \
  -output "$OUT/model" -lexicon "$OUT/lexicon/lexicon.train.tsv" \
  -thread "$T" -verbose 1
echo "wrote $OUT/model.safetensors"
