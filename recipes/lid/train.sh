#!/bin/bash
# GlotLID-Byte: the language-ID settings of docs/architecture.md, end to end.
#
#   recipes/lid/train.sh CORPUS OUT [lid|lid-accurate] [experts]
#
# CORPUS is a GlotLID-style corpus (VERSION/LABEL/LABEL_SOURCE.txt). Writes
# OUT/model.safetensors: the PRESET model trained with per-label weights
# (fitted bias, confused pairs) and a lexicon built from its training
# split; with "experts", also an expert for each group of expert_groups.tsv.
# BYTEFAST (./bytefast), THREADS (32) and TOTAL (20000000 lines) can be set.
set -euo pipefail
CORPUS=$1
OUT=$2
PRESET=${3:-lid}
EXPERTS=${4:-}
BF=${BYTEFAST:-./bytefast}
T=${THREADS:-32}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$OUT/data
mkdir -p "$D"

# 1. Train/dev/test split and GlotLID's upsampling.
[ -f "$D/train.txt" ] || python3 "$HERE/prepare.py" --corpus "$CORPUS" --out "$D" \
  --sampling upsample --alpha 0.3 --max-repeat 20 --total "${TOTAL:-20000000}" --workers "$T"
# 2. Per-label lexicons from the training split.
[ -f "$OUT/lexicon/lexicon.train.tsv" ] || python3 "$HERE/build_lexicon.py" --corpus "$CORPUS" \
  --out "$OUT/lexicon" --workers "$T"
# 3. A first model, to fit the label weights on dev.
[ -f "$OUT/base.safetensors" ] || "$BF" supervised -preset "$PRESET" -input "$D/train.txt" \
  -output "$OUT/base" -thread "$T" -verbose 1
[ -f "$OUT/weights.tsv" ] || python3 "$HERE/fit_weights.py" --bin "$BF" --model "$OUT/base.safetensors" \
  --dev "$D/dev.txt" --labels "$D/labels.tsv" --threads "$T" --out "$OUT/weights.tsv"
# 4. The model.
[ -f "$OUT/main.safetensors" ] || "$BF" supervised -preset "$PRESET" -input "$D/train.txt" \
  -output "$OUT/main" -labelWeights "$OUT/weights.tsv" -lexicon "$OUT/lexicon/lexicon.train.tsv" \
  -thread "$T" -verbose 1
cp "$OUT/main.safetensors" "$OUT/model.safetensors"

# 5. Optional experts: one model per group of close labels, trained on that
# group's unique training lines (at most 300k per label).
if [ "$EXPERTS" = experts ]; then
  E=$OUT/experts
  mkdir -p "$E"
  grep -v '^#' "$HERE/expert_groups.tsv" | while IFS=$'\t' read -r group members; do
    [ -n "$group" ] || continue
    if [ ! -f "$E/$group.safetensors" ]; then
      pattern=$(echo "$members" | sed 's/[^ ][^ ]*/__label__& /g; s/ $//; s/  */|/g')
      grep -E "^($pattern) " "$D/train.txt" | sort -u \
        | awk 'BEGIN { srand(1) } { print rand() "\t" $0 }' | sort -k1,1 | cut -f2- \
        | awk '{ if (++c[$1] <= 300000) print }' > "$E/$group.txt" || true
      if [ "$(cut -d' ' -f1 "$E/$group.txt" | sort -u | wc -l)" -lt 2 ]; then
        echo "skipping expert $group: fewer than two of its labels in the data"
        continue
      fi
      "$BF" supervised -input "$E/$group.txt" -output "$E/$group" -minCount 3 -minb 2 -maxb 8 \
        -maxBytes 512 -bucket 1048576 -dim 64 -epoch 10 -lr 0.5 -cropProb 0.3 -cropMin 5 \
        -cropMax 20 -normProb 0.3 -thread "$T" -verbose 0
    fi
    "$BF" add-expert "$OUT/model.safetensors" "$E/$group.safetensors" "$OUT/model.safetensors" 0.8
  done
fi
echo "wrote $OUT/model.safetensors"
