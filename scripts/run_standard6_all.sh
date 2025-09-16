#!/usr/bin/env bash
set -e

# Load .env if present (non-fatal)
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs -d '\n' -r)
fi

# Defaults if not provided in .env
DATA_DIR="${DATA_DIR:-./data}"

MODELS=(padim patchcore ae fastflow)
CLASSES=(bottle cable screw leather tile grid)

echo "=== PREPARE ==="
python -m src prepare --data-dir "$DATA_DIR" --verify

echo "=== TRAIN + EVAL (Standard 6) ==="
for m in "${MODELS[@]}"; do
  for c in "${CLASSES[@]}"; do
    echo "--- TRAIN: $m / $c ---"
    python -m src train --model "$m" --class_name "$c" --config configs/standard6.yaml
    echo "--- EVAL:  $m / $c ---"
    python -m src eval  --model "$m" --class_name "$c" --config configs/standard6.yaml
  done
done

echo "=== REPORT ==="
# <-- this is the line you asked for
python -m src report --models "${MODELS[@]}" --classes "${CLASSES[@]}" --out runs/summary

echo "DONE. See runs/summary for CSV/MD/plots."
