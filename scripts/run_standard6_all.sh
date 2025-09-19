#!/usr/bin/env bash
set -euo pipefail


DATA_DIR="${DATA_DIR:-./data}"
RUNS_DIR="${RUNS_DIR:-./runs}"
DEVICE="${DEVICE:-cuda}"                     # default to GPU on server; override with DEVICE=cpu if needed
BASE_CFG="${BASE_CFG:-configs/standard6.yaml}"  # use your robust config from earlier

MODELS=(ae padim patchcore fastflow)
CLASSES=(bottle cable screw toothbrush transistor zipper)

echo "[robust] DATA_DIR: $DATA_DIR"
echo "[robust] RUNS_DIR: $RUNS_DIR"
echo "[robust] DEVICE  : $DEVICE"
echo "[robust] CONFIG  : $BASE_CFG"
echo "[robust] MODELS  : ${MODELS[*]}"
echo "[robust] CLASSES : ${CLASSES[*]}"
echo

export DATA_DIR RUNS_DIR DEVICE

echo "=== PREPARE ==="
python -m src prepare --data-dir "$DATA_DIR" --verify

echo "=== TRAIN + EVAL (robust) ==="
for m in "${MODELS[@]}"; do
  for c in "${CLASSES[@]}"; do
    echo "--- TRAIN: $m / $c ---"
    python -m src train --model "$m" --class_name "$c" --config "$BASE_CFG"

    echo "--- EVAL : $m / $c ---"
    python -m src eval  --model "$m" --class_name "$c" --config "$BASE_CFG"
  done
done

echo "=== REPORT ==="
python -m src report --models "${MODELS[@]}" --classes "${CLASSES[@]}" --out "$RUNS_DIR/robust_summary"

echo
echo "  - CSV:    $RUNS_DIR/robust_summary/summary.csv"
echo "  - Markdown summary: $RUNS_DIR/robust_summary/summary.md"
echo "  - Plots:  $RUNS_DIR/robust_summary/plots/"
