#!/usr/bin/env bash
set -euo pipefail


DATA_DIR="${DATA_DIR:-./data}"
RUNS_DIR="${RUNS_DIR:-./runs}"
DEVICE="${DEVICE:-cpu}"    
BASE_CFG="${BASE_CFG:-configs/smoke.yaml}"


MODELS=(ae padim patchcore fastflow)
CLASSES=(bottle cable screw)

echo "[smoke] DATA_DIR: $DATA_DIR"
echo "[smoke] RUNS_DIR: $RUNS_DIR"
echo "[smoke] DEVICE  : $DEVICE"
echo "[smoke] CONFIG  : $BASE_CFG"
echo "[smoke] MODELS  : ${MODELS[*]}"
echo "[smoke] CLASSES : ${CLASSES[*]}"
echo


export DATA_DIR RUNS_DIR DEVICE



echo "=== PREPARE ==="
python -m src prepare --data-dir "$DATA_DIR" --verify


echo "=== TRAIN + EVAL (smoke) ==="
for m in "${MODELS[@]}"; do
  for c in "${CLASSES[@]}"; do
    echo "--- TRAIN: $m / $c ---"
    python -m src train --model "$m" --class_name "$c" --config "$BASE_CFG"

    echo "--- EVAL : $m / $c ---"
    python -m src eval  --model "$m" --class_name "$c" --config "$BASE_CFG"
  done
done


echo "=== REPORT ==="
python -m src report --models "${MODELS[@]}" --classes "${CLASSES[@]}" --out "$RUNS_DIR/smoke_summary"

echo

echo "  - CSV:    $RUNS_DIR/smoke_summary/summary.csv"
echo "  - Markdown summary: $RUNS_DIR/smoke_summary/summary.md"
echo "  - Plots:  $RUNS_DIR/smoke_summary/plots/"
