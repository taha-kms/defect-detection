#!/usr/bin/env bash
# One-liner to run Standard(6) across PaDiM and PatchCore

set -e

MODELS=("padim" "patchcore")
CLASSES=("bottle" "cable" "screw" "leather" "tile" "grid")

for model in "${MODELS[@]}"; do
  for cls in "${CLASSES[@]}"; do
    echo "==== Training $model on $cls ===="
    python -m src.train --model "$model" --class_name "$cls" --batch_size 16 --num_workers 4

    echo "==== Evaluating $model on $cls ===="
    python -m src.eval  --model "$model" --class_name "$cls" --batch_size 16 --num_workers 4
  done
done

echo "Standard(6) experiment finished."