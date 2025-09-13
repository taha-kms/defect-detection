#!/usr/bin/env bash
set -e

# Load .env if present (non-fatal)
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs -d '\n' -r)
fi

echo "=== PREPARE ==="
python -m src prepare --dataset mvtec --classes bottle cable screw leather tile grid

echo "=== TRAIN + EVAL (Standard 6) ==="
python scripts/run_experiment.py --models padim patchcore ae fastflow --classes bottle cable screw leather tile grid

echo "=== REPORT ==="
python -m src report --models padim patchcore ae fastflow --classes bottle cable screw leather tile grid

echo "DONE. See runs/summary for CSV/MD/plots."