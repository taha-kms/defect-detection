#!/usr/bin/env bash
set -euo pipefail

# Load .env if present (non-fatal)
if [ -f ".env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -d '\n' -r)
fi

# Defaults if not provided in .env
DATA_DIR="${DATA_DIR:-./data}"
BASE_CFG="configs/base.yaml"
STD_CFG="configs/standard6.yaml"   # used only to read lists

# --- Read MODELS and CLASSES from YAML (yq -> Python -> grep/sed fallback) ---
if command -v yq >/dev/null 2>&1; then
  readarray -t MODELS < <(yq '.models[]'  "$STD_CFG")
  readarray -t CLASSES < <(yq '.classes[]' "$STD_CFG")
else
  # Use Python + PyYAML (repo already uses PyYAML)
  readarray -t MODELS < <(python - <<'PY' "$STD_CFG"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for x in cfg.get('models', []): print(x)
PY
)
  readarray -t CLASSES < <(python - <<'PY' "$STD_CFG"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for x in cfg.get('classes', []): print(x)
PY
)
fi

echo "Using models:  ${MODELS[*]}"
echo "Using classes: ${CLASSES[*]}"

echo "=== PREPARE ==="
python -m src prepare --data-dir "$DATA_DIR" --verify  # unified CLI entrypoint :contentReference[oaicite:3]{index=3}

echo "=== TRAIN + EVAL (Standard 6) ==="
for m in "${MODELS[@]}"; do
  EXTRA_CFG="configs/${m}.yaml"   # ae.yaml / padim.yaml / patchcore.yaml / fastflow.yaml
  if [[ ! -f "$EXTRA_CFG" ]]; then
    echo "[warn] Missing per-model config: $EXTRA_CFG — continuing with base only."
    EXTRA_ARGS=()
  else
    EXTRA_ARGS=(--extra "$EXTRA_CFG")
  fi

  for c in "${CLASSES[@]}"; do
    echo "--- TRAIN: $m / $c ---"
    python -m src train --model "$m" --class_name "$c" --config "$BASE_CFG" "${EXTRA_ARGS[@]}"  # train.py expects cfg.models dict :contentReference[oaicite:4]{index=4}

    echo "--- EVAL:  $m / $c ---"
    python -m src eval  --model "$m" --class_name "$c" --config "$BASE_CFG" "${EXTRA_ARGS[@]}"  # eval.py uses same config shape :contentReference[oaicite:5]{index=5}
  done
done

echo "=== REPORT ==="
python -m src report --models "${MODELS[@]}" --classes "${CLASSES[@]}" --out runs/summary  # generates CSV/MD/plots :contentReference[oaicite:6]{index=6}

echo "DONE. See runs/summary for CSV/MD/plots."
