# Defect Detection on MVTec AD

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular framework for **unsupervised anomaly detection and localization** on the [MVTec Anomaly Detection (AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

Implemented models:

* **Autoencoder (AE)** – convolutional autoencoder with SSIM + MSE loss
* **PaDiM** – patch distribution modeling with multivariate Gaussians
* **PatchCore** – memory-bank nearest-neighbor matching on patch embeddings
* **FastFlow** – normalizing-flow–based density estimation on features

Utilities are included for data preparation, training, evaluation, and reporting, with metrics for **Image-level AUROC, Pixel-level AUROC, AUPRC, and PRO**.

## Project structure

```
.
├── configs/            # YAML configs (base, smoke, standard6, per-model)
├── scripts/            # Bash entry points (smoke.sh, run_standard6_all.sh)
├── src/
│   ├── __main__.py     # CLI dispatcher: train | eval | report | prepare
│   ├── prepare.py      # Dataset preparation / verification
│   ├── train.py        # Training loop
│   ├── eval.py         # Evaluation + metrics
│   ├── models/         # ae, padim, patchcore, fastflow
│   ├── mvtec_ad/       # Dataset + transforms
│   ├── reporting/      # Aggregation + summary report
│   └── utils/          # config, env, metrics, postproc, visualization
├── summary/            # Example aggregated results (CSV + Markdown + plots)
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Installation

### Local (Python venv)

```bash
git clone https://github.com/taha-kms/defect-detection.git
cd defect-detection

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env   # then edit DEVICE, paths, etc.
```

### Docker

```bash
docker build -t defect-detection .
docker run --rm -it --gpus all \
  -v "$PWD/data:/app/data" \
  -v "$PWD/runs:/app/runs" \
  defect-detection
```

## Dataset setup

Download the MVTec AD dataset from the [official site](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it under `data/` so each class lives at the top level:

```
data/
├── bottle/
├── cable/
├── screw/
├── toothbrush/
├── transistor/
└── zipper/
```

Verify:

```bash
python -m src prepare --data-dir ./data --verify
```

## Usage

The CLI is dispatched through `python -m src <command>`:

```bash
python -m src prepare --help
python -m src train   --help
python -m src eval    --help
python -m src report  --help
```

### Train + evaluate a single model/class

```bash
python -m src train --model patchcore --class_name bottle --config configs/standard6.yaml
python -m src eval  --model patchcore --class_name bottle --config configs/standard6.yaml
```

### Smoke test (CPU-friendly, small subset)

```bash
bash scripts/smoke.sh
```

### Full benchmark (4 models × 6 classes)

```bash
bash scripts/run_standard6_all.sh
```

Outputs land in `runs/<model>/<class>/<run_id>/` and an aggregated summary is written to `runs/robust_summary/` (CSV, Markdown, plots).

## Results

I ran the full benchmark (`scripts/run_standard6_all.sh`) across **4 models × 6 MVTec AD classes** — `bottle`, `cable`, `screw`, `toothbrush`, `transistor`, `zipper` — at 256×256 resolution with a pretrained ResNet-50 backbone for the feature-based methods. All numbers below come from `summary/summary.md` and the run artifacts under `runs/`.

### Per-model averages (across the 6 classes)

| Model     | Image AUROC | Pixel AUROC | AUPRC  | PRO    | Latency (s/img) |
|-----------|:-----------:|:-----------:|:------:|:------:|:---------------:|
| PatchCore | 0.746       | **0.913**   | **0.276** | **0.161** | 2.79            |
| PaDiM     | **0.754**   | 0.773       | 0.101  | 0.082  | 0.011           |
| FastFlow  | 0.579       | 0.666       | 0.070  | 0.025  | 0.014           |
| AE        | 0.507       | 0.447       | 0.039  | 0.058  | **0.005**       |

### Best per-class (image-level AUROC)

| Class      | Best model | Image AUROC | Pixel AUROC | AUPRC  | PRO    |
|------------|------------|:-----------:|:-----------:|:------:|:------:|
| bottle     | PatchCore  | 0.947       | 0.921       | 0.414  | 0.093  |
| cable      | PatchCore  | 0.683       | 0.889       | 0.179  | 0.172  |
| screw      | PaDiM      | 0.950       | 0.895       | 0.013  | 0.152  |
| toothbrush | PatchCore  | **0.989**   | 0.955       | 0.322  | 0.279  |
| transistor | PatchCore  | 0.848       | 0.964       | 0.593  | 0.138  |
| zipper     | PatchCore  | 0.848       | 0.887       | 0.138  | 0.164  |

### Takeaways

* **PatchCore wins on 5 of 6 classes** for image-level detection and dominates pixel-level localization (mean Pixel AUROC **0.913**), at the cost of being ~250× slower than PaDiM per image.
* **PaDiM** is the strongest accuracy-per-millisecond tradeoff and the best model on `screw` (where PatchCore unexpectedly underperformed at 0.16 image AUROC — likely a thresholding issue worth investigating).
* **FastFlow** trailed PatchCore/PaDiM on detection but produced respectable pixel-level scores on `screw` (0.941) and `toothbrush` (0.852).
* **AE** is fastest but the weakest detector overall, as expected for a reconstruction baseline without a pretrained backbone.

Best overall result: **PatchCore on `toothbrush`** with image-level AUROC of **0.989** (28/30 defective images detected, 0 false positives).

Full per-class / per-model tables and confusion-matrix counts are in [`summary/summary.md`](summary/summary.md); plots are in [`summary/plots/`](summary/plots/).

## Configuration

Configs are layered YAML files in `configs/`:

* `base.yaml` – global defaults (image size, epochs, backbone, per-model hyperparams)
* `smoke.yaml` – fast sanity-check settings
* `standard6.yaml` – full-run settings used for the results above
* `ae.yaml`, `padim.yaml`, `patchcore.yaml`, `fastflow.yaml` – per-model overrides

Override via `--config` on any subcommand.

## License

Released under the [MIT License](LICENSE).
