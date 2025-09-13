# 🧪 Defect Detection on MVTec AD

This repository provides a modular framework for **unsupervised anomaly detection and localization** on the [MVTec Anomaly Detection (AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.  
It implements several popular models for visual anomaly detection:

- **Autoencoder (AE)** – convolutional autoencoder with SSIM + MSE loss  
- **PaDiM** – patch distribution modeling with multivariate Gaussians  
- **PatchCore** – memory-bank nearest-neighbor matching on patch embeddings  
- **FastFlow** – normalizing flow–based density estimation on features  

It also includes utilities for **data preparation, training, evaluation, and reporting** with metrics like **Image-level AUROC, Pixel-level AUROC, AUPRC, and PRO**.

---

## 📂 Project Structure

```
defect-detection/
├── configs/                     # YAML configuration files
│   ├── ae.yaml                  # Autoencoder-specific config
│   ├── base.yaml                # Global default config (data, training, backbone)
│   ├── fastflow.yaml            # FastFlow model config
│   ├── padim.yaml               # PaDiM model config
│   ├── patchcore.yaml           # PatchCore model config
│   └── standard6.yaml           # Example config for 6-class benchmark
│
├── scripts/                     # Experiment runner scripts
│   ├── run_experiment.py        # Run multiple models on multiple classes
│   ├── run_standard6.sh         # Run PaDiM + PatchCore on 6 MVTec classes
│   └── run_standard6_all.sh     # Run AE, PaDiM, PatchCore, FastFlow on 6 classes
│
├── src/                         # Main source code
│   ├── __main__.py              # Entrypoint for `python -m src`
│   ├── eval.py                  # Evaluation logic (metrics + visualization)
│   ├── train.py                 # Training pipeline for models
│   │
│   ├── cli/                     # Command-line interface entrypoints
│   │   ├── eval.py              # CLI wrapper for evaluation
│   │   ├── prepare.py           # Validate/prepare dataset
│   │   ├── report.py            # CLI wrapper for report generation
│   │   ├── test.py              # (Stub) CLI for inference on test images
│   │   └── train.py             # (Stub) CLI for training
│   │
│   ├── models/                  # Anomaly detection models
│   │   ├── __init__.py
│   │   ├── ae.py                # Convolutional Autoencoder (AE)
│   │   ├── fastflow.py          # FastFlow (normalizing flow–based)
│   │   ├── padim.py             # PaDiM (patch distribution modeling)
│   │   └── patchcore.py         # PatchCore (memory-bank kNN)
│   │
│   ├── mvtec_ad/                # Dataset utilities
│   │   ├── dataset.py           # MVTec dataset loader
│   │   └── transforms.py        # Image & mask preprocessing transforms
│   │
│   ├── reporting/               # Metrics aggregation & reporting
│   │   ├── __init__.py
│   │   ├── aggregate.py         # Collect results, summarize, plot
│   │   └── report.py            # CLI entry for report generation
│   │
│   └── utils/                   # Helper modules
│       ├── __init__.py
│       ├── config.py            # Config loader & overrides
│       ├── env.py               # Environment variables & paths
│       ├── metrics.py           # AUROC, AUPRC, PRO computation
│       ├── postproc.py          # Thresholding & map normalization
│       └── visualization.py     # Heatmaps, ROC/PR plots, image grids
│
├── .gitignore
├── README.md                    # Project overview (to be expanded)
└── requirements.txt             # Python dependencies

````

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-org>/defect-detection.git
   cd defect-detection
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Create a `.env` file to override defaults:

   ```env
   DATA_DIR=./data        # Path to MVTec dataset root
   RUNS_DIR=./runs        # Where checkpoints & results are stored
   DEVICE=cuda            # or "cpu"
   NUM_WORKERS=4
   SEED=42
   IMAGE_SIZE=256
   BACKBONE=resnet50
   ```

---

## 📊 Dataset Setup

This project uses **MVTec AD dataset** (15 categories of industrial objects with normal and defective samples).

Download the dataset from: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

Extract it under the `data/` directory so the structure looks like:

```
data/
  mvtec/
    bottle/
      train/good/*.png
      test/good/*.png
      test/broken_large/*.png
      ground_truth/broken_large/*.png
      ...
    cable/
    screw/
    ...
```

You can validate the dataset structure with:

```bash
python -m src.cli.prepare --dataset mvtec --list-classes
```

---

## 🚀 Usage

### 1. Prepare Dataset

Validate dataset structure and list available classes:

```bash
python -m src.cli.prepare --dataset mvtec --classes bottle cable screw
```

### 2. Train & Evaluate a Single Model

Example: Train and evaluate **PaDiM** on the `bottle` class:

```bash
python -m src.train --model padim --class_name bottle --config configs/padim.yaml
python -m src.eval --model padim --class_name bottle --config configs/padim.yaml
```

### 3. Run Standard Benchmark (6 classes, multiple models)

```bash
bash scripts/run_standard6_all.sh
```

This will:

* Validate dataset
* Train & evaluate **PaDiM, PatchCore, AE, FastFlow** on `bottle, cable, screw, leather, tile, grid`
* Generate reports under `runs/summary/`

### 4. Generate Reports

After training & evaluation:

```bash
python -m src.reporting.report --models padim patchcore ae fastflow --classes bottle cable screw leather tile grid
```

This produces:

* `runs/summary/summary.csv` – per-class metrics table
* `runs/summary/summary.md` – Markdown summary with best models
* `runs/summary/plots/*.png` – bar charts of average metrics

---

## 📈 Metrics

The evaluation computes:

* **Image-level AUROC** – classification performance (normal vs. anomalous images)
* **Pixel-level AUROC** – segmentation performance for anomaly localization
* **AUPRC** – area under the precision-recall curve (pixel-level)
* **PRO** – Per-Region Overlap metric (region-based detection quality)

---

## 🧩 Models

### 🔹 Autoencoder (AE)

* Simple convolutional autoencoder (`src/models/ae.py`)
* Trained with **MSE + SSIM loss**
* Predicts anomaly maps via reconstruction error

### 🔹 PaDiM

* Implements **PaDiM: Patch Distribution Modeling for Anomaly Detection**
* Uses a pretrained ResNet backbone (`resnet18` or `resnet50`)
* Extracts multi-scale features and fits **multivariate Gaussian distributions** per patch location
* Anomaly score = Mahalanobis distance

### 🔹 PatchCore

* Implements **PatchCore: Anomaly Detection via Memory Bank of Normal Patches**
* Uses pretrained ResNet backbone features
* Builds a **memory bank** of embeddings from normal patches (via kNN search)
* Inference: anomaly score = nearest-neighbor distance

### 🔹 FastFlow

* Implements **FastFlow** (normalizing flow on patch embeddings)
* Uses flows (ActNorm, AffineCoupling, Permute) for density estimation
* Training: Maximum likelihood on normal patches
* Inference: Anomaly score = Negative log-likelihood

---

## ⚡️ Configuration

All experiments are configured using YAML files in [`configs/`](configs/):

* `base.yaml` – default training & data settings
* `ae.yaml`, `padim.yaml`, `patchcore.yaml`, `fastflow.yaml` – model-specific settings
* `standard6.yaml` – predefined setup for the 6 MVTec classes

You can override config values via CLI:

```bash
python -m src.train --model ae --class_name bottle --config configs/base.yaml \
  --extra configs/ae.yaml train.epochs=50 train.lr=0.0005
```

---

## 📜 Example: Full Pipeline

```bash
# 1. Prepare dataset
python -m src.cli.prepare --dataset mvtec --classes bottle cable screw leather tile grid

# 2. Run training + evaluation for multiple models
python scripts/run_experiment.py --models padim patchcore ae fastflow \
  --classes bottle cable screw leather tile grid --config configs/base.yaml

# 3. Aggregate metrics & generate report
python -m src.reporting.report --models padim patchcore ae fastflow \
  --classes bottle cable screw leather tile grid
```

Results will be saved in:

```
runs/
  padim/bottle/
    padim_bottle.pt
    eval/metrics.txt
    eval/roc_curve.png
    eval/pr_curve.png
  ...
  summary/
    summary.csv
    summary.md
    plots/*.png
```

---

## 📌 Roadmap

* [ ] Implement `src/cli/train.py` and `src/cli/eval.py` for direct CLI usage
* [ ] Add `src/cli/test.py` for inference on custom images
* [ ] Extend support for more MVTec AD categories
* [ ] Add hyperparameter optimization and logging integrations

---

## 🛠️ Requirements

* Python **3.9+**
* PyTorch **≥2.2**
* torchvision **≥0.17**
* scikit-learn, scikit-image, matplotlib, tqdm, pyyaml, python-dotenv

Install everything with:

```bash
pip install -r requirements.txt
```


---

## 🙌 Acknowledgements

This repo is inspired by state-of-the-art anomaly detection methods:

* [PaDiM (Defect Detection via Per-Patch Distribution Modeling)](https://arxiv.org/abs/2011.08785)
* [PatchCore](https://arxiv.org/abs/2106.08265)
* [FastFlow](https://arxiv.org/abs/2111.07677)

Dataset: [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---
