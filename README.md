
# Defect Detection on MVTec AD

This repository provides a modular framework for **unsupervised anomaly detection and localization** on the [MVTec Anomaly Detection (AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

It implements several popular models for visual anomaly detection:

* **Autoencoder (AE)** – convolutional autoencoder with SSIM + MSE loss
* **PaDiM** – patch distribution modeling with multivariate Gaussians
* **PatchCore** – memory-bank nearest-neighbor matching on patch embeddings
* **FastFlow** – normalizing-flow–based density estimation on features

Utilities are included for **data preparation, training, evaluation, and reporting**, with metrics like **Image-level AUROC, Pixel-level AUROC, AUPRC, and PRO**.
