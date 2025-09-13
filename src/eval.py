import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils import env, metrics, visualization
from src.mvtec_ad.dataset import MVTecDataset
from src.mvtec_ad import transforms as T
from src.models import PaDiMModel, PatchCoreModel


def load_model(model_name: str, class_name: str, device: str):
    """Load trained model checkpoint."""
    ckpt_path = env.RUNS_DIR / model_name / class_name / f"{model_name}_{class_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_name == "padim":
        model = PaDiMModel(backbone=env.BACKBONE, device=device)
    else:
        model = PatchCoreModel(backbone=env.BACKBONE, device=device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)

    # Restore Gaussian stats if PaDiM
    if isinstance(model, PaDiMModel):
        model.means = ckpt["means"]
        model.covs_inv = ckpt["covs_inv"]

    return model


def evaluate(model_name: str, class_name: str, batch_size: int, num_workers: int, device: str, output_dir: Path):
    # Dataset + Loader
    transform = T.get_image_transform(image_size=env.IMAGE_SIZE)
    mask_transform = T.get_mask_transform(image_size=env.IMAGE_SIZE)

    test_dataset = MVTecDataset(
        root=env.DATA_DIR,
        class_name=class_name,
        split="test",
        transform=transform,
        mask_transform=mask_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load trained model
    model = load_model(model_name, class_name, device)
    model.eval()

    all_scores, all_labels, all_maps, all_masks = [], [], [], []

    print(f"Evaluating {model_name} on class '{class_name}' with {len(test_dataset)} test images")

    for imgs, masks, labels, paths in test_loader:
        imgs = imgs.to(device)
        maps, scores = model.predict(imgs)

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_maps.append(maps)
        all_masks.append(masks.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_maps = np.concatenate(all_maps, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Compute metrics
    img_auc = metrics.compute_image_auroc(all_labels, all_scores)
    pix_auc = metrics.compute_pixel_auroc(all_masks, all_maps)
    pr_auc = metrics.compute_auprc(all_masks, all_maps)
    pro = metrics.compute_pro(all_masks, all_maps)

    print(f"Image-level AUROC: {img_auc:.4f}")
    print(f"Pixel-level AUROC: {pix_auc:.4f}")
    print(f"Pixel-level AUPRC: {pr_auc:.4f}")
    print(f"PRO: {pro:.4f}")

    # Save ROC/PR plots
    visualization.plot_roc_curve(all_labels, all_scores, output_dir / "roc_curve.png")
    visualization.plot_pr_curve(all_labels, all_scores, output_dir / "pr_curve.png")

    return {"image_auroc": img_auc, "pixel_auroc": pix_auc, "auprc": pr_auc, "pro": pro}


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained anomaly detection models")
    parser.add_argument("--model", choices=["padim", "patchcore"], required=True)
    parser.add_argument("--class_name", required=True, help="MVTec class name")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=int(env.NUM_WORKERS))
    args = parser.parse_args()

    output_dir = env.RUNS_DIR / args.model / args.class_name / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = env.DEVICE if torch.cuda.is_available() or env.DEVICE == "cpu" else "cpu"

    results = evaluate(args.model, args.class_name, args.batch_size, args.num_workers, device, output_dir)

    # Save metrics
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()