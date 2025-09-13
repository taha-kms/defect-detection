# src/eval.py

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils import env, metrics, visualization
from src.utils.config import load_config
from src.mvtec_ad.dataset import MVTecDataset
from src.mvtec_ad import transforms as T
from src.models import PaDiMModel, PatchCoreModel, AEModel, FastFlowModel


def load_model(model_name: str, class_name: str, device: str, cfg: dict):
    """
    Instantiate the requested model, then load its checkpoint.
    For PaDiM we also restore the Gaussian stats (means, covs_inv).
    """
    ckpt_path = env.RUNS_DIR / model_name / class_name / f"{model_name}_{class_name}.pt"
    m = model_name.lower()
    bb = cfg.get("backbone", {}).get("name", env.BACKBONE)

    if m == "padim":
        model = PaDiMModel(backbone=bb, device=device)
    elif m == "patchcore":
        model = PatchCoreModel(backbone=bb, device=device)
    elif m == "ae":
        base_ch = cfg.get("models", {}).get("ae", {}).get("base_ch", 32)
        model = AEModel(base_ch=base_ch, device=device)
    elif m == "fastflow":
        ff = cfg.get("models", {}).get("fastflow", {})
        layers = ff.get("layers", ["layer2", "layer3"])
        num_blocks = ff.get("num_blocks", 4)
        hidden = ff.get("hidden", 256)
        model = FastFlowModel(
            backbone=bb, device=device, layers=layers, num_blocks=num_blocks, hidden=hidden
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # IMPORTANT: allow loading numpy extras from our trusted checkpoint (PyTorch 2.6+ change)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Load model weights
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)

    # Restore PaDiM stats if present
    if "means" in ckpt and hasattr(model, "means"):
        means = ckpt["means"]
        if isinstance(means, torch.Tensor):
            means = means.cpu().numpy()
        model.means = means
    if "covs_inv" in ckpt and hasattr(model, "covs_inv"):
        covs_inv = ckpt["covs_inv"]
        if isinstance(covs_inv, torch.Tensor):
            covs_inv = covs_inv.cpu().numpy()
        model.covs_inv = covs_inv

    return model


def _align_maps_to_masks(all_maps: np.ndarray, all_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure anomaly maps and masks have matching spatial resolution for pixel metrics.
    - Squeeze mask channel: [N,1,H,W] -> [N,H,W]
    - Upsample maps to mask resolution using bilinear interpolation.
    """
    # Squeeze mask channel if present
    if all_masks.ndim == 4 and all_masks.shape[1] == 1:
        all_masks = all_masks[:, 0, :, :]

    # If shapes already match, nothing to do
    if all_maps.shape[-2:] == all_masks.shape[-2:]:
        return all_maps, all_masks

    # Resize maps to mask size
    import torch.nn.functional as F
    maps_t = torch.from_numpy(all_maps).unsqueeze(1).float()  # [N,1,Hf,Wf]
    maps_up = F.interpolate(maps_t, size=all_masks.shape[-2:], mode="bilinear", align_corners=False)
    all_maps = maps_up.squeeze(1).cpu().numpy()               # [N,Hm,Wm]
    return all_maps, all_masks


def evaluate(model_name: str, class_name: str, cfg: dict, output_dir: Path):
    # Resolve params (env acts as fallback)
    device = env.DEVICE if torch.cuda.is_available() or env.DEVICE == "cpu" else "cpu"
    bs = cfg.get("train", {}).get("batch_size", 16)
    nw = cfg.get("train", {}).get("num_workers", int(env.NUM_WORKERS))
    im_size = cfg.get("data", {}).get("image_size", env.IMAGE_SIZE)
    center_crop = cfg.get("data", {}).get("center_crop", im_size)

    # Data
    transform = T.get_image_transform(image_size=im_size, center_crop=center_crop)
    mask_transform = T.get_mask_transform(image_size=im_size, center_crop=center_crop)
    test_dataset = MVTecDataset(
        root=env.DATA_DIR, class_name=class_name, split="test",
        transform=transform, mask_transform=mask_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw)

    # Load model + checkpoint
    model = load_model(model_name, class_name, device, cfg)
    model.eval()

    all_scores, all_labels, all_maps, all_masks = [], [], [], []
    print(f"Evaluating {model_name} on '{class_name}'")

    with torch.no_grad():
        for imgs, masks, labels, _ in tqdm(test_loader, desc=f"[Eval:{model_name}] {class_name}", leave=False):
            imgs = imgs.to(device)
            maps, scores = model.predict(imgs)

            # Collect
            all_scores.extend(scores.tolist() if hasattr(scores, "tolist") else list(scores))
            all_labels.extend(labels.numpy().tolist())
            all_maps.append(maps)            # maps: [B,Hf,Wf] (numpy)
            all_masks.append(masks.numpy())  # masks: [B,1,Hm,Wm] (torch->numpy)

    # Concatenate
    all_scores = np.asarray(all_scores)
    all_labels = np.asarray(all_labels)
    all_maps = np.concatenate(all_maps, axis=0)    # [N,Hf,Wf]
    all_masks = np.concatenate(all_masks, axis=0)  # [N,1,Hm,Wm]

    # Align shapes for pixel metrics
    all_maps, all_masks = _align_maps_to_masks(all_maps, all_masks)

    # Image-level metrics
    img_auc = metrics.compute_image_auroc(all_labels, all_scores)

    # Pixel-level metrics (expect [N,H,W] both)
    pix_auc = metrics.compute_pixel_auroc(all_masks, all_maps)
    pr_auc = metrics.compute_auprc(all_masks, all_maps)
    pro = metrics.compute_pro(all_masks, all_maps)

    print(f"Image AUROC: {img_auc:.4f} | Pixel AUROC: {pix_auc:.4f} | AUPRC: {pr_auc:.4f} | PRO: {pro:.4f}")

    # Plots (image-level)
    visualization.plot_roc_curve(all_labels, all_scores, output_dir / "roc_curve.png")
    visualization.plot_pr_curve(all_labels, all_scores, output_dir / "pr_curve.png")

    # Save metrics
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"image_auroc: {img_auc:.6f}\n")
        f.write(f"pixel_auroc: {pix_auc:.6f}\n")
        f.write(f"auprc: {pr_auc:.6f}\n")
        f.write(f"pro: {pro:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models with YAML config")
    parser.add_argument("--model", required=True, choices=["ae", "padim", "patchcore", "fastflow"])
    parser.add_argument("--class_name", required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--extra", type=str, nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, *args.extra)
    out_dir = env.RUNS_DIR / args.model / args.class_name / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate(args.model, args.class_name, cfg, out_dir)


if __name__ == "__main__":
    main()
