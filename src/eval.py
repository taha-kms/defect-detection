#src/eval.py

import argparse
from pathlib import Path
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils import env, metrics, visualization
from src.utils.config import load_config
from src.mvtec_ad.dataset import MVTecDataset
from src.mvtec_ad import transforms as T
from src.models import PaDiMModel, PatchCoreModel, AEModel, FastFlowModel
from src.utils.postproc import normalize_map, find_best_threshold


def load_model(model_name: str, class_name: str, device: str, cfg: dict,
               ckpt_path: Path | None = None):
    """
    Instantiate the requested model, then load its checkpoint.
    For PaDiM we also restore the Gaussian stats (means, covs_inv).
    """
    # Prefer an explicitly provided checkpoint path (per-run); otherwise fall back to legacy location.
    if ckpt_path is None:
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

    # IMPORTANT: allow loading numpy extras from our trusted checkpoint
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

    # Restore PatchCore memory bank if present
    if m == "patchcore" and "memory_bank_data" in ckpt:
        from sklearn.neighbors import NearestNeighbors
        mb = ckpt["memory_bank_data"]
        if isinstance(mb, torch.Tensor):
            mb = mb.cpu().numpy()
        n_neighbors = ckpt.get("n_neighbors", getattr(model, "n_neighbors", 1))
        model.memory_bank = NearestNeighbors(n_neighbors=n_neighbors)
        model.memory_bank.fit(mb)

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


def _denormalize_imagenet(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Inverse the ImageNet normalization used in transforms to get nice RGB overlays.
    Input: [3,H,W] (torch), Output: HxWx3 float in [0,1]
    """
    IM_MEAN = [0.485, 0.456, 0.406]
    IM_STD = [0.229, 0.224, 0.225]
    x = img_tensor.detach().cpu().float()
    for c in range(3):
        x[c] = x[c] * IM_STD[c] + IM_MEAN[c]
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def evaluate(model_name: str, class_name: str, cfg: dict, output_dir: Path, run_id: Path):
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

    # Load model + checkpoint (from the run root)
    ckpt_path = run_id / f"{model_name}_{class_name}.pt"
    model = load_model(model_name, class_name, device, cfg, ckpt_path=ckpt_path)
    model.eval()

    all_scores, all_labels, all_maps, all_masks = [], [], [], []

    # keep a limited cache of denormalized images for visualization to avoid high RAM usage
    kept_images: list[np.ndarray] = []  # HxWx3 in [0,1]
    kept_limit = 200  # adjust if you want larger galleries

    print(f"Evaluating {model_name} on '{class_name}'")

    batch_times = []
    with torch.no_grad():
        for imgs, masks, labels, _ in tqdm(test_loader, desc=f"[Eval:{model_name}] {class_name}", leave=False):
            start = time.time()
            imgs = imgs.to(device)
            maps, scores = model.predict(imgs)
            elapsed = time.time() - start
            batch_times.append(elapsed / imgs.size(0))  # per-image latency


            # Collect
            all_scores.extend(scores.tolist() if hasattr(scores, "tolist") else list(scores))
            all_labels.extend(labels.numpy().tolist())
            all_maps.append(maps)            # maps: [B,Hf,Wf] (numpy)
            all_masks.append(masks.numpy())  # masks: [B,1,Hm,Wm] (torch->numpy)

            # Keep some RGBs for later visualization
            if len(kept_images) < kept_limit:
                take = min(imgs.size(0), kept_limit - len(kept_images))
                for i in range(take):
                    kept_images.append(_denormalize_imagenet(imgs[i]))


    avg_time_per_image = float(np.mean(batch_times))
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

    print(f"Image AUROC: {img_auc:.4f} | Pixel AUROC: {pix_auc:.4f} | AUPRC: {pr_auc:.4f} | PRO: {pro:.4f} | Latancy: {avg_time_per_image:.6f} sec")

    # Plots (image-level)
    visualization.plot_roc_curve(all_labels, all_scores, output_dir / "roc_curve.png")
    visualization.plot_pr_curve(all_labels, all_scores, output_dir / "pr_curve.png")

    # Save metrics
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"image_auroc: {img_auc:.6f}\n")
        f.write(f"pixel_auroc: {pix_auc:.6f}\n")
        f.write(f"auprc: {pr_auc:.6f}\n")
        f.write(f"pro: {pro:.6f}\n")
        f.write(f"latency_sec: {avg_time_per_image:.6f}\n")

    # ---------- Qualitative overlays ----------
    # Create 4-panel visuals: image / GT / normalized map / overlay
    out_vis = output_dir
    out_vis.mkdir(parents=True, exist_ok=True)

    labels_np = all_labels
    scores_np = all_scores
    masks_np = all_masks  # [N,H,W]
    maps_np = all_maps    # [N,H,W] (aligned)

    # Normalize maps to [0,1] for nicer heatmaps
    maps_norm = np.stack([normalize_map(m) for m in maps_np], axis=0)

    # Determine a reasonable operating threshold (Youden’s J)
    thr = find_best_threshold(labels_np, scores_np)

    # Predictions at image-level
    preds = (scores_np >= thr).astype(int)

    # Indices for TP/FP/FN and a sample of TN
    idx_tp = np.where((preds == 1) & (labels_np == 1))[0]
    idx_fp = np.where((preds == 1) & (labels_np == 0))[0]
    idx_fn = np.where((preds == 0) & (labels_np == 1))[0]
    idx_tn = np.where((preds == 0) & (labels_np == 0))[0]

    # Because we only cached the first `kept_limit` denormalized images,
    # restrict galleries to indices within that range
    max_vis_idx = len(kept_images)
    idx_tp = idx_tp[idx_tp < max_vis_idx]
    idx_fp = idx_fp[idx_fp < max_vis_idx]
    idx_fn = idx_fn[idx_fn < max_vis_idx]
    idx_tn = idx_tn[idx_tn < max_vis_idx]


    visualization.save_gallery(idx_tp, "TP_top", out_vis, scores_np, kept_images, masks_np, maps_norm)
    visualization.save_gallery(idx_fp, "FP_top", out_vis, scores_np, kept_images, masks_np, maps_norm)
    visualization.save_gallery(idx_fn, "FN_top", out_vis, scores_np, kept_images, masks_np, maps_norm)

    if idx_tn.size > 0:
        rng = np.random.default_rng(0)
        tn_sample = rng.choice(idx_tn, size=min(12, idx_tn.size), replace=False)
        visualization.save_gallery(tn_sample, "TN_sample", out_vis, scores_np, kept_images, masks_np, maps_norm)


    # Also save a single “teaser” grid mixing a few of each type
    mix_parts = []
    for arr, k in ((idx_tp, 3), (idx_fp, 3), (idx_fn, 3), (idx_tn, 3)):
        if arr.size > 0:
            mix_parts.append(arr[:k])
    if mix_parts:
        mix = np.concatenate(mix_parts)[:12]
        images, titles = [], []
        for idx in mix:
            panels, t = visualization.make_panels(idx, kept_images, masks_np, maps_norm)
            images.extend(panels)
            titles.extend(t)
        if images:
            visualization.save_image_grid(images, titles, out_vis / "teaser.png", cols=4, figsize=(12, 12))

# (idx, kept_images, masks_np, maps_norm, out_vis)
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"image_auroc: {img_auc:.6f}\n")
        f.write(f"pixel_auroc: {pix_auc:.6f}\n")
        f.write(f"auprc: {pr_auc:.6f}\n")
        f.write(f"pro: {pro:.6f}\n")
        f.write(f"threshold: {thr:.6f}\n")
        f.write(f"tp: {len(idx_tp)}\n")
        f.write(f"fp: {len(idx_fp)}\n")
        f.write(f"fn: {len(idx_fn)}\n")
        f.write(f"tn: {len(idx_tn)}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models with YAML config")
    parser.add_argument("--model", required=True, choices=["ae", "padim", "patchcore", "fastflow"])
    parser.add_argument("--class_name", required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--extra", type=str, nargs="*", default=[])
    parser.add_argument("--run_id", type=str, default=None, help="Optional run ID to evaluate a specific run")
    args = parser.parse_args()

    cfg = load_config(args.config, *args.extra)

    base = env.RUNS_DIR / args.model / args.class_name
    run_id = (base / "runs" / args.run_id) if args.run_id else (base / "latest")
    out_dir = run_id / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluate(args.model, args.class_name, cfg, out_dir, run_id=run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
