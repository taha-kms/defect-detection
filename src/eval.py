import argparse
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils import env, metrics, visualization
from src.utils.config import load_config
from src.mvtec_ad.dataset import MVTecDataset
from src.mvtec_ad import transforms as T
from src.models import PaDiMModel, PatchCoreModel, AEModel, FastFlowModel


def load_model(model_name: str, class_name: str, device: str, cfg: dict):
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
        model = FastFlowModel(backbone=bb, device=device, layers=layers, num_blocks=num_blocks, hidden=hidden)
    else:
        raise ValueError("Unknown model")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model


def evaluate(model_name: str, class_name: str, cfg: dict, output_dir: Path):
    device = env.DEVICE if torch.cuda.is_available() or env.DEVICE == "cpu" else "cpu"
    bs = cfg.get("train", {}).get("batch_size", 16)
    nw = cfg.get("train", {}).get("num_workers", int(env.NUM_WORKERS))
    im_size = cfg.get("data", {}).get("image_size", env.IMAGE_SIZE)
    center_crop = cfg.get("data", {}).get("center_crop", im_size)

    transform = T.get_image_transform(image_size=im_size, center_crop=center_crop)
    mask_transform = T.get_mask_transform(image_size=im_size, center_crop=center_crop)
    test_dataset = MVTecDataset(root=env.DATA_DIR, class_name=class_name, split="test",
                                transform=transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw)

    model = load_model(model_name, class_name, device, cfg)
    model.eval()

    all_scores, all_labels, all_maps, all_masks = [], [], [], []
    print(f"🔍 Evaluating {model_name} on '{class_name}'")

    for imgs, masks, labels, _ in test_loader:
        imgs = imgs.to(device)
        maps, scores = model.predict(imgs)
        all_scores.extend(scores.tolist() if hasattr(scores, "tolist") else list(scores))
        all_labels.extend(labels.numpy().tolist())
        all_maps.append(maps)
        all_masks.append(masks.numpy())

    import numpy as np
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_maps = np.concatenate(all_maps, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    img_auc = metrics.compute_image_auroc(all_labels, all_scores)
    pix_auc = metrics.compute_pixel_auroc(all_masks, all_maps)
    pr_auc = metrics.compute_auprc(all_masks, all_maps)
    pro = metrics.compute_pro(all_masks, all_maps)

    print(f"📊 Image AUROC: {img_auc:.4f} | Pixel AUROC: {pix_auc:.4f} | AUPRC: {pr_auc:.4f} | PRO: {pro:.4f}")

    visualization.plot_roc_curve(all_labels, all_scores, output_dir / "roc_curve.png")
    visualization.plot_pr_curve(all_labels, all_scores, output_dir / "pr_curve.png")

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