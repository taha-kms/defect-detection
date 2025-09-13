import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils import env
from src.mvtec_ad.dataset import MVTecDataset
from src.mvtec_ad import transforms as T
from src.models import PaDiMModel, PatchCoreModel


def get_model(model_name: str, device: str):
    if model_name.lower() == "padim":
        return PaDiMModel(backbone=env.BACKBONE, device=device)
    elif model_name.lower() == "patchcore":
        return PatchCoreModel(backbone=env.BACKBONE, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train(model_name: str, class_name: str, batch_size: int, num_workers: int, device: str, output_dir: Path):
    # Dataset + Loader
    transform = T.get_image_transform(image_size=env.IMAGE_SIZE)
    mask_transform = T.get_mask_transform(image_size=env.IMAGE_SIZE)

    train_dataset = MVTecDataset(
        root=env.DATA_DIR,
        class_name=class_name,
        split="train",
        transform=transform,
        mask_transform=mask_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Model
    model = get_model(model_name, device)
    print(f"🚀 Training {model_name} on class '{class_name}' with {len(train_dataset)} samples")

    # Fit model (no optimizer — unsupervised methods)
    model.fit(train_loader)

    # Save model
    save_path = output_dir / f"{model_name}_{class_name}.pt"
    torch.save({"model_state": model.state_dict(),
                "means": getattr(model, "means", None),
                "covs_inv": getattr(model, "covs_inv", None)}, save_path)
    print(f"✅ Model saved at {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection models on MVTec AD")
    parser.add_argument("--model", choices=["padim", "patchcore"], required=True, help="Model type")
    parser.add_argument("--class_name", required=True, help="MVTec class name (e.g., 'bottle')")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=int(env.NUM_WORKERS))
    args = parser.parse_args()

    output_dir = env.RUNS_DIR / args.model / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = env.DEVICE if torch.cuda.is_available() or env.DEVICE == "cpu" else "cpu"

    train(args.model, args.class_name, args.batch_size, args.num_workers, device, output_dir)


if __name__ == "__main__":
    main()