import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils import env
from src.mvtec_ad.dataset import MVTecDataset
from src.mvtec_ad import transforms as T
from src.models import PaDiMModel, PatchCoreModel, AEModel, FastFlowModel


def get_model(model_name: str, device: str):
    if model_name.lower() == "padim":
        return PaDiMModel(backbone=env.BACKBONE, device=device)
    elif model_name.lower() == "patchcore":
        return PatchCoreModel(backbone=env.BACKBONE, device=device)
    elif model_name.lower() == "ae":
        return AEModel(device=device)
    if model_name.lower() == "fastflow":
        return FastFlowModel(backbone=env.BACKBONE, device=device)
    raise ValueError(f"Unknown model: {model_name}")


def train(model_name: str, class_name: str, batch_size: int, num_workers: int, device: str, output_dir: Path, epochs: int = 30, lr: float = 1e-3):
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
    print(f"Training {model_name} on class '{class_name}' with {len(train_dataset)} samples")

    if model_name.lower() in {"padim", "patchcore"}:
        model.fit(train_loader)
    elif model_name.lower() == "ae":
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            epoch_loss = 0.0
            for imgs, _, _, _ in train_loader:
                imgs = imgs.to(device)
                opt.zero_grad()
                recon = model.net(imgs)
                loss, _, _ = model.loss_fn(imgs, recon)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * imgs.size(0)
            print(f"[AE] epoch {ep}/{epochs} | loss={epoch_loss/len(train_loader.dataset):.4f}")

    elif model_name.lower() == "fastflow":
        
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for imgs, _, _, _ in train_loader:
                opt.zero_grad()
                loss = model.training_step(imgs)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * imgs.size(0)
            print(f"[FastFlow] epoch {ep}/{epochs} | nll={epoch_loss/len(train_loader.dataset):.4f}")
    else:
        raise ValueError("Unsupported model")



    # Save model
    save_path = output_dir / f"{model_name}_{class_name}.pt"
    torch.save({"model_state": model.state_dict(),
                "means": getattr(model, "means", None),
                "covs_inv": getattr(model, "covs_inv", None)}, save_path)
    print(f"Model saved at {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection models on MVTec AD")
    parser.add_argument("--model", choices=["padim", "patchcore", "ae", "fastflow"], required=True)
    parser.add_argument("--class_name", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=int(env.NUM_WORKERS))
    parser.add_argument("--epochs", type=int, default=30, help="(AE only) training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="(AE only) learning rate")
    args = parser.parse_args()

    output_dir = env.RUNS_DIR / args.model / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = env.DEVICE if torch.cuda.is_available() or env.DEVICE == "cpu" else "cpu"

    train(args.model, args.class_name, args.batch_size, args.num_workers, device, output_dir, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()