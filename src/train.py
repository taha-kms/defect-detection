import argparse
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime as dt
import torch
import json
from torch.utils.data import DataLoader
from colorama import Fore, Back, Style, init
init(autoreset=True)


from src.utils import env
from src.utils.config import load_config
from src.mvtec_ad.dataset import MVTecDataset
from src.mvtec_ad import transforms as T
from src.models import PaDiMModel, PatchCoreModel, AEModel, FastFlowModel


def get_model(model_name: str, device: str, cfg: dict):
    m = model_name.lower()
    bb = cfg.get("backbone", {}).get("name", env.BACKBONE)
    if m == "padim":
        # Note: in this simplified stub PaDiM ignores provided layers
        return PaDiMModel(backbone=bb, device=device)
    if m == "patchcore":
        return PatchCoreModel(backbone=bb, device=device)
    if m == "ae":
        base_ch = cfg.get("models", {}).get("ae", {}).get("base_ch", 32)
        return AEModel(base_ch=base_ch, device=device)
    if m == "fastflow":
        ff = cfg.get("models", {}).get("fastflow", {})
        layers = ff.get("layers", ["layer2", "layer3"])
        num_blocks = ff.get("num_blocks", 4)
        hidden = ff.get("hidden", 256)
        return FastFlowModel(
            backbone=bb,
            device=device,
            layers=layers,
            num_blocks=num_blocks,
            hidden=hidden,
        )
    raise ValueError(f"Unknown model: {model_name}")


def train(model_name: str, class_name: str, cfg: dict):
    # Resolve params (env acts as fallback)
    device = env.DEVICE if torch.cuda.is_available() or env.DEVICE == "cpu" else "cpu"
    bs = cfg.get("train", {}).get("batch_size", 16)
    nw = cfg.get("train", {}).get("num_workers", int(env.NUM_WORKERS))
    epochs = cfg.get("train", {}).get("epochs", 30)
    lr = cfg.get("train", {}).get("lr", 1e-3)

    im_size = cfg.get("data", {}).get("image_size", env.IMAGE_SIZE)
    center_crop = cfg.get("data", {}).get("center_crop", im_size)

    # Data
    transform = T.get_image_transform(image_size=im_size, center_crop=center_crop)
    mask_transform = T.get_mask_transform(image_size=im_size, center_crop=center_crop)
    train_dataset = MVTecDataset(
        root=env.DATA_DIR,
        class_name=class_name,
        split="train",
        transform=transform,
        mask_transform=mask_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw)

    # Model
    run_id = dt.now().strftime("%Y%m%d-%H%M%S")
    base_dir = env.RUNS_DIR / model_name / class_name
    output_dir = env.RUNS_DIR / model_name / class_name / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model = get_model(model_name, device, cfg)
    print(f"Training {model_name} on '{class_name}' (bs={bs}, epochs={epochs}, lr={lr})")

    latest_dir = base_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    # Training logic per model
    m = model_name.lower()
    if m in {"padim", "patchcore"}:
        # Non-gradient "fitting" procedures
        model.fit(train_loader)
    elif m == "ae":
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"[AE] epoch {ep}/{epochs}", leave=False)
            for imgs, _, _, _ in pbar:
                imgs = imgs.to(device)
                opt.zero_grad()
                recon = model.net(imgs)
                loss, _, _ = model.loss_fn(imgs, recon)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * imgs.size(0)

            avg = epoch_loss/len(train_loader.dataset)
            pbar.close()
            print(f"[AE] epoch {ep}/{epochs} | loss={avg:.4f}")


    elif m == "fastflow":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"[FastFlow] epoch {ep}/{epochs}", leave=False)
            for imgs, _, _, _ in pbar:
                opt.zero_grad()
                loss = model.training_step(imgs)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * imgs.size(0)
            avg = epoch_loss/len(train_loader.dataset)
            pbar.close()
            print(f"[FastFlow] epoch {ep}/{epochs} | nll={epoch_loss/len(train_loader.dataset):.4f}")
    else:
        raise ValueError("Unsupported model")

    # ---- Single checkpoint save (includes PaDiM stats when available) ----
    save_path = output_dir / f"{model_name}_{class_name}.pt"
    save_latest_path = latest_dir / f"{model_name}_{class_name}.pt"
    ckpt = {"model_state": model.state_dict()}

    # Store per-model extras
    if hasattr(model, "means"):      # PaDiM
        ckpt["means"] = getattr(model, "means")
    if hasattr(model, "covs_inv"):   # PaDiM
        ckpt["covs_inv"] = getattr(model, "covs_inv")

    # PatchCore memory bank (NearestNeighbors) — save fitted patch embeddings
    if m == "patchcore" and getattr(model, "memory_bank", None) is not None:
        # sklearn stores the fitted data in _fit_X
        mb_data = getattr(model.memory_bank, "_fit_X", None)
        if mb_data is not None:
            # torch.save can handle numpy arrays; ensure it’s a dense ndarray
            import numpy as np
            mb_np = np.asarray(mb_data)
            ckpt["memory_bank_data"] = mb_np
            # also store neighbors hyperparam for convenience
            ckpt["n_neighbors"] = getattr(model, "n_neighbors", 1)
            print(f"Saved PatchCore memory bank with {mb_np.shape[0]} patches (dim={mb_np.shape[1]}).")

    torch.save(ckpt, save_path)
    torch.save(ckpt, save_latest_path)

    (output_dir / "run.json").write_text(json.dumps({
    "run_id": run_id,
    "model": model_name,
    "class": class_name,
    "config": cfg,
    }, indent=2))
    print(f"Model saved at {save_path}")

    (latest_dir / "run.json").write_text(json.dumps({
    "model": model_name,
    "class": class_name,
    "config": cfg,
    }, indent=2))

    print(Back.WHITE + Fore.BLACK + "run_id: " + run_id + Style.RESET_ALL)

def main():
    parser = argparse.ArgumentParser(description="Train models with YAML config")
    parser.add_argument("--model", required=True, choices=["ae", "padim", "patchcore", "fastflow"])
    parser.add_argument("--class_name", required=True)
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="YAML config file"
    )
    parser.add_argument(
        "--extra", type=str, nargs="*", default=[], help="Extra YAMLs merged after --config"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, *args.extra)
    train(args.model, args.class_name, cfg)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
