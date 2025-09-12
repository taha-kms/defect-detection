import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    def __init__(
        self,
        root: Path,
        class_name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):


        self.root = Path(root)
        self.class_name = class_name
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        self.data = []
        self._prepare()

    def _prepare(self):
        base_dir = self.root / self.class_name / self.split
        if not base_dir.exists():
            raise FileNotFoundError(f"Dataset not found at {base_dir}")

        if self.split == "train":
            # Only "good" images for training
            img_dir = base_dir / "good"
            for img_path in sorted(img_dir.glob("*.png")):
                self.data.append((img_path, None, 0))  # label=0 (normal)
        else:
            # Test: include both good and defective categories
            for defect_type in sorted(os.listdir(base_dir)):
                img_dir = base_dir / defect_type
                if not img_dir.is_dir():
                    continue
                gt_dir = base_dir / "ground_truth" / defect_type
                for img_path in sorted(img_dir.glob("*.png")):
                    if defect_type == "good":
                        self.data.append((img_path, None, 0))
                    else:
                        mask_path = gt_dir / (img_path.stem + "_mask.png")
                        self.data.append((img_path, mask_path, 1))  # label=1 (defect)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        img_path, mask_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if mask_path is None:
            mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
        else:
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = torch.from_numpy((mask != 0).astype("float32")).unsqueeze(0)

        return image, mask, label, str(img_path)
