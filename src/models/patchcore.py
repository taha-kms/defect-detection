
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm.auto import tqdm

class PatchCoreModel(nn.Module):
    def __init__(self, backbone: str = "resnet18", layers=None, device: str = "cuda", n_neighbors: int = 1):
        super().__init__()
        if layers is None:
            layers = ["layer2", "layer3"]

        self.device = device
        self.backbone = models.__dict__[backbone](pretrained=True).to(device).eval()
        self.layers = layers
        self.feature_maps = {}
        self.n_neighbors = n_neighbors
        self.memory_bank = None

        # Register hooks
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()
            return hook

        for lname in layers:
            layer = getattr(self.backbone, lname)
            layer.register_forward_hook(get_activation(lname))

    def _embed(self, x: torch.Tensor) -> np.ndarray:
        """Extract concatenated patch embeddings for one batch of images."""
        _ = self.backbone(x)
        feats = [self.feature_maps[l] for l in self.layers]
        target_size = feats[0].shape[-2:]
        feats = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in feats]
        emb = torch.cat(feats, dim=1)  # [B, C, H, W]
        B, C, H, W = emb.shape
        emb = emb.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        return emb.cpu().numpy(), (B, H, W)

    @torch.no_grad()
    def fit(self, dataloader):
        """
        Build memory bank from normal training patches.
        """
        all_feats = []
        for imgs, _, _, _ in tqdm(dataloader, desc="[PatchCore] building memory bank", leave=False):
            imgs = imgs.to(self.device)
            feats, _ = self._embed(imgs)
            all_feats.append(feats)
        all_feats = np.concatenate(all_feats, axis=0)
        self.memory_bank = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.memory_bank.fit(all_feats)
        print(f"✅ Memory bank built with {len(all_feats)} patches.")

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor):
        """
        Compute anomaly maps and image-level scores using nearest-neighbor distances.
        """
        imgs = imgs.to(self.device)
        feats, shape = self._embed(imgs)  # [B*H*W, C]
        B, H, W = shape

        # Query memory bank
        distances, _ = self.memory_bank.kneighbors(feats, return_distance=True)
        # Use distance to nearest neighbor as anomaly score
        patch_scores = distances[:, 0].reshape(B, H, W)

        # Image-level score = max patch distance
        img_scores = patch_scores.reshape(B, -1).max(axis=1)

        return patch_scores, img_scores