import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import Tensor
import numpy as np
from tqdm.auto import tqdm



class FeatureExtractor(nn.Module):
    """Wraps a pretrained backbone and extracts intermediate feature maps."""

    def __init__(self, backbone: str = "resnet50", layers=None):
        super().__init__()
        if layers is None:
            layers = ["layer1", "layer2", "layer3"]

        backbone_model = getattr(models, backbone)(pretrained=True)
        self.layers = layers
        self.feature_extractor = nn.ModuleDict()
        self.feature_extractor["layer1"] = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
            backbone_model.layer1,
        )
        self.feature_extractor["layer2"] = backbone_model.layer2
        self.feature_extractor["layer3"] = backbone_model.layer3

    def forward(self, x: Tensor):
        feats = {}
        x = self.feature_extractor["layer1"](x)
        feats["layer1"] = x
        x = self.feature_extractor["layer2"](x)
        feats["layer2"] = x
        x = self.feature_extractor["layer3"](x)
        feats["layer3"] = x
        return feats


class PaDiMModel(nn.Module):
    """
    PaDiM-like anomaly detection model.
    Stores Gaussian parameters (mean, covariance) for patch embeddings.
    """

    def __init__(self, backbone: str = "resnet18", layers=None, device: str = "cuda"):
        super().__init__()
        self.device = device
        if layers is None:
            layers = ["layer1", "layer2", "layer3"]

        self.feature_extractor = FeatureExtractor(backbone=backbone, layers=layers)
        self.feature_extractor.eval().to(self.device)

        self.means = None
        self.covs_inv = None

    def _embedding_concat(self, features):
        """Concatenate multi-scale features (upsample to same size)."""
        resized = []
        target_size = features[list(features.keys())[0]].shape[-2:]
        for f in features.values():
            resized.append(
                F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            )
        return torch.cat(resized, dim=1)

    @torch.no_grad()
    def fit(self, dataloader):
        """
        Extract features on normal training set and compute per-patch Gaussian stats.
        """
        self.feature_extractor.eval()
        embedding_list = []

        for imgs, _, _, _ in tqdm(dataloader, desc="[PaDiM] building stats", leave=False):
            imgs = imgs.to(self.device)
            feats = self.feature_extractor(imgs)
            emb = self._embedding_concat(feats)  # [B, C, H, W]
            emb = emb.permute(0, 2, 3, 1).contiguous().view(-1, emb.shape[1])
            embedding_list.append(emb.cpu().numpy())

        all_feats = np.concatenate(embedding_list, axis=0)  # [N*HW, C]
        mean = np.mean(all_feats, axis=0)
        cov = np.cov(all_feats, rowvar=False) + 0.01 * np.identity(all_feats.shape[1])
        cov_inv = np.linalg.inv(cov)

        self.means = mean
        self.covs_inv = cov_inv

    @torch.no_grad()
    def predict(self, imgs: Tensor):
        """
        Compute anomaly score map for given images.
        Args:
            imgs: Tensor [B,3,H,W]
        Returns:
            anomaly_map: np.ndarray [B,H,W]
            scores: np.ndarray [B] (image-level scores)
        """
        self.feature_extractor.eval()
        imgs = imgs.to(self.device)
        feats = self.feature_extractor(imgs)
        emb = self._embedding_concat(feats)  # [B,C,H,W]
        B, C, H, W = emb.shape
        emb = emb.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

        mean = torch.from_numpy(self.means).to(self.device)
        cov_inv = torch.from_numpy(self.covs_inv).to(self.device)

        diffs = emb - mean
        # Mahalanobis distance squared
        dists = torch.einsum("bi,ij,bj->b", diffs, cov_inv, diffs)
        dists = dists.view(B, H, W).cpu().numpy()

        # Image-level score = max anomaly score in map
        scores = dists.reshape(B, -1).max(axis=1)
        return dists, scores