from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------- Normalizing flow building blocks ---------------- #

class ActNorm(nn.Module):
    """Per-channel affine normalization initialized by data (first batch)."""
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features))
        self.initialized = False
        self.eps = eps

    @torch.no_grad()
    def _init(self, x: torch.Tensor):
        # x: [N, C]
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + self.eps
        self.bias.data = -mean
        self.log_scale.data = (-torch.log(std))
        self.initialized = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self._init(x)
        z = (x + self.bias) * torch.exp(self.log_scale)
        logdet = self.log_scale.sum(dim=1).unsqueeze(0).expand(x.size(0), -1).sum(dim=1)  # per-sample sum over C
        return z, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z * torch.exp(-self.log_scale) - self.bias
        return x


class AffineCoupling(nn.Module):
    """
    Channel-wise affine coupling: split x into x1|x2, predict s,t for x2 from x1 via MLP.
    """
    def __init__(self, num_features: int, hidden: int = 256):
        super().__init__()
        self.num_features = num_features
        self.h = nn.Sequential(
            nn.Linear(num_features // 2, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_features)  # outputs [t | s]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[:, : self.num_features // 2], x[:, self.num_features // 2 :]
        h = self.h(x1)
        t, s = torch.chunk(h, 2, dim=1)
        s = torch.tanh(s)  # stabilize
        z2 = (x2 + t) * torch.exp(s)
        z = torch.cat([x1, z2], dim=1)
        logdet = s.sum(dim=1)
        return z, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[:, : self.num_features // 2], z[:, self.num_features // 2 :]
        h = self.h(z1)
        t, s = torch.chunk(h, 2, dim=1)
        s = torch.tanh(s)
        x2 = z2 * torch.exp(-s) - t
        x = torch.cat([z1, x2], dim=1)
        return x


class Permute(nn.Module):
    """Fixed random permutation of channels."""
    def __init__(self, num_features: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(num_features, generator=g)
        self.register_buffer("idx", idx)
        inv = torch.empty_like(idx)
        inv[idx] = torch.arange(num_features)
        self.register_buffer("inv_idx", inv)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[:, self.idx], torch.zeros(x.size(0), device=x.device)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, self.inv_idx]


class SimpleFlow(nn.Module):
    """A simple flow: [ActNorm -> (Permute -> Coupling)*K]"""
    def __init__(self, num_features: int, num_blocks: int = 4, hidden: int = 256, seed: int = 0):
        super().__init__()
        self.actnorm = ActNorm(num_features)
        blocks = []
        for k in range(num_blocks):
            blocks += [Permute(num_features, seed=seed + k), AffineCoupling(num_features, hidden)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns z and log|det J|
        logdet_total = torch.zeros(x.size(0), device=x.device)
        z, logdet = self.actnorm(x)
        logdet_total += logdet
        for b in self.blocks:
            if isinstance(b, Permute):
                z, logdet = b(z)
            else:
                z, logdet = b(z)
            logdet_total += logdet
        return z, logdet_total

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        # base distribution: standard normal
        z, logdet = self.forward(x)
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)
        log_px = log_pz + logdet
        return -log_px  # negative log-likelihood per sample


# ---------------- Feature extractor ---------------- #

class FeatureExtractor(nn.Module):
    """Extract multi-layer features from a pretrained backbone and concat."""
    def __init__(self, backbone: str = "resnet50", layers: List[str] = None):
        super().__init__()
        if layers is None:
            layers = ["layer2", "layer3"]
        self.backbone = getattr(models, backbone)(pretrained=True)
        self.backbone.eval()
        self.layers = layers
        self._feats = {}
        # register hooks
        def hooker(name):
            def hook(_, __, out):
                self._feats[name] = out
            return hook
        for lname in layers:
            getattr(self.backbone, lname).register_forward_hook(hooker(lname))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.backbone(x)
        feats = [self._feats[l] for l in self.layers]
        target = feats[0].shape[-2:]
        feats = [F.interpolate(f, size=target, mode="bilinear", align_corners=False) for f in feats]
        emb = torch.cat(feats, dim=1)  # [B,C,H,W]
        return emb


# ---------------- FastFlow model ---------------- #

class FastFlowModel(nn.Module):
    """
    Flow on features at each spatial location.
    Training: maximum likelihood on normal images.
    Inference: NLL per location -> anomaly map; image score = max NLL
    """
    def __init__(self, backbone: str = "resnet50", device: str = "cuda",
                 layers: List[str] = None, num_blocks: int = 4, hidden: int = 256, seed: int = 0):
        super().__init__()
        self.device = device
        self.feat = FeatureExtractor(backbone=backbone, layers=layers).to(device).eval()
        # placeholder to build flow after seeing channel dim
        self.flow: SimpleFlow | None = None
        self.num_blocks = num_blocks
        self.hidden = hidden
        self.seed = seed

    def _ensure_flow(self, c: int):
        if self.flow is None:
            self.flow = SimpleFlow(num_features=c, num_blocks=self.num_blocks, hidden=self.hidden, seed=self.seed).to(self.device)

    def _embed_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int]]:
        emb = self.feat(x)  # [B,C,H,W]
        B, C, H, W = emb.shape
        self._ensure_flow(C)
        z = emb.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        return z, (B, H, W)

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        self.train()
        z, _ = self._embed_batch(imgs.to(self.device))
        nll = self.flow.nll(z)
        return nll.mean()

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor):
        self.eval()
        z, shp = self._embed_batch(imgs.to(self.device))
        # compute per-sample NLL
        z_out, logdet = self.flow.forward(z)
        log_pz = -0.5 * (z_out ** 2 + math.log(2 * math.pi)).sum(dim=1)
        log_px = log_pz + logdet
        nll = (-log_px)  # [B*H*W]
        B, H, W = shp
        amap = nll.view(B, H, W).detach().cpu().numpy()
        img_scores = amap.reshape(B, -1).max(axis=1)
        return amap, img_scores

    def parameters(self, recurse: bool = True):
        # Only train flow params; feature extractor is frozen
        return self.flow.parameters() if self.flow is not None else super().parameters()