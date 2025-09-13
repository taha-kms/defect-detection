from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- SSIM (single-scale, window=11) ----
def _gaussian_window(window_size: int, sigma: float, channels: int, device: str):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_1d = g.unsqueeze(0)  # [1, W]
    window_2d = (window_1d.t() @ window_1d).unsqueeze(0).unsqueeze(0)  # [1,1,W,W]
    window = window_2d.repeat(channels, 1, 1, 1)  # [C,1,W,W]
    return window

def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    # assumes x,y in [0,1] after normalization inverse or use raw recon vs input post-normalization
    C = x.shape[1]
    device = x.device
    window = _gaussian_window(window_size, sigma, C, device)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=C) - mu_xy

    # SSIM constants (for images in [0,1])
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return ssim_map.mean(dim=(1, 2, 3))  # per-sample SSIM


# ---- Model ----
class ConvAE(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 2, 1), nn.ReLU(inplace=True),     # H/2
            nn.Conv2d(base_ch, base_ch*2, 3, 2, 1), nn.ReLU(inplace=True), # H/4
            nn.Conv2d(base_ch*2, base_ch*4, 3, 2, 1), nn.ReLU(inplace=True),# H/8
            nn.Conv2d(base_ch*4, base_ch*4, 3, 2, 1), nn.ReLU(inplace=True),# H/16
        )
        # Bottleneck
        self.btl = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*4, 3, 1, 1), nn.ReLU(inplace=True)
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*4, 4, 2, 1), nn.ReLU(inplace=True), # x2
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1), nn.ReLU(inplace=True), # x4
            nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1), nn.ReLU(inplace=True),   # x8
            nn.ConvTranspose2d(base_ch, in_ch, 4, 2, 1), nn.Sigmoid(),                # x16 -> [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        z = self.btl(z)
        out = self.dec(z)
        return out


class AEModel(nn.Module):
    """
    Wrapper to provide:
    - training step with SSIM+MSE loss
    - predict() that returns anomaly maps & image-level scores
    """
    def __init__(self, base_ch: int = 32, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.net = ConvAE(base_ch=base_ch).to(device)

    def loss_fn(self, x: torch.Tensor, recon: torch.Tensor, w_ssim: float = 0.5, w_mse: float = 0.5):
        # Inputs are normalized by dataset transforms; recon is sigmoid [0,1].
        # For SSIM, we compare in the normalized space (ok for relative scoring).
        ssim_val = ssim(x, recon)  # per-sample
        mse_val = F.mse_loss(recon, x, reduction="none").mean(dim=(1,2,3))
        loss = w_ssim * (1 - ssim_val) + w_mse * mse_val
        return loss.mean(), ssim_val.mean().item(), mse_val.mean().item()

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor) -> Tuple:
        """
        Returns:
            anomaly_map: [B,H,W] (per-pixel 1 - SSIM_local approx via abs diff average)
            scores: [B] (image-level, max over anomaly map)
        """
        self.eval()
        recon = self.net(imgs.to(self.device))
        # Per-pixel error (channel-avg absolute diff)
        err = (imgs.to(self.device) - recon).abs().mean(dim=1)  # [B,H,W]
        # Image-level score
        scores = err.view(err.shape[0], -1).max(dim=1).values
        return err.detach().cpu().numpy(), scores.detach().cpu().numpy()