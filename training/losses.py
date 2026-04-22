"""Loss functions for stereo synthesis training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


class VGGPerceptualLoss(nn.Module):
    """VGG16 feature-matching loss at relu1_2, relu2_2, relu3_3."""

    def __init__(self):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        # Slice indices for relu1_2, relu2_2, relu3_3
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _feat(self, x):
        x = (x - self.mean) / self.std
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return f1, f2, f3

    def forward(self, pred: torch.Tensor, target: torch.Tensor, max_side: int = 384) -> torch.Tensor:
        # Downsample to cap memory on small GPUs; perceptual quality is preserved.
        h, w = pred.shape[-2:]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            new_hw = (int(h * scale), int(w * scale))
            pred = F.interpolate(pred, size=new_hw, mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=new_hw, mode="bilinear", align_corners=False)
        f_p = self._feat(pred)
        f_t = self._feat(target)
        return sum(F.l1_loss(p, t) for p, t in zip(f_p, f_t)) / 3.0


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    device, dtype = pred.device, pred.dtype
    g1 = _gaussian_window(window_size, 1.5, device, dtype)
    window = (g1[:, None] @ g1[None, :]).expand(pred.size(1), 1, window_size, window_size)
    pad = window_size // 2
    mu_x = F.conv2d(pred, window, padding=pad, groups=pred.size(1))
    mu_y = F.conv2d(target, window, padding=pad, groups=pred.size(1))
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(pred * pred, window, padding=pad, groups=pred.size(1)) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=pad, groups=pred.size(1)) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=pred.size(1)) - mu_xy
    ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (ssim_n / ssim_d).mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(pred, target)


def warp_by_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp img by optical flow. img: [B,C,H,W], flow: [B,2,H,W] in px."""
    B, C, H, W = img.shape
    ys = torch.linspace(-1.0, 1.0, H, device=img.device).view(1, H, 1).expand(B, H, W)
    xs = torch.linspace(-1.0, 1.0, W, device=img.device).view(1, 1, W).expand(B, H, W)
    fx = flow[:, 0] / max(W - 1, 1) * 2.0
    fy = flow[:, 1] / max(H - 1, 1) * 2.0
    grid = torch.stack([xs + fx, ys + fy], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def temporal_consistency_loss(pred_t: torch.Tensor, pred_t1: torch.Tensor, flow_t_to_t1: torch.Tensor) -> torch.Tensor:
    """L1 between pred_t warped forward by flow and pred_t1."""
    warped = warp_by_flow(pred_t, flow_t_to_t1)
    return F.l1_loss(warped, pred_t1)


def masked_disparity_l1(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Robust Charbonnier loss on disparity where valid (1) pixels are."""
    diff = pred - gt
    loss = torch.sqrt(diff * diff + 1e-3)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def edge_aware_smoothness(disparity: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(disparity[:, :, :, 1:] - disparity[:, :, :, :-1])
    dy = torch.abs(disparity[:, :, 1:, :] - disparity[:, :, :-1, :])
    ix = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    iy = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)
    return (dx * torch.exp(-ix)).mean() + (dy * torch.exp(-iy)).mean()
