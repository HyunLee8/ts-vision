"""Stereo synthesis: depth -> disparity -> warp -> refinement U-Net."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18, ResNet18_Weights


class DisparityCalibration(nn.Module):
    """Maps depth to pixel disparity.

    If `metric=False` (relative depth, higher=closer): learned affine +
    spatial residual, disparity = scale*depth + shift + residual(depth).

    If `metric=True` (metric depth in meters, lower=closer): physical
    disparity = baseline_px_m / depth, then a learned residual refines it.
    baseline_px_m is derived from camera baseline (m) and learnable focal px.
    """

    def __init__(
        self,
        init_scale: float = 20.0,
        init_shift: float = 0.0,
        metric: bool = False,
        baseline_mm: float = 12.0,
        init_focal_px: float = 900.0,
    ):
        super().__init__()
        self.metric = metric
        if metric:
            self.log_focal = nn.Parameter(torch.tensor(math.log(init_focal_px)))
            self.baseline_m = baseline_mm / 1000.0
        else:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
            self.shift = nn.Parameter(torch.tensor(float(init_shift)))
        self.residual = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if self.metric:
            focal = torch.exp(self.log_focal)
            base = (self.baseline_m * focal) / depth.clamp_min(1e-3)
            return base + 2.0 * torch.tanh(self.residual(base))
        base = self.scale * depth + self.shift
        return base + 2.0 * torch.tanh(self.residual(depth))


def warp_by_disparity(img: torch.Tensor, disparity: torch.Tensor, direction: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp `img` by horizontal `disparity` (pixels).

    direction="to_left":  treat `img` as the right-eye view and produce the left-eye view.
                          Sample from x + disparity.
    direction="to_right": treat `img` as the left-eye view and produce the right-eye view.
                          Sample from x - disparity.

    Returns (warped, validity_mask) where validity_mask is 1 where the sample fell in-frame.
    """
    B, C, H, W = img.shape
    device = img.device
    ys = torch.linspace(-1.0, 1.0, H, device=device).view(1, H, 1).expand(B, H, W)
    xs = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, W).expand(B, H, W)
    disp_norm = (disparity.squeeze(1) / max(W - 1, 1)) * 2.0
    if direction == "to_left":
        xs_new = xs + disp_norm
    elif direction == "to_right":
        xs_new = xs - disp_norm
    else:
        raise ValueError(direction)
    grid = torch.stack([xs_new, ys], dim=-1)
    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = ((xs_new >= -1) & (xs_new <= 1)).float().unsqueeze(1)
    return warped, valid


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RefinementUNet(nn.Module):
    """Lightweight U-Net that refines a warped stereo view given disparity+occlusion."""

    def __init__(self, in_channels: int = 5, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chans = [base_channels * (2 ** i) for i in range(depth)]
        self.inc = _ConvBlock(in_channels, chans[0])
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(_ConvBlock(chans[i], chans[i + 1]))
        self.ups = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.ups.append(_ConvBlock(chans[i] + chans[i - 1], chans[i - 1]))
        self.out = nn.Conv2d(chans[0], 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_ckpt = self.training and x.requires_grad
        def _ck(m, t):
            return checkpoint(m, t, use_reentrant=False) if use_ckpt else m(t)
        feats = [_ck(self.inc, x)]
        for d in self.downs:
            feats.append(_ck(d, F.avg_pool2d(feats[-1], 2)))
        h = feats[-1]
        for i, up in enumerate(self.ups):
            skip = feats[-2 - i]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = _ck(up, torch.cat([h, skip], dim=1))
        return torch.sigmoid(self.out(h))


class ResNetRefiner(nn.Module):
    """Refinement network with a pretrained ResNet18 encoder.

    Input  : [B, 5, H, W] (warped RGB + disparity + occlusion mask)
    Output : [B, 3, H, W] refined RGB in [0,1]

    The 5-channel input is mapped to 3 via a 1x1 conv before feeding the
    ImageNet-pretrained backbone (so the pretraining isn't wasted on a
    mismatched stem).
    """

    def __init__(self, in_channels: int = 5, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        rn = resnet18(weights=weights)
        self.stem = nn.Conv2d(in_channels, 3, 1)
        nn.init.kaiming_normal_(self.stem.weight)
        # Encoder stages (drop avgpool/fc)
        self.enc0 = nn.Sequential(rn.conv1, rn.bn1, rn.relu)  # /2, 64c
        self.enc1 = nn.Sequential(rn.maxpool, rn.layer1)      # /4, 64c
        self.enc2 = rn.layer2                                  # /8, 128c
        self.enc3 = rn.layer3                                  # /16, 256c
        self.enc4 = rn.layer4                                  # /32, 512c

        def _up(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
            )

        self.up4 = _up(512 + 256, 256)
        self.up3 = _up(256 + 128, 128)
        self.up2 = _up(128 + 64, 64)
        self.up1 = _up(64 + 64, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        y = self.stem(x)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        y = (y - mean) / std
        e0 = self.enc0(y)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        def _cat_up(a, b):
            a = F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([a, b], dim=1)

        d = self.up4(_cat_up(e4, e3))
        d = self.up3(_cat_up(d, e2))
        d = self.up2(_cat_up(d, e1))
        d = self.up1(_cat_up(d, e0))
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.out(d))


class StereoSynthesizer(nn.Module):
    """End-to-end stereo synthesis from a monocular RGB + frozen depth."""

    def __init__(
        self,
        depth_model: nn.Module,
        init_scale: float = 20.0,
        init_shift: float = 0.0,
        refinement_channels: int = 32,
        refinement_blocks: int = 4,
        refiner_type: str = "unet",           # "unet" | "resnet18"
        baseline_mm: float = 12.0,
        init_focal_px: float = 900.0,
    ):
        super().__init__()
        self.depth_model = depth_model  # frozen
        metric = getattr(depth_model, "metric", False)
        self.calibration = DisparityCalibration(
            init_scale=init_scale, init_shift=init_shift,
            metric=metric, baseline_mm=baseline_mm, init_focal_px=init_focal_px,
        )
        if refiner_type == "resnet18":
            self.refine = ResNetRefiner(in_channels=5, pretrained=True)
        else:
            self.refine = RefinementUNet(in_channels=5, base_channels=refinement_channels, depth=refinement_blocks)

    def synthesize(self, source_rgb: torch.Tensor, direction: str) -> dict:
        depth = self.depth_model(source_rgb)
        disparity = self.calibration(depth)
        warped, valid = warp_by_disparity(source_rgb, disparity, direction)
        occlusion = 1.0 - valid
        refined = self.refine(torch.cat([warped, disparity, occlusion], dim=1))
        # Blend: keep warped content where valid, let refinement fill disocclusions.
        out = warped * valid + refined * (1.0 - valid)
        return {
            "depth": depth,
            "disparity": disparity,
            "warped": warped,
            "valid": valid,
            "refined": refined,
            "output": out.clamp(0.0, 1.0),
        }

    def forward(self, monocular_rgb: torch.Tensor, direction: str = "to_right") -> dict:
        return self.synthesize(monocular_rgb, direction)
