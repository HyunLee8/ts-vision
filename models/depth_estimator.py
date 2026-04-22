"""Monocular depth backbones — pluggable.

Supported:
  - "midas" / "DPT_Large" / "DPT_Hybrid": intel-isl MiDaS (relative inverse depth)
  - "depth_anything_v2_small" / "_base" / "_large": Depth Anything V2 via
    torch.hub (LiheYoung repo); outputs relative inverse depth, strong
    generalization.
  - "zoe": ZoeDepth (Intel ISL) — metric depth in meters.

All models are frozen; gradients don't flow through them. Input is RGB in
[0,1]. Output is [B,1,H,W]:

  - relative backbones: per-image min-max normalized to [0,1] (higher=closer)
  - metric backbone ("zoe"): depth in meters (lower=closer); a flag on the
    module (`metric=True`) tells the calibration head to use `disparity =
    baseline * focal / depth` instead of a learned scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _FrozenDepthBase(nn.Module):
    metric: bool = False
    infer_size: int = 384

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _freeze(self, net: nn.Module):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

    def _normalize_per_image(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        flat = x.view(B, -1)
        mn = flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        mx = flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        return (x - mn) / (mx - mn + 1e-6)


class MiDaSDepth(_FrozenDepthBase):
    def __init__(self, model_type: str = "DPT_Large"):
        super().__init__()
        self.net = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self._freeze(self.net)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        B, _, H, W = rgb.shape
        x = F.interpolate(rgb, size=(self.infer_size, self.infer_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        pred = self.net(x)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
        return self._normalize_per_image(pred)


class DepthAnythingV2(_FrozenDepthBase):
    """Depth Anything V2 via LiheYoung/Depth-Anything-V2 hub repo.

    Relative inverse depth. Much better generalization than MiDaS on
    out-of-distribution scenes.
    """

    SIZE_MAP = {
        "small": "vits",
        "base":  "vitb",
        "large": "vitl",
    }

    def __init__(self, size: str = "small"):
        super().__init__()
        encoder = self.SIZE_MAP.get(size, "vits")
        self.infer_size = 518  # DA-V2 default
        # Uses HF weights under the hood; model is ~25M params for "small".
        self.net = torch.hub.load(
            "LiheYoung/Depth-Anything",
            f"DepthAnything_{encoder.upper()}14",
            pretrained=True,
            trust_repo=True,
        )
        self._freeze(self.net)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        B, _, H, W = rgb.shape
        x = F.interpolate(rgb, size=(self.infer_size, self.infer_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        pred = self.net(x)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
        return self._normalize_per_image(pred)


class ZoeDepth(_FrozenDepthBase):
    """ZoeDepth — metric depth in meters. Enables physical disparity calc."""

    metric = True

    def __init__(self, variant: str = "ZoeD_N"):
        super().__init__()
        self.net = torch.hub.load("isl-org/ZoeDepth", variant, pretrained=True, trust_repo=True)
        self._freeze(self.net)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # ZoeDepth wrapper handles its own normalization/resizing via infer().
        out = self.net.infer(rgb)  # [B,1,H,W] metric depth
        if out.dim() == 3:
            out = out.unsqueeze(1)
        return out


def build_depth_model(name: str) -> _FrozenDepthBase:
    name = name.lower()
    if name in ("midas", "dpt_large"):
        return MiDaSDepth("DPT_Large")
    if name == "dpt_hybrid":
        return MiDaSDepth("DPT_Hybrid")
    if name.startswith("depth_anything_v2"):
        size = "small"
        for s in ("small", "base", "large"):
            if s in name:
                size = s; break
        return DepthAnythingV2(size)
    if name in ("zoe", "zoedepth"):
        return ZoeDepth()
    raise ValueError(f"Unknown depth model: {name}")
