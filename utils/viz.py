"""Visualization helpers."""
from __future__ import annotations

import numpy as np
import torch


def side_by_side(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Concatenate two [B,3,H,W] tensors horizontally -> [B,3,H,2W]."""
    return torch.cat([left, right], dim=-1)


def disparity_heatmap(disp: torch.Tensor) -> torch.Tensor:
    """Normalize [B,1,H,W] disparity to a viewable [B,3,H,W] tensor."""
    d = disp.detach().float()
    d = (d - d.amin()) / (d.amax() - d.amin() + 1e-6)
    return d.repeat(1, 3, 1, 1)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
