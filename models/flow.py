"""Frozen RAFT optical flow (torchvision) for temporal consistency losses."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class RaftFlow(nn.Module):
    """Computes forward optical flow from frame t -> t+1. Frozen."""

    def __init__(self):
        super().__init__()
        self.weights = Raft_Small_Weights.DEFAULT
        self.net = raft_small(weights=self.weights, progress=False).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.transforms = self.weights.transforms()

    @torch.no_grad()
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """img1, img2: [B,3,H,W] in [0,1]. Returns [B,2,H,W] flow in pixels."""
        # torchvision transforms expect [0,1] and handle normalization/padding.
        a, b = self.transforms(img1, img2)
        flows = self.net(a, b)
        return flows[-1]  # final refinement
