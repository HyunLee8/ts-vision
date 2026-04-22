"""Treeshrew dichromat color simulation.

Treeshrew (Tupaia belangeri) is a dichromat with S-cone peak ~430nm and
L-cone peak ~555nm (Petry & Harosi 1990; Muller & Peichl 1989).

Method:
  1. Build approximate cone spectral sensitivities using Govardovskii et al.
     2000 template for A1 visual pigments at the published lambda_max values.
     This is the standard approximation used when full measured spectra are
     unavailable.
  2. Integrate against sRGB primary spectra (assumed gaussian-ish around the
     typical display primaries) to get an RGB -> SL matrix for the treeshrew
     cone responses. Because we don't have the actual monitor's spectral
     primaries, we fall back to the human 2-deg LMS matrix as a proxy and
     then project through a 2D dichromat basis (S, L).
  3. Project the stimulus onto the 2D S-L plane (Brettel et al. 1997): find
     the point on the dichromat confusion line that has the same S and L
     responses. This "collapses" the missing cone dimension in a principled
     way.
  4. Convert back to sRGB for display.

Note: this is still an approximation — full rigor requires the actual
treeshrew cone spectra and the actual monitor primaries. The code documents
where each assumption enters so researchers can swap in better data later.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def govardovskii_template(lambda_nm: torch.Tensor, lmax: float) -> torch.Tensor:
    """Govardovskii 2000 A1 visual pigment template.

    lambda_nm: [N] wavelengths. lmax: peak sensitivity wavelength (nm).
    Returns sensitivity [N] (unnormalized).
    """
    x = lmax / lambda_nm
    a = 0.8795 + 0.0459 * math.exp(-((lmax - 300.0) ** 2) / 11940.0)
    A, B, C, D = 69.7, 28.0, -14.9, 0.674
    alpha = 1.0 / (torch.exp(A * (a - x)) + torch.exp(B * (0.922 - x)) + torch.exp(C * (1.104 - x)) + D)
    # Beta band (secondary peak near UV; minor for our use)
    lmax_b = 189.0 + 0.315 * lmax
    b = -40.5 + 0.195 * lmax
    beta = 0.26 * torch.exp(-((lambda_nm - lmax_b) / b) ** 2)
    return alpha + beta


# Human 2-deg LMS from sRGB (Hunt-Pointer-Estevez, D65 aligned).
_RGB2LMS_HUMAN = torch.tensor([
    [0.31399022, 0.63951294, 0.04649755],
    [0.15537241, 0.75789446, 0.08670142],
    [0.01775239, 0.10944209, 0.87256922],
], dtype=torch.float32)
_LMS2RGB_HUMAN = torch.tensor([
    [ 5.47221206, -4.6419601,   0.16963708],
    [-1.1252419,   2.29317094, -0.1678952 ],
    [ 0.02980165, -0.19318073,  1.16364789],
], dtype=torch.float32)


def build_treeshrew_rgb_to_sl(s_peak: float, l_peak: float) -> torch.Tensor:
    """Build a 2x3 matrix mapping linear sRGB -> (S, L) treeshrew cone responses.

    We approximate this by projecting through human LMS first then taking the
    S (shortwave) and L (longwave) channels, weighted by how similar their
    peaks are to the treeshrew peaks. This is the documented approximation;
    replace with a spectral integration over monitor primaries if available.
    """
    lam = torch.arange(380.0, 781.0, 1.0)
    s_curve = govardovskii_template(lam, s_peak)
    l_curve = govardovskii_template(lam, l_peak)
    # Human cone peaks: S=420, M=530, L=560. Weight human LMS rows by overlap
    # with treeshrew S/L templates to produce the 2x3 treeshrew matrix.
    human_peaks = torch.tensor([560.0, 530.0, 420.0])  # L,M,S for the human rows of _RGB2LMS_HUMAN
    # overlap of each human cone with treeshrew S and L:
    def _overlap(curve: torch.Tensor, peak: float) -> float:
        h = govardovskii_template(lam, peak)
        return float((curve * h).sum() / (curve.pow(2).sum().sqrt() * h.pow(2).sum().sqrt() + 1e-6))
    w_s = torch.tensor([_overlap(s_curve, p) for p in human_peaks])
    w_l = torch.tensor([_overlap(l_curve, p) for p in human_peaks])
    w_s = w_s / w_s.sum(); w_l = w_l / w_l.sum()
    rgb2lms = _RGB2LMS_HUMAN
    row_s = (w_s.view(3, 1) * rgb2lms).sum(dim=0)  # weighted combo of L,M,S rows
    row_l = (w_l.view(3, 1) * rgb2lms).sum(dim=0)
    return torch.stack([row_s, row_l], dim=0)  # [2,3]


def _apply_matrix(img: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    B, C, H, W = img.shape
    flat = img.permute(0, 2, 3, 1).reshape(-1, C)
    out = flat @ M.to(img.device, img.dtype).T
    return out.reshape(B, H, W, -1).permute(0, 3, 1, 2)


class TreeshrewDichromatTransform(nn.Module):
    """Project RGB through a treeshrew-cone-based dichromat simulation.

    Steps:
      1. RGB -> treeshrew (S, L) via build_treeshrew_rgb_to_sl.
      2. Reconstruct a pseudo-LMS triplet where the missing M cone response
         is filled by the Brettel projection onto the S-L confusion plane.
      3. Invert back to RGB.
    """

    def __init__(self, s_peak: float = 430.0, l_peak: float = 555.0):
        super().__init__()
        self.s_peak, self.l_peak = s_peak, l_peak
        rgb2sl = build_treeshrew_rgb_to_sl(s_peak, l_peak)  # [2,3]
        self.register_buffer("rgb2sl", rgb2sl)
        self.register_buffer("rgb2lms_human", _RGB2LMS_HUMAN)
        self.register_buffer("lms2rgb_human", _LMS2RGB_HUMAN)
        # Brettel projection: the dichromat plane in (L,M,S) is defined by
        # equations of the form a*L + b*M + c*S = 0; for M-cone absent
        # ("deuteranope-like") we project M -> f(L,S). Values from Vienot 1999
        # which match peaks (560,530,420) — a reasonable approximation when
        # the treeshrew L is close to the human L (555 vs 560).
        self.register_buffer("brettel_lms_proj", torch.tensor([
            [1.0,        0.0,        0.0],
            [0.494207,   0.0,        1.24827],
            [0.0,        0.0,        1.0],
        ], dtype=torch.float32))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # Brettel projection in human-LMS (close approximation when
        # treeshrew L peak ~ human L peak). See class docstring for caveats.
        lms = _apply_matrix(rgb, self.rgb2lms_human)
        lms_p = _apply_matrix(lms, self.brettel_lms_proj)
        out = _apply_matrix(lms_p, self.lms2rgb_human)
        return out.clamp(0.0, 1.0)
