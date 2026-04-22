"""Precompute ground-truth disparity maps from cam0/cam1 pairs.

Uses OpenCV's StereoSGBM — classical, CPU-only, no model download. Results are
noisier than RAFT-Stereo but unbiased supervision, and running it offline
keeps VRAM free for training. A WLS filter smooths the output.

Assumes `data/processed/left/` and `data/processed/right/` already exist
(produced by `data/preprocess.py`). Outputs `data/processed/disparity/NNNNNNN.npy`
as float32 pixel disparities (invalid pixels = NaN).

Run:
  python -m data.build_disparity --processed_dir data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def build_matchers(max_disp: int = 128, block: int = 5):
    # SGBM parameters tuned for 720p natural scenes.
    n = max_disp
    left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=n,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
    )
    right = cv2.ximgproc.createRightMatcher(left) if hasattr(cv2, "ximgproc") else None
    wls = None
    if hasattr(cv2, "ximgproc") and right is not None:
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
    return left, right, wls


def disparity_for_pair(l_bgr: np.ndarray, r_bgr: np.ndarray, left_m, right_m, wls) -> np.ndarray:
    lg = cv2.cvtColor(l_bgr, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2GRAY)
    dl = left_m.compute(lg, rg).astype(np.float32) / 16.0
    if wls is not None and right_m is not None:
        dr = right_m.compute(rg, lg).astype(np.float32) / 16.0
        d = wls.filter(dl, lg, disparity_map_right=dr).astype(np.float32) / 16.0
    else:
        d = dl
    # Mark invalid (SGBM returns large negatives for unmatched pixels)
    d[d < 0] = np.nan
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--max_disparity", type=int, default=128)
    ap.add_argument("--block_size", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.processed_dir).expanduser()
    left_dir = root / "left"
    right_dir = root / "right"
    out_dir = root / "disparity"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in left_dir.glob("*.png"))
    if not names:
        raise SystemExit(f"No frames in {left_dir}. Run preprocess first.")

    left_m, right_m, wls = build_matchers(args.max_disparity, args.block_size)
    if wls is None:
        print("NOTE: opencv-contrib not available; WLS filtering disabled (disparity will be noisier).")

    written = 0
    for name in tqdm(names, desc="disparity"):
        out_path = out_dir / (Path(name).stem + ".npy")
        if out_path.exists() and not args.overwrite:
            continue
        l = cv2.imread(str(left_dir / name))
        r = cv2.imread(str(right_dir / name))
        if l is None or r is None:
            continue
        d = disparity_for_pair(l, r, left_m, right_m, wls)
        np.save(out_path, d.astype(np.float32))
        written += 1

    # Save a preview visualization of a few disparities for eyeballing
    pick = np.linspace(0, len(names) - 1, min(4, len(names)), dtype=int)
    tiles = []
    for i in pick:
        d = np.load(out_dir / (Path(names[i]).stem + ".npy"))
        d_vis = np.nan_to_num(d, nan=0.0)
        d_vis = cv2.normalize(d_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_vis = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
        tiles.append(d_vis)
    if tiles:
        cv2.imwrite(str(root / "disparity_preview.png"), np.concatenate(tiles, axis=0))

    print(f"Wrote {written} disparity maps to {out_dir}")


if __name__ == "__main__":
    main()
