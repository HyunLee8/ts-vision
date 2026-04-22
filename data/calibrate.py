"""Camera calibration + stereo rectification stub.

When you have time to shoot a checkerboard calibration video with both
ArduCams, run this to:
  1. Detect checkerboard corners in both streams.
  2. Compute intrinsics per camera (cv2.calibrateCamera).
  3. Compute extrinsics between cams (cv2.stereoCalibrate).
  4. Build rectification maps (cv2.stereoRectify + cv2.initUndistortRectifyMap).
  5. Save the maps to data/processed/rectify.npz for use during preprocessing.

Until calibration data exists, preprocess.py skips rectification (the current
model learns to absorb residual misalignment through the refinement net).

Typical calibration footage: hold a 9x6 checkerboard (25 mm squares) in front
of both cameras, move it through the volume the treeshrew will see, cover
multiple depths and angles. 30 seconds is usually enough.

Run:
  python -m data.calibrate \
    --cam0 calib/cam0.mp4 --cam1 calib/cam1.mp4 \
    --rows 6 --cols 9 --square_mm 25 \
    --output data/processed/rectify.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_corners(cap: cv2.VideoCapture, pattern: tuple[int, int], step: int = 10):
    corners_list, frames_used = [], 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        corners_list.append(corners)
        frames_used += 1
    return corners_list, (gray.shape[1], gray.shape[0]) if corners_list else (0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam0", required=True)
    ap.add_argument("--cam1", required=True)
    ap.add_argument("--rows", type=int, default=6, help="inner corners vertically")
    ap.add_argument("--cols", type=int, default=9, help="inner corners horizontally")
    ap.add_argument("--square_mm", type=float, default=25.0)
    ap.add_argument("--step", type=int, default=10, help="sample every Nth frame")
    ap.add_argument("--output", default="data/processed/rectify.npz")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm

    c0 = cv2.VideoCapture(args.cam0); c1 = cv2.VideoCapture(args.cam1)
    corners0, size0 = detect_corners(c0, pattern, args.step)
    corners1, size1 = detect_corners(c1, pattern, args.step)
    n = min(len(corners0), len(corners1))
    if n < 10:
        raise SystemExit(f"Only {n} synchronized checkerboard frames; need ≥10.")
    obj_pts = [objp.copy() for _ in range(n)]
    corners0, corners1 = corners0[:n], corners1[:n]
    size = size0

    _, K0, d0, _, _ = cv2.calibrateCamera(obj_pts, corners0, size, None, None)
    _, K1, d1, _, _ = cv2.calibrateCamera(obj_pts, corners1, size, None, None)
    _, K0, d0, K1, d1, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, corners0, corners1, K0, d0, K1, d1, size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(K0, d0, K1, d1, size, R, T)
    map00, map01 = cv2.initUndistortRectifyMap(K0, d0, R0, P0, size, cv2.CV_16SC2)
    map10, map11 = cv2.initUndistortRectifyMap(K1, d1, R1, P1, size, cv2.CV_16SC2)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, K0=K0, d0=d0, K1=K1, d1=d1, R=R, T=T, Q=Q,
             map00=map00, map01=map01, map10=map10, map11=map11, size=size)
    print(f"Wrote rectification maps to {out} (from {n} checkerboard pairs)")


if __name__ == "__main__":
    main()
