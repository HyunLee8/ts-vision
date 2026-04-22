"""Preprocess raw ArduCam footage into aligned, rotation-corrected stereo pairs.

Expected input layout (either works):
  input_dir/
    <session>/cam0_*.mp4
    <session>/cam1_*.mp4
  OR
    input_dir/cam0/*.mp4
    input_dir/cam1/*.mp4

Run:
  python -m data.preprocess --input_dir ~/Desktop/footage --output_dir data/processed
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from utils.config import load_config


ROT_MAP = {
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
    "none": None,
}


@dataclass
class ClipPair:
    cam0: Path
    cam1: Path
    session: str


def find_pairs(input_dir: Path) -> list[ClipPair]:
    input_dir = input_dir.expanduser()
    pairs: list[ClipPair] = []
    # Case A: cam0/ cam1/ subfolders
    if (input_dir / "cam0").is_dir() and (input_dir / "cam1").is_dir():
        c0 = sorted((input_dir / "cam0").glob("*.mp4"))
        c1 = sorted((input_dir / "cam1").glob("*.mp4"))
        for a, b in zip(c0, c1):
            pairs.append(ClipPair(a, b, a.stem))
        return pairs
    # Case B: session subfolders each containing cam0_*.mp4 and cam1_*.mp4
    for session_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        c0 = sorted(session_dir.glob("cam0*.mp4"))
        c1 = sorted(session_dir.glob("cam1*.mp4"))
        if c0 and c1:
            pairs.append(ClipPair(c0[0], c1[0], session_dir.name))
    return pairs


def rotate_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    op = ROT_MAP.get(mode, None)
    if op is None:
        return frame
    return cv2.rotate(frame, op)


def frame_signature(frame: np.ndarray) -> float:
    """Compact scalar signature of a frame used for temporal cross-correlation.

    Using mean luminance is cheap and robust to small exposure changes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def compute_signatures(video: cv2.VideoCapture, rot_mode: str, n_frames: int) -> np.ndarray:
    sigs = []
    for _ in range(n_frames):
        ok, frame = video.read()
        if not ok:
            break
        frame = rotate_frame(frame, rot_mode)
        sigs.append(frame_signature(frame))
    return np.asarray(sigs, dtype=np.float32)


def estimate_offset(sig_a: np.ndarray, sig_b: np.ndarray, max_offset: int) -> int:
    """Return the integer offset k such that sig_b[t] ~ sig_a[t + k].

    Positive k: cam1 starts k frames earlier than cam0 (cam0 is delayed).
    """
    a = sig_a - sig_a.mean()
    b = sig_b - sig_b.mean()
    best_k, best_score = 0, -np.inf
    for k in range(-max_offset, max_offset + 1):
        if k >= 0:
            x, y = a[: len(a) - k], b[k:]
        else:
            x, y = a[-k:], b[: len(b) + k]
        n = min(len(x), len(y))
        if n < 10:
            continue
        score = float(np.dot(x[:n], y[:n]) / (np.linalg.norm(x[:n]) * np.linalg.norm(y[:n]) + 1e-8))
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def advance(cap: cv2.VideoCapture, n: int) -> None:
    for _ in range(n):
        cap.grab()


def process_pair(
    pair: ClipPair,
    out_left: Path,
    out_right: Path,
    resolution_hw: tuple[int, int],
    rot_cam0: str,
    rot_cam1: str,
    left_is_cam0: bool,
    align_seconds: float,
    max_offset_frames: int,
    frame_counter_start: int,
) -> tuple[int, dict]:
    cap0 = cv2.VideoCapture(str(pair.cam0))
    cap1 = cv2.VideoCapture(str(pair.cam1))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError(f"Failed to open video(s) for session {pair.session}")

    fps0 = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    n_align = int(align_seconds * fps0)

    sig0 = compute_signatures(cap0, rot_cam0, n_align)
    sig1 = compute_signatures(cap1, rot_cam1, n_align)
    offset = estimate_offset(sig0, sig1, max_offset_frames)

    # Reset and advance by offset to align streams
    cap0.release(); cap1.release()
    cap0 = cv2.VideoCapture(str(pair.cam0))
    cap1 = cv2.VideoCapture(str(pair.cam1))
    if offset > 0:
        advance(cap1, offset)
    elif offset < 0:
        advance(cap0, -offset)

    H, W = resolution_hw
    n_written = 0
    idx = frame_counter_start
    while True:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not (ok0 and ok1):
            break
        f0 = rotate_frame(f0, rot_cam0)
        f1 = rotate_frame(f1, rot_cam1)
        f0 = cv2.resize(f0, (W, H), interpolation=cv2.INTER_AREA)
        f1 = cv2.resize(f1, (W, H), interpolation=cv2.INTER_AREA)
        left, right = (f0, f1) if left_is_cam0 else (f1, f0)
        cv2.imwrite(str(out_left / f"{idx:07d}.png"), left)
        cv2.imwrite(str(out_right / f"{idx:07d}.png"), right)
        idx += 1
        n_written += 1

    cap0.release(); cap1.release()
    info = {"session": pair.session, "offset_frames": int(offset), "pairs": n_written}
    return n_written, info


def save_preview(out_dir: Path, left_dir: Path, right_dir: Path, n: int = 4) -> None:
    lefts = sorted(left_dir.glob("*.png"))
    if not lefts:
        return
    pick = np.linspace(0, len(lefts) - 1, num=min(n, len(lefts)), dtype=int)
    tiles = []
    for i in pick:
        l = cv2.imread(str(lefts[i]))
        r = cv2.imread(str(right_dir / lefts[i].name))
        tiles.append(np.concatenate([l, r], axis=1))
    grid = np.concatenate(tiles, axis=0)
    cv2.imwrite(str(out_dir / "preview.png"), grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--limit_sessions", type=int, default=0, help="Process at most N sessions (0=all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    input_dir = Path(args.input_dir or data_cfg["input_dir"]).expanduser()
    output_dir = Path(args.output_dir or data_cfg["processed_dir"]).expanduser()
    left_dir = output_dir / "left"
    right_dir = output_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    rot_cam0 = data_cfg["rotation"]["cam0"]
    rot_cam1 = data_cfg["rotation"]["cam1"]
    left_is_cam0 = data_cfg["left_cam"] == "cam0"
    H, W = data_cfg["resolution"]

    pairs = find_pairs(input_dir)
    if args.limit_sessions > 0:
        pairs = pairs[: args.limit_sessions]
    if not pairs:
        raise SystemExit(f"No cam0/cam1 video pairs found under {input_dir}")

    total = 0
    summary = []
    frame_idx = 0
    for pair in tqdm(pairs, desc="sessions"):
        n, info = process_pair(
            pair,
            left_dir,
            right_dir,
            (H, W),
            rot_cam0,
            rot_cam1,
            left_is_cam0,
            data_cfg["sync"]["align_seconds"],
            data_cfg["sync"]["max_offset_frames"],
            frame_idx,
        )
        total += n
        frame_idx += n
        summary.append(info)

    save_preview(output_dir, left_dir, right_dir)
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"total_pairs": total, "sessions": summary}, f, indent=2)
    print(f"Wrote {total} frame pairs to {output_dir}")


if __name__ == "__main__":
    main()
