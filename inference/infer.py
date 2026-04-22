"""Run a trained stereo synthesis model on a monocular video."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.color_transform import TreeshrewDichromatTransform
from models.depth_estimator import build_depth_model
from models.stereo_synth import StereoSynthesizer
from utils.config import load_config
from utils.device import get_device


def _str2bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y", "t")


def load_model(ckpt_path: str, cfg: dict, device: torch.device) -> StereoSynthesizer:
    depth = build_depth_model(cfg["model"]["depth_model"]).to(device)
    model = StereoSynthesizer(
        depth_model=depth,
        init_scale=cfg["model"].get("disparity_init_scale", 20.0),
        init_shift=cfg["model"].get("disparity_init_shift", 0.0),
        refinement_channels=cfg["model"]["refinement_channels"],
        refinement_blocks=cfg["model"]["refinement_blocks"],
        refiner_type=cfg["model"].get("refiner_type", "unet"),
        baseline_mm=cfg["data"]["baseline_mm"],
        init_focal_px=cfg["model"].get("init_focal_px", 900.0),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def frame_to_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--color_transform", type=_str2bool, default=None,
                    help="Override config's inference.apply_color_transform")
    ap.add_argument("--format", choices=["side_by_side", "separate"], default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    apply_color = cfg["inference"]["apply_color_transform"] if args.color_transform is None else args.color_transform
    out_format = args.format or cfg["inference"]["output_format"]
    H, W = cfg["data"]["resolution"]

    model = load_model(args.checkpoint, cfg, device)
    color = TreeshrewDichromatTransform().to(device) if apply_color else None

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if out_format == "side_by_side":
        writer = cv2.VideoWriter(args.output, fourcc, fps, (W * 2, H))
        writer_l = writer_r = None
    else:
        out_path = Path(args.output)
        writer_l = cv2.VideoWriter(str(out_path.with_name(out_path.stem + "_left.mp4")), fourcc, fps, (W, H))
        writer_r = cv2.VideoWriter(str(out_path.with_name(out_path.stem + "_right.mp4")), fourcc, fps, (W, H))
        writer = None

    with torch.no_grad():
        pbar = tqdm(total=n_frames, desc="infer")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            t = frame_to_tensor(frame, device)
            # Treat the monocular input as the "center" view and synthesize both eyes
            # by warping left and right with half disparity each.
            out_r = model.synthesize(t, direction="to_right")
            out_l = model.synthesize(t, direction="to_left")
            left_img = out_r["output"]
            right_img = out_l["output"]
            if color is not None:
                left_img = color(left_img)
                right_img = color(right_img)
            l_bgr = tensor_to_bgr(left_img)
            r_bgr = tensor_to_bgr(right_img)
            if writer is not None:
                writer.write(np.concatenate([l_bgr, r_bgr], axis=1))
            else:
                writer_l.write(l_bgr); writer_r.write(r_bgr)
            pbar.update(1)
        pbar.close()

    cap.release()
    if writer is not None: writer.release()
    if writer_l is not None: writer_l.release()
    if writer_r is not None: writer_r.release()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
