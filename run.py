#!/usr/bin/env python3
"""Treeshrew Stereo — unified CLI.

Usage:
    python run.py train                          # train from scratch
    python run.py train --resume checkpoints/epoch_020.pt
    python run.py infer  --input video.mp4 --output stereo.mp4
    python run.py eval                           # compare against ground truth
    python run.py eval   --checkpoint checkpoints/epoch_055.pt

All settings live in configs/default.yaml (or pass --config <path>).
"""
import argparse
import sys


def cmd_train(args):
    from training.train import train as _train

    class TrainArgs:
        config = args.config
        resume = args.resume
        dry_run = args.dry_run

    _train(TrainArgs())


def cmd_infer(args):
    from pathlib import Path
    import cv2, numpy as np, torch
    from tqdm import tqdm
    from inference.infer import load_model, frame_to_tensor, tensor_to_bgr
    from models.color_transform import TreeshrewDichromatTransform
    from utils.config import load_config
    from utils.device import get_device

    cfg = load_config(args.config)
    device = get_device()

    ckpt = args.checkpoint or cfg["inference"]["checkpoint"]
    apply_color = cfg["inference"]["apply_color_transform"] if args.color_transform is None else args.color_transform
    out_format = args.format or cfg["inference"]["output_format"]
    H, W = cfg["data"]["resolution"]
    input_dir = Path(cfg["inference"].get("input_dir", "input"))
    output_dir = Path(cfg["inference"].get("output_dir", "output"))

    if args.input:
        videos = [(Path(args.input), args.output)]
    else:
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        vids = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)
        if not vids:
            print(f"No videos found in {input_dir}/. Drop .mp4 files there or use --input.", file=sys.stderr)
            sys.exit(1)
        videos = [(v, None) for v in vids]
        print(f"Found {len(videos)} video(s) in {input_dir}/")

    model = load_model(ckpt, cfg, device)
    color = TreeshrewDichromatTransform().to(device) if apply_color else None

    for vid_path, out_override in videos:
        if out_override:
            out_file = out_override
        else:
            out_file = str(output_dir / f"{vid_path.stem}_stereo.mp4")

        print(f"\n▸ {vid_path.name} → {out_file}")
        _run_inference_single(str(vid_path), out_file, model, color, H, W, out_format)

    print(f"\nAll done. Results in {output_dir}/" if not args.input else "")


def _run_inference_single(input_path, output_path, model, color, H, W, out_format):
    import cv2, numpy as np, torch
    from pathlib import Path
    from tqdm import tqdm
    from inference.infer import tensor_to_bgr, frame_to_tensor

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if out_format == "side_by_side":
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W * 2, H))
        writer_l = writer_r = None
    else:
        out_path = Path(output_path)
        writer_l = cv2.VideoWriter(str(out_path.with_name(out_path.stem + "_left.mp4")), fourcc, fps, (W, H))
        writer_r = cv2.VideoWriter(str(out_path.with_name(out_path.stem + "_right.mp4")), fourcc, fps, (W, H))
        writer = None

    with torch.no_grad():
        pbar = tqdm(total=n_frames, desc=Path(input_path).stem)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            t = frame_to_tensor(frame, model.depth_model.parameters().__next__().device)
            out_r = model.synthesize(t, direction="to_right")
            out_l = model.synthesize(t, direction="to_left")
            left_img, right_img = out_r["output"], out_l["output"]
            if color is not None:
                left_img = color(left_img)
                right_img = color(right_img)
            l_bgr = tensor_to_bgr(left_img)
            r_bgr = tensor_to_bgr(right_img)
            if writer is not None:
                writer.write(np.concatenate([l_bgr, r_bgr], axis=1))
            else:
                writer_l.write(l_bgr)
                writer_r.write(r_bgr)
            pbar.update(1)
        pbar.close()

    cap.release()
    if writer is not None:
        writer.release()
    if writer_l is not None:
        writer_l.release()
    if writer_r is not None:
        writer_r.release()


def cmd_eval(args):
    import torch, cv2, numpy as np
    from skimage.metrics import structural_similarity
    from inference.infer import load_model, frame_to_tensor
    from utils.config import load_config

    cfg = load_config(args.config)
    ev = cfg.get("evaluate", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.checkpoint or ev.get("checkpoint", "checkpoints/best.pt")
    left_path = args.left or ev.get("left_frame", "data/processed/left/0000050.png")
    right_path = args.right or ev.get("right_frame", "data/processed/right/0000050.png")
    out_path = args.output or ev.get("output_image", "comparison.png")

    model = load_model(ckpt, cfg, device)
    left_gt = cv2.imread(left_path)
    right_gt = cv2.imread(right_path)
    if left_gt is None:
        raise SystemExit(f"Cannot read {left_path}")
    if right_gt is None:
        raise SystemExit(f"Cannot read {right_path}")

    t = frame_to_tensor(left_gt, device)
    with torch.no_grad():
        out = model.synthesize(t, direction="to_right")
        pred = out["output"].squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        disp = out["disparity"].squeeze().cpu().numpy()

    right_gt_rgb = cv2.cvtColor(right_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mse = np.mean((pred - right_gt_rgb) ** 2)
    psnr_val = 10 * np.log10(1.0 / mse)
    ssim_val = structural_similarity(pred, right_gt_rgb, channel_axis=2, data_range=1.0)

    print(f"Checkpoint : {ckpt}")
    print(f"PSNR       : {psnr_val:.2f} dB")
    print(f"SSIM       : {ssim_val:.4f}")
    print(f"Disparity  : {disp.min():.1f} – {disp.max():.1f} px  (mean {disp.mean():.1f})")

    pred_bgr = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    h, w = left_gt.shape[:2]
    bar_h = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["LEFT INPUT", "RIGHT GROUND TRUTH", "RIGHT PREDICTED"]
    panels = [left_gt, right_gt, pred_bgr]
    rows = []
    for label, panel in zip(labels, panels):
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        ts = cv2.getTextSize(label, font, 0.7, 2)[0]
        cv2.putText(bar, label, ((w - ts[0]) // 2, (bar_h + ts[1]) // 2), font, 0.7, (255, 255, 255), 2)
        rows.append(np.concatenate([bar, panel], axis=0))
    comp = np.concatenate(rows, axis=1)
    info = f"PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}  |  Mean Disp: {disp.mean():.1f} px"
    info_bar = np.zeros((bar_h, comp.shape[1], 3), dtype=np.uint8)
    ts = cv2.getTextSize(info, font, 0.6, 1)[0]
    cv2.putText(info_bar, info, ((comp.shape[1] - ts[0]) // 2, 28), font, 0.6, (0, 255, 255), 1)
    comp = np.concatenate([comp, info_bar], axis=0)
    cv2.imwrite(out_path, comp)
    print(f"Saved      : {out_path}")


def main():
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Treeshrew Stereo — mono-to-stereo synthesis for neuroscience",
    )
    p.add_argument("--config", default="configs/default.yaml", help="path to YAML config")
    sub = p.add_subparsers(dest="command")

    # --- train ---
    t = sub.add_parser("train", help="train the model")
    t.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    t.add_argument("--dry_run", action="store_true", help="run one batch and exit")

    # --- infer ---
    i = sub.add_parser("infer", help="run stereo inference on a video")
    i.add_argument("--input", type=str, help="input monocular video")
    i.add_argument("--output", type=str, default=None, help="output path (default: <input>_stereo.mp4)")
    i.add_argument("--checkpoint", type=str, default=None, help="model checkpoint (default: from config)")
    i.add_argument("--color_transform", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    i.add_argument("--format", choices=["side_by_side", "separate"], default=None)

    # --- eval ---
    e = sub.add_parser("eval", help="evaluate against ground-truth stereo pair")
    e.add_argument("--checkpoint", type=str, default=None, help="model checkpoint (default: from config)")
    e.add_argument("--left", type=str, default=None, help="left frame path (default: from config)")
    e.add_argument("--right", type=str, default=None, help="right frame path (default: from config)")
    e.add_argument("--output", type=str, default=None, help="comparison image path (default: from config)")

    args = p.parse_args()
    if args.command is None:
        p.print_help()
        sys.exit(0)

    {"train": cmd_train, "infer": cmd_infer, "eval": cmd_eval}[args.command](args)


if __name__ == "__main__":
    main()
