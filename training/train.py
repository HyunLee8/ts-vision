"""Training loop — v2.

Uses:
  - pluggable depth backbone (MiDaS / Depth Anything V2 / ZoeDepth)
  - cached stereo disparity supervision from cv2 SGBM
  - temporal consistency loss via frozen RAFT optical flow
  - ResNet18-encoder-based refinement (or lightweight U-Net)

Single-GPU default. DDP via torchrun. Mixed precision on CUDA.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import StereoPairDataset, ExternalStereoDataset, build_splits, collate
from models.depth_estimator import build_depth_model
from models.stereo_synth import StereoSynthesizer
from training.losses import (
    VGGPerceptualLoss, ssim, ssim_loss, edge_aware_smoothness,
    masked_disparity_l1, temporal_consistency_loss, psnr,
)
from utils.config import load_config
from utils.device import get_device


def _seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _maybe_init_dist() -> tuple[bool, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank(); world = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
        return True, rank, world
    return False, 0, 1


def build_loaders(cfg, world: int):
    processed = Path(cfg["data"]["processed_dir"]).expanduser()
    n = len(sorted((processed / "left").glob("*.png")))
    train_idx, val_idx = build_splits(n, cfg["data"]["train_val_split"], seed=cfg["training"]["seed"])
    aug = cfg["training"]["augment"]
    use_disp = cfg["training"].get("use_disparity_gt", True)
    use_temp = cfg["training"].get("use_temporal", True)

    train_ds = StereoPairDataset(processed, indices=train_idx, augment=aug, training=True,
                                 use_disparity_gt=use_disp, use_temporal=use_temp)
    val_ds = StereoPairDataset(processed, indices=val_idx, augment=None, training=False,
                               use_disparity_gt=use_disp, use_temporal=False)

    ext_paths = cfg["training"].get("external_datasets", [])
    if ext_paths:
        from torch.utils.data import ConcatDataset
        crop = cfg["training"]["augment"].get("crop_size", [384, 640])
        ext_datasets = [ExternalStereoDataset(p, crop_size=tuple(crop)) for p in ext_paths]
        train_ds = ConcatDataset([train_ds] + ext_datasets)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if world > 1 else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False) if world > 1 else None

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=cfg["training"]["num_workers"], pin_memory=True,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, sampler=val_sampler,
                            num_workers=cfg["training"]["num_workers"], pin_memory=True,
                            collate_fn=collate)
    return train_loader, val_loader, train_sampler


def compute_losses(out, target, source, disp_gt, disp_valid, weights, perceptual):
    pred = out["output"]
    logs = {}
    loss = weights["l1"] * F.l1_loss(pred, target); logs["l1"] = loss.detach().clone()
    if weights["perceptual"] > 0:
        perc = perceptual(pred, target); loss = loss + weights["perceptual"] * perc
        logs["perc"] = perc.detach()
    if weights["ssim"] > 0:
        s = ssim_loss(pred, target); loss = loss + weights["ssim"] * s
        logs["ssim"] = s.detach()
    if weights["smoothness"] > 0:
        sm = edge_aware_smoothness(out["disparity"], source); loss = loss + weights["smoothness"] * sm
        logs["smooth"] = sm.detach()
    if weights.get("disp_sup", 0.0) > 0 and disp_gt is not None:
        d_loss = masked_disparity_l1(out["disparity"], disp_gt, disp_valid)
        loss = loss + weights["disp_sup"] * d_loss
        logs["disp_sup"] = d_loss.detach()
    return loss, logs


def train(args):
    cfg = load_config(args.config)
    _seed_all(cfg["training"]["seed"])

    distributed, rank, world = _maybe_init_dist()
    device = get_device()
    is_main = rank == 0

    train_loader, val_loader, train_sampler = build_loaders(cfg, world)

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

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        if is_main:
            print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    # Optional frozen RAFT for temporal loss
    flow_net = None
    if cfg["training"]["loss_weights"].get("temporal", 0.0) > 0:
        from models.flow import RaftFlow
        flow_net = RaftFlow().to(device).eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["training"]["epochs"])

    perceptual = VGGPerceptualLoss().to(device)

    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank % torch.cuda.device_count()] if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"]).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg["training"]["log_dir"]) if is_main else None

    w = cfg["training"]["loss_weights"]
    best_ssim = -1.0
    global_step = 0

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, disable=not is_main, desc=f"train {epoch}")
        for batch in pbar:
            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            disp_gt = batch["disp_gt"].to(device) if "disp_gt" in batch else None
            disp_valid = batch["disp_valid"].to(device) if "disp_valid" in batch else None

            opt.zero_grad(set_to_none=True)
            m = model.module if distributed else model

            # Direction 1: left -> right
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_r = m.synthesize(left, direction="to_right")
                loss_r, logs_r = compute_losses(out_r, right, left, disp_gt, disp_valid, w, perceptual)
                cyc_r = F.l1_loss(out_r["output"], right)
                loss_r = loss_r + w["stereo_consistency"] * cyc_r
            scaler.scale(loss_r).backward()
            del out_r
            torch.cuda.empty_cache()

            # Direction 2: right -> left
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_l = m.synthesize(right, direction="to_left")
                loss_l, logs_l = compute_losses(out_l, left, right, None, None, w, perceptual)
                cyc_l = F.l1_loss(out_l["output"], left)
                loss_l = loss_l + w["stereo_consistency"] * cyc_l
            scaler.scale(loss_l).backward()
            del out_l
            torch.cuda.empty_cache()

            # Temporal consistency on the right-view synthesis between t and t+1
            if flow_net is not None and "left_next" in batch:
                left_next = batch["left_next"].to(device)
                right_next = batch["right_next"].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred_t = m.synthesize(left, direction="to_right")["output"]
                    pred_t1 = m.synthesize(left_next, direction="to_right")["output"]
                    flow = flow_net(pred_t.float(), pred_t1.float())
                    t_loss = w["temporal"] * temporal_consistency_loss(pred_t, pred_t1, flow)
                scaler.scale(t_loss).backward()
                if is_main and writer:
                    writer.add_scalar("train/temporal", t_loss.item(), global_step)

            scaler.unscale_(opt)

            total_loss = loss_r + loss_l
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                opt.zero_grad(set_to_none=True)
                scaler.update()
                if is_main:
                    pbar.set_postfix(l="SKIPPED_NAN")
                global_step += 1
                continue

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(opt); scaler.update()

            if is_main and writer is not None:
                writer.add_scalar("train/loss", total_loss.item(), global_step)
                for k, v in logs_r.items():
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
                writer.add_scalar("train/cycle", (cyc_r + cyc_l).item(), global_step)
                pbar.set_postfix(l=f"{total_loss.item():.3f}")
            global_step += 1

            if args.dry_run:
                print("[dry_run] one batch OK; exiting.")
                return

        sched.step()

        if is_main and (epoch % cfg["training"]["val_every"] == 0):
            model.eval()
            ssim_sum, psnr_sum, n = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    left = batch["left"].to(device); right = batch["right"].to(device)
                    out = (model.module if distributed else model).synthesize(left, direction="to_right")
                    pred = out["output"]
                    ssim_sum += ssim(pred, right).item() * left.size(0)
                    psnr_sum += psnr(pred, right).item() * left.size(0)
                    n += left.size(0)
                    if writer is not None and n <= cfg["training"]["batch_size"]:
                        writer.add_images("val/left_input", left, epoch)
                        writer.add_images("val/right_gt", right, epoch)
                        writer.add_images("val/right_pred", pred, epoch)
                        disp = out["disparity"]
                        writer.add_images("val/disparity", (disp - disp.amin()) / (disp.amax() - disp.amin() + 1e-6), epoch)
            val_ssim = ssim_sum / max(n, 1); val_psnr = psnr_sum / max(n, 1)
            writer.add_scalar("val/ssim", val_ssim, epoch)
            writer.add_scalar("val/psnr", val_psnr, epoch)
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                torch.save({"epoch": epoch, "model": (model.module if distributed else model).state_dict(),
                            "cfg": cfg, "val_ssim": val_ssim, "val_psnr": val_psnr},
                           ckpt_dir / "best.pt")

        if is_main and ((epoch + 1) % cfg["training"]["save_every"] == 0):
            torch.save({"epoch": epoch, "model": (model.module if distributed else model).state_dict(),
                        "cfg": cfg}, ckpt_dir / f"epoch_{epoch+1:03d}.pt")

    if distributed: dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
