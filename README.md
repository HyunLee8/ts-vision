# Treeshrew Stereo Vision

Pipeline: monocular video → treeshrew-calibrated stereo pair (with optional
dichromatic color transform) for UVA neuroscience experiments.

## Overview

UVA researchers need to replicate treeshrew vision for experiments where
treeshrews run on a wheel while viewing stereo footage and having their brains
monitored. This model takes any single-panel video and outputs a two-panel
stereo video (left eye + right eye) with ~12mm interpupillary distance matching
the treeshrew's anatomy.

## Architecture

1. **Frozen depth backbone** (MiDaS DPT_Hybrid) estimates per-frame depth
2. **Disparity calibration head** maps depth → pixel disparity (learned scale/shift + residual CNN)
3. **Disparity-based warping** shifts pixels horizontally to simulate the second eye
4. **ResNet18 refinement network** (ImageNet-pretrained encoder) fills disoccluded regions and corrects artifacts
5. **Optional dichromat color transform** simulates treeshrew S-cone (430nm) + L-cone (555nm) vision via Brettel 1997 projection

Training data: ~10,649 stereo pairs from dual ArduCam rigs (12mm baseline, cam0 rotated CCW, cam1 rotated CW).

## Server Setup

GPU server: `gpusrv19` via `portal.cs.virginia.edu` (two-hop SSH).

```bash
# From local Mac
ssh chn9bm@portal.cs.virginia.edu
ssh gpusrv19
cd ~/TS_FINAL
```

Python 3.10 with `--user` installed packages (no venv/conda — `python3-venv`
not available on the server). CUDA 12.8, RTX 4000 Ada (20GB × 4, but usually
only GPU 0 has free VRAM).

PyTorch 2.6.0+cu124 (not 2.11 — server driver is too old for cu130).

```bash
# Install deps (already done)
curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --user
~/.local/bin/pip install --user -r requirements.txt
~/.local/bin/pip install --user torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

## 1. Preprocess

```bash
python3 -m data.preprocess --input_dir ~/Desktop/footage --output_dir data/processed
```
Outputs rotation-corrected, temporally-synced, resized stereo PNGs + `preview.png`.

## 2. Build cached disparity (one-time)

```bash
python3 -m data.build_disparity --processed_dir data/processed
```
Produces `data/processed/disparity/*.npy`. Required if `training.use_disparity_gt: true`.

## 3. (Optional) Camera calibration

```bash
python3 -m data.calibrate \
  --cam0 calib/cam0.mp4 --cam1 calib/cam1.mp4 \
  --rows 6 --cols 9 --square_mm 25 \
  --output data/processed/rectify.npz
```

## 4. Train

```bash
# Single GPU (use whichever has free VRAM)
CUDA_VISIBLE_DEVICES=0 python3 -m training.train --config configs/default.yaml

# Dry run (one batch, verify everything loads)
CUDA_VISIBLE_DEVICES=0 python3 -m training.train --config configs/default.yaml --dry_run

# Background training (survives SSH disconnect)
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python3 -m training.train --config configs/default.yaml" > train.log 2>&1 &

# Check progress
tail -5 train.log

# Multi-GPU DDP (if GPUs are free)
torchrun --nproc_per_node=4 -m training.train --config configs/default.yaml
```

### Current training config

- **Depth model**: MiDaS DPT_Hybrid (frozen)
- **Refiner**: ResNet18 (pretrained, stronger than U-Net with limited data)
- **Batch size**: 1 (constrained by ~5.5GB free VRAM on GPU 0)
- **Epochs**: 40 (~16 hours at ~6 it/s)
- **Losses**: L1 + VGG perceptual + SSIM + edge-aware smoothness + SGBM disparity supervision + stereo consistency
- **Temporal loss**: disabled for initial training (enable for fine-tuning pass)
- **Augmentations**: hflip, elastic deformation, color jitter (0.25), gaussian noise, random blur, JPEG artifacts

### Training strategy

1. Train 40 epochs without temporal loss (~16 hours)
2. Fine-tune ~10 epochs with `temporal: 0.1` to smooth inter-frame jitter

### External datasets (optional)

Drop any stereo dataset with `left/` + `right/` folders and add path to config:
```yaml
external_datasets:
  - "/path/to/flickr1024"
  - "/path/to/kitti_stereo"
```

## 5. Inference

```bash
python3 -m inference.infer --input my_video.mp4 --output stereo.mp4 \
  --checkpoint checkpoints/best.pt --color_transform true
```

Outputs side-by-side (left|right) video or separate files with `--format separate`.

## Config knobs

| Setting | Options | Notes |
|---------|---------|-------|
| `model.depth_model` | `dpt_hybrid`, `midas`, `zoe` | `dpt_hybrid` works on server; `depth_anything_v2_small` needs hub fix |
| `model.refiner_type` | `resnet18`, `unet` | `resnet18` recommended with limited data |
| `training.batch_size` | 1-8 | Limited by free GPU VRAM |
| `training.loss_weights.temporal` | 0.0-0.1 | 0.0 = off (faster), 0.1 = on (smoother video) |
| `training.augment.*` | various | elastic, noise, blur, jpeg all help with generalization |

## Known limitations

1. Dichromat matrix uses human-LMS as intermediate basis — substitute measured treeshrew cone fundamentals for research-grade output.
2. Depth Anything V2 hub loader is broken (no hubconf.py in repo) — using MiDaS DPT_Hybrid instead.
3. Rectification script exists but calibration data not yet captured.
4. No LPIPS in-loop yet (install `lpips` to add).
5. Disocclusion handling is learned (refinement net), not multi-plane/LDI.
