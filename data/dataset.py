"""PyTorch Dataset for stereo pairs + cached disparity + optional adjacent frames."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def _disp_to_tensor(d: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (disparity [1,H,W], valid_mask [1,H,W])."""
    valid = np.isfinite(d).astype(np.float32)
    d = np.nan_to_num(d, nan=0.0).astype(np.float32)
    return torch.from_numpy(d).unsqueeze(0), torch.from_numpy(valid).unsqueeze(0)


def _elastic_deform(l, r, l2, r2, disp, alpha=80, sigma=10):
    """Apply identical elastic deformation to all images in a stereo pair."""
    H, W = l.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(H, W).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(H, W).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    def _w(img):
        if img is None: return None
        return cv2.remap(img, map_x[:img.shape[0], :img.shape[1]],
                         map_y[:img.shape[0], :img.shape[1]],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    d_out = cv2.remap(disp, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT_101) if disp is not None else None
    return _w(l), _w(r), _w(l2), _w(r2), d_out


class ExternalStereoDataset(Dataset):
    """Loads stereo pairs from external datasets (KITTI, Middlebury, Flickr1024).

    Expected layout:
      root/left/  — left images (png or jpg)
      root/right/ — right images (png or jpg)
    """

    def __init__(self, root: str | Path, crop_size: tuple[int, int] = (384, 640)):
        root = Path(root)
        self.left_dir = root / "left"
        self.right_dir = root / "right"
        exts = ("*.png", "*.jpg", "*.jpeg")
        names = set()
        for ext in exts:
            names.update(p.stem for p in self.left_dir.glob(ext))
        self.names = sorted(names)
        if not self.names:
            raise FileNotFoundError(f"No images in {self.left_dir}")
        right_names = set()
        for ext in exts:
            right_names.update(p.stem for p in self.right_dir.glob(ext))
        self.names = [n for n in self.names if n in right_names]
        self.crop_size = crop_size

    def _find_file(self, directory: Path, stem: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg"):
            p = directory / (stem + ext)
            if p.exists():
                return p
        raise FileNotFoundError(f"No image for {stem} in {directory}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        stem = self.names[idx]
        l = cv2.imread(str(self._find_file(self.left_dir, stem)))
        r = cv2.imread(str(self._find_file(self.right_dir, stem)))
        ch, cw = self.crop_size
        l = cv2.resize(l, (cw, ch), interpolation=cv2.INTER_AREA)
        r = cv2.resize(r, (cw, ch), interpolation=cv2.INTER_AREA)
        if random.random() < 0.3:
            j = 0.2
            alpha = 1.0 + random.uniform(-j, j)
            beta = random.uniform(-j * 255, j * 255)
            l = np.clip(l.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            r = np.clip(r.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return {
            "left": _to_tensor(l),
            "right": _to_tensor(r),
            "name": stem,
            "has_neighbor": False,
            "has_disp": False,
        }


class StereoPairDataset(Dataset):
    """Serves aligned (left, right) pairs with optional disparity ground truth
    and an optional adjacent-in-time partner for temporal losses.

    Samples are only returned as "temporal" pairs when both frame t and t+1
    exist and belong to the same underlying session (session changes break
    temporal contiguity).
    """

    def __init__(
        self,
        processed_dir: str | Path,
        indices: list[int] | None = None,
        augment: dict | None = None,
        training: bool = True,
        use_disparity_gt: bool = True,
        use_temporal: bool = True,
    ):
        root = Path(processed_dir)
        self.left_dir = root / "left"
        self.right_dir = root / "right"
        self.disp_dir = root / "disparity"
        names = sorted(p.name for p in self.left_dir.glob("*.png"))
        if not names:
            raise FileNotFoundError(f"No frames found in {self.left_dir}")
        self.all_names = names  # for neighbor lookup
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        self.names = names if indices is None else [names[i] for i in indices]
        self.augment = augment or {}
        self.training = training
        self.use_disparity_gt = use_disparity_gt and self.disp_dir.exists()
        self.use_temporal = use_temporal

    def __len__(self) -> int:
        return len(self.names)

    def _load_bgr(self, name: str):
        l = cv2.imread(str(self.left_dir / name))
        r = cv2.imread(str(self.right_dir / name))
        if l is None or r is None:
            raise FileNotFoundError(name)
        return l, r

    def _load_disp(self, name: str) -> np.ndarray | None:
        p = self.disp_dir / (Path(name).stem + ".npy")
        if not p.exists():
            return None
        return np.load(p)

    def _apply_augment(self, l, r, l2=None, r2=None, disp=None):
        a = self.augment
        if self.training and a.get("hflip", False) and random.random() < 0.5:
            l = cv2.flip(l, 1); r = cv2.flip(r, 1)
            l, r = r, l
            if l2 is not None:
                l2 = cv2.flip(l2, 1); r2 = cv2.flip(r2, 1)
                l2, r2 = r2, l2
            disp = None
        if self.training and a.get("random_crop", False):
            ch, cw = a.get("crop_size", [l.shape[0], l.shape[1]])
            H, W = l.shape[:2]
            if ch < H and cw < W:
                y = random.randint(0, H - ch); x = random.randint(0, W - cw)
                def _c(img): return img[y:y+ch, x:x+cw] if img is not None else None
                l, r, l2, r2 = _c(l), _c(r), _c(l2), _c(r2)
                disp = disp[y:y+ch, x:x+cw] if disp is not None else None
        if self.training and a.get("elastic", False) and random.random() < 0.3:
            l, r, l2, r2, disp = _elastic_deform(l, r, l2, r2, disp,
                                                   alpha=a.get("elastic_alpha", 80),
                                                   sigma=a.get("elastic_sigma", 10))
        if self.training and a.get("color_jitter", 0.0) > 0:
            j = float(a["color_jitter"])
            alpha = 1.0 + random.uniform(-j, j); beta = random.uniform(-j * 255, j * 255)
            hue_shift = random.uniform(-j * 10, j * 10)
            def _j(img):
                if img is None: return None
                out = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                if abs(hue_shift) > 0.5:
                    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 0] = np.clip(hsv[:, :, 0].astype(np.float32) + hue_shift, 0, 179).astype(np.uint8)
                    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                return out
            l, r, l2, r2 = _j(l), _j(r), _j(l2), _j(r2)
        if self.training and a.get("gaussian_noise", 0.0) > 0 and random.random() < 0.4:
            sigma = random.uniform(1, a["gaussian_noise"])
            noise = np.random.randn(*l.shape).astype(np.float32) * sigma
            def _n(img):
                if img is None: return None
                return np.clip(img.astype(np.float32) + noise[:img.shape[0], :img.shape[1]], 0, 255).astype(np.uint8)
            l, r, l2, r2 = _n(l), _n(r), _n(l2), _n(r2)
        if self.training and a.get("random_blur", False) and random.random() < 0.2:
            k = random.choice([3, 5])
            def _b(img):
                if img is None: return None
                return cv2.GaussianBlur(img, (k, k), 0)
            l, r, l2, r2 = _b(l), _b(r), _b(l2), _b(r2)
        if self.training and a.get("jpeg_artifact", False) and random.random() < 0.2:
            quality = random.randint(30, 70)
            def _jp(img):
                if img is None: return None
                _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                return cv2.imdecode(buf, cv2.IMREAD_COLOR)
            l, r, l2, r2 = _jp(l), _jp(r), _jp(l2), _jp(r2)
        return l, r, l2, r2, disp

    def _neighbor_name(self, name: str) -> str | None:
        """Return the adjacent-frame name if it's sequentially next (same session)."""
        i = self.name_to_idx.get(name)
        if i is None or i + 1 >= len(self.all_names):
            return None
        # Preprocess writes consecutive session frames as consecutive 7-digit
        # indices, so i and i+1 are temporally adjacent within the same session.
        return self.all_names[i + 1]

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        l, r = self._load_bgr(name)
        l2 = r2 = None
        if self.use_temporal:
            nname = self._neighbor_name(name)
            if nname is not None:
                l2, r2 = self._load_bgr(nname)
        disp = self._load_disp(name) if self.use_disparity_gt else None

        l, r, l2, r2, disp = self._apply_augment(l, r, l2, r2, disp)

        item = {
            "left": _to_tensor(l),
            "right": _to_tensor(r),
            "name": name,
            "has_neighbor": l2 is not None,
        }
        if l2 is not None:
            item["left_next"] = _to_tensor(l2)
            item["right_next"] = _to_tensor(r2)
        item["has_disp"] = disp is not None
        if disp is not None:
            d, v = _disp_to_tensor(disp)
            item["disp_gt"] = d
            item["disp_valid"] = v
        return item


def build_splits(n: int, train_frac: float, seed: int = 42) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    k = int(round(train_frac * n))
    return idx[:k].tolist(), idx[k:].tolist()


def collate(batch: list[dict]) -> dict:
    """Custom collate: stacks tensors, tolerates missing temporal/disp entries
    by only stacking when all items in the batch have them."""
    out = {}
    keys = ["left", "right"]
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    out["name"] = [b["name"] for b in batch]
    if all(b["has_neighbor"] for b in batch):
        out["left_next"] = torch.stack([b["left_next"] for b in batch])
        out["right_next"] = torch.stack([b["right_next"] for b in batch])
    if all(b["has_disp"] for b in batch):
        out["disp_gt"] = torch.stack([b["disp_gt"] for b in batch])
        out["disp_valid"] = torch.stack([b["disp_valid"] for b in batch])
    return out
