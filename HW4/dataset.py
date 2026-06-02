"""Dataset and augmentation utilities for HW4 rain/snow image restoration."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RestorationPair:
    degraded_path: Path
    clean_path: Path
    label: int
    filename: str


def _cfg(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _enabled(config: Dict[str, Any], *keys: str) -> bool:
    return bool(_cfg(config, *keys, "enabled", default=False))


def _prob(config: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    return float(_cfg(config, *keys, "p", default=default))


def _load_rgb(path: Path) -> Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _tensor_to_pil(x: Tensor) -> Image.Image:
    arr = (x.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(image: Image.Image) -> Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _jpeg_compress_tensor(x: Tensor, quality: int) -> Tensor:
    buffer = io.BytesIO()
    _tensor_to_pil(x.cpu()).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return _pil_to_tensor(Image.open(buffer))


def _pad_if_needed(x: Tensor, min_height: int, min_width: int) -> Tensor:
    _, height, width = x.shape
    pad_h = max(0, min_height - height)
    pad_w = max(0, min_width - width)
    if pad_h == 0 and pad_w == 0:
        return x
    x = x.unsqueeze(0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x.squeeze(0)


def _random_crop_pair(lq: Tensor, gt: Tensor, crop_size: int) -> Tuple[Tensor, Tensor]:
    lq = _pad_if_needed(lq, crop_size, crop_size)
    gt = _pad_if_needed(gt, crop_size, crop_size)
    _, height, width = lq.shape
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return (
        lq[:, top : top + crop_size, left : left + crop_size],
        gt[:, top : top + crop_size, left : left + crop_size],
    )


def _apply_geometric_pair(lq: Tensor, gt: Tensor, aug: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
    if _enabled(aug, "hflip") and random.random() < _prob(aug, "hflip", default=0.5):
        lq = torch.flip(lq, dims=[2])
        gt = torch.flip(gt, dims=[2])
    if _enabled(aug, "vflip") and random.random() < _prob(aug, "vflip", default=0.5):
        lq = torch.flip(lq, dims=[1])
        gt = torch.flip(gt, dims=[1])
    if _enabled(aug, "transpose") and random.random() < _prob(aug, "transpose", default=0.5):
        lq = lq.transpose(1, 2)
        gt = gt.transpose(1, 2)
    if _enabled(aug, "rot90") and random.random() < _prob(aug, "rot90", default=0.5):
        k = random.randint(1, 3)
        lq = torch.rot90(lq, k=k, dims=[1, 2])
        gt = torch.rot90(gt, k=k, dims=[1, 2])
    return lq.contiguous(), gt.contiguous()


def _adjust_contrast(x: Tensor, factor: float) -> Tensor:
    mean = x.mean(dim=(1, 2), keepdim=True)
    return (x - mean) * factor + mean


def _adjust_saturation(x: Tensor, factor: float) -> Tensor:
    gray = (0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]).expand_as(x)
    return (x - gray) * factor + gray


def _apply_pair_color_jitter(lq: Tensor, gt: Tensor, aug: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
    cfg = _cfg(aug, "paired_color_jitter", default={})
    if not cfg.get("enabled", False) or random.random() >= float(cfg.get("p", 0.0)):
        return lq, gt
    brightness = float(cfg.get("brightness", 0.0))
    contrast = float(cfg.get("contrast", 0.0))
    saturation = float(cfg.get("saturation", 0.0))

    if brightness > 0:
        factor = random.uniform(1.0 - brightness, 1.0 + brightness)
        lq = lq * factor
        gt = gt * factor
    if contrast > 0:
        factor = random.uniform(1.0 - contrast, 1.0 + contrast)
        lq = _adjust_contrast(lq, factor)
        gt = _adjust_contrast(gt, factor)
    if saturation > 0:
        factor = random.uniform(1.0 - saturation, 1.0 + saturation)
        lq = _adjust_saturation(lq, factor)
        gt = _adjust_saturation(gt, factor)
    return lq.clamp(0.0, 1.0), gt.clamp(0.0, 1.0)


def _apply_pair_gamma(lq: Tensor, gt: Tensor, aug: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
    cfg = _cfg(aug, "paired_gamma", default={})
    if not cfg.get("enabled", False) or random.random() >= float(cfg.get("p", 0.0)):
        return lq, gt
    gamma = random.uniform(float(cfg.get("gamma_min", 0.9)), float(cfg.get("gamma_max", 1.1)))
    return lq.clamp_min(1e-6).pow(gamma), gt.clamp_min(1e-6).pow(gamma)


def _apply_cutblur_clean_to_degraded(lq: Tensor, gt: Tensor, aug: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
    cfg = _cfg(aug, "cutblur_clean_to_degraded", default={})
    if not cfg.get("enabled", False) or random.random() >= float(cfg.get("p", 0.0)):
        return lq, gt
    _, height, width = lq.shape
    min_ratio = float(cfg.get("min_ratio", 0.15))
    max_ratio = float(cfg.get("max_ratio", 0.45))
    ratio = random.uniform(min_ratio, max_ratio)
    patch_h = max(1, int(height * ratio))
    patch_w = max(1, int(width * ratio))
    top = random.randint(0, height - patch_h)
    left = random.randint(0, width - patch_w)
    lq = lq.clone()
    lq[:, top : top + patch_h, left : left + patch_w] = gt[:, top : top + patch_h, left : left + patch_w]
    return lq, gt


def _apply_extra_lq_degradation(lq: Tensor, aug: Dict[str, Any]) -> Tensor:
    noise_cfg = _cfg(aug, "degraded_gaussian_noise", default={})
    if noise_cfg.get("enabled", False) and random.random() < float(noise_cfg.get("p", 0.0)):
        sigma = random.uniform(float(noise_cfg.get("sigma_min", 0.0)), float(noise_cfg.get("sigma_max", 0.01)))
        lq = (lq + torch.randn_like(lq) * sigma).clamp(0.0, 1.0)

    jpeg_cfg = _cfg(aug, "degraded_jpeg", default={})
    if jpeg_cfg.get("enabled", False) and random.random() < float(jpeg_cfg.get("p", 0.0)):
        quality = random.randint(int(jpeg_cfg.get("quality_min", 85)), int(jpeg_cfg.get("quality_max", 98)))
        lq = _jpeg_compress_tensor(lq, quality)
    return lq


def apply_pair_augmentations(lq: Tensor, gt: Tensor, aug: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
    if not aug.get("enabled", True):
        return lq, gt
    lq, gt = _apply_geometric_pair(lq, gt, aug)
    lq, gt = _apply_pair_color_jitter(lq, gt, aug)
    lq, gt = _apply_pair_gamma(lq, gt, aug)
    lq, gt = _apply_cutblur_clean_to_degraded(lq, gt, aug)
    lq = _apply_extra_lq_degradation(lq, aug)
    return lq.contiguous(), gt.contiguous()


def scan_train_pairs(root: str | Path) -> List[RestorationPair]:
    root = Path(root)
    degraded_dir = root / "train" / "degraded"
    clean_dir = root / "train" / "clean"
    if not degraded_dir.is_dir():
        raise FileNotFoundError(f"Missing degraded directory: {degraded_dir}")
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Missing clean directory: {clean_dir}")

    pairs: List[RestorationPair] = []
    for prefix, label in (("rain", 0), ("snow", 1)):
        for degraded_path in sorted(degraded_dir.glob(f"{prefix}-*.png")):
            number = degraded_path.stem.split("-")[-1]
            clean_path = clean_dir / f"{prefix}_clean-{number}.png"
            if not clean_path.is_file():
                raise FileNotFoundError(f"Missing clean image for {degraded_path.name}: {clean_path}")
            pairs.append(RestorationPair(degraded_path, clean_path, label, degraded_path.name))
    if not pairs:
        raise RuntimeError(f"No training pairs found under {root}")
    return pairs


def split_pairs(
    pairs: Sequence[RestorationPair],
    val_ratio: float = 0.05,
    seed: int = 3407,
) -> Tuple[List[RestorationPair], List[RestorationPair]]:
    rng = random.Random(seed)
    train_pairs: List[RestorationPair] = []
    val_pairs: List[RestorationPair] = []
    for label in (0, 1):
        group = [p for p in pairs if p.label == label]
        rng.shuffle(group)
        val_count = max(1, int(round(len(group) * val_ratio))) if val_ratio > 0 else 0
        val_pairs.extend(group[:val_count])
        train_pairs.extend(group[val_count:])
    rng.shuffle(train_pairs)
    val_pairs.sort(key=lambda p: p.filename)
    return train_pairs, val_pairs


class RainSnowTrainDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[RestorationPair],
        crop_size: int | None,
        augment_config: Dict[str, Any] | None,
        train: bool = True,
    ) -> None:
        self.pairs = list(pairs)
        self.crop_size = crop_size
        self.augment_config = augment_config or {}
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.pairs[index]
        degraded = _load_rgb(item.degraded_path)
        clean = _load_rgb(item.clean_path)
        if self.train and self.crop_size is not None and self.augment_config.get("random_crop", True):
            degraded, clean = _random_crop_pair(degraded, clean, self.crop_size)
        if self.train:
            degraded, clean = apply_pair_augmentations(degraded, clean, self.augment_config)
        return {
            "degraded": degraded,
            "clean": clean,
            "label": torch.tensor(item.label, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "filename": item.filename,
        }


class RainSnowTestDataset(Dataset):
    def __init__(self, root: str | Path) -> None:
        degraded_dir = Path(root) / "degraded"
        if not degraded_dir.is_dir():
            raise FileNotFoundError(f"Missing test degraded directory: {degraded_dir}")
        self.paths = sorted(
            degraded_dir.glob("*.png"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )
        if not self.paths:
            raise RuntimeError(f"No test images found under {degraded_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[index]
        return {
            "degraded": _load_rgb(path),
            "filename": path.name,
        }
