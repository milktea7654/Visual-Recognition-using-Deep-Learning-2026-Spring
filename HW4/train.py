"""Train a configurable PromptIR-style model for HW4 rain/snow restoration."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # type: ignore[assignment]
from tqdm import tqdm

from dataset import RainSnowTrainDataset, scan_train_pairs, split_pairs


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = load_yaml(path)
    includes = cfg.pop("include", []) or []
    merged: Dict[str, Any] = {}
    for item in includes:
        include_path = Path(item)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        deep_update(merged, load_config(include_path))
    deep_update(merged, cfg)
    return merged


def set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur: Any = cfg
    for idx, key in enumerate(keys[:-1]):
        next_key = keys[idx + 1]
        if isinstance(cur, list):
            cur = cur[int(key)]
            continue
        if key not in cur or cur[key] is None:
            cur[key] = [] if next_key.isdigit() else {}
        cur = cur[key]
    last = keys[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HW4 image restoration model.")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data_root", type=str, default="", help="Override data.root.")
    parser.add_argument("--resume", type=str, default="", help="Override train.resume.")
    parser.add_argument("--run_name", type=str, default="", help="Override project.run_name.")
    parser.add_argument("--output_root", type=str, default="", help="Override project.output_root.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with dotted keys. Example: --set model.dim=64 --set train.epochs=400",
    )
    return parser.parse_args()


def prepare_config() -> Dict[str, Any]:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        set_by_dotted_key(cfg, "data.root", args.data_root)
    if args.resume:
        set_by_dotted_key(cfg, "train.resume", args.resume)
    if args.run_name:
        set_by_dotted_key(cfg, "project.run_name", args.run_name)
    if args.output_root:
        set_by_dotted_key(cfg, "project.output_root", args.output_root)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        set_by_dotted_key(cfg, key, value)
    return cfg


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    # Faster Tensor Core math for restoration training on supported NVIDIA GPUs.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def tensor_psnr(pred: Tensor, target: Tensor, eps: float = 1e-10) -> Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def average_logs(logs: Iterable[Dict[str, float]]) -> Dict[str, float]:
    logs = list(logs)
    if not logs:
        return {}
    keys = logs[0].keys()
    return {key: sum(item[key] for item in logs) / len(logs) for key in keys}


def format_logs(logs: Dict[str, float]) -> str:
    return " | ".join(f"{key}: {value:.5f}" for key, value in logs.items())


def create_run_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = mkdir(Path(cfg.get("project", {}).get("output_root", "output")) / timestamp)
    dirs = {
        "run": run_dir,
        "checkpoints": mkdir(run_dir / "checkpoints"),
        "logs": mkdir(run_dir / "logs"),
        "visuals": mkdir(run_dir / "visuals"),
    }
    return dirs


def tensor_to_image(tensor: Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8))


def add_title_bar(image: Image.Image, title: str) -> Image.Image:
    bar_height = 24
    canvas = Image.new("RGB", (image.width, image.height + bar_height), (245, 245, 245))
    canvas.paste(image, (0, bar_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(20, 20, 20))
    return canvas


def laplacian_response(x: Tensor) -> Tensor:
    kernel = x.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    weight = kernel.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, weight, padding=1, groups=x.shape[1])


def save_visual_batch(
    degraded: Tensor,
    pred: Tensor,
    clean: Tensor,
    path: Path,
    max_items: int = 4,
    aux: Dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = pred.clamp(0.0, 1.0)
    clean = clean.clamp(0.0, 1.0)
    coarse = None
    if aux is not None and "coarse_pred" in aux:
        coarse = aux["coarse_pred"].detach().clamp(0.0, 1.0)
    count = min(int(max_items), degraded.shape[0])
    if count <= 0:
        return

    psnrs = tensor_psnr(pred[:count], clean[:count]).detach().cpu()
    coarse_psnrs = tensor_psnr(coarse[:count], clean[:count]).detach().cpu() if coarse is not None else None
    rows: List[Image.Image] = []
    for idx in range(count):
        error = (pred[idx] - clean[idx]).abs().mul(6.0).clamp(0.0, 1.0)
        panels = [add_title_bar(tensor_to_image(degraded[idx]), "input")]
        if coarse is not None and coarse_psnrs is not None:
            coarse_error = (coarse[idx] - clean[idx]).abs().mul(6.0).clamp(0.0, 1.0)
            lap_coarse = (laplacian_response(coarse[idx : idx + 1]) - laplacian_response(clean[idx : idx + 1])).abs()
            lap_final = (laplacian_response(pred[idx : idx + 1]) - laplacian_response(clean[idx : idx + 1])).abs()
            panels.extend(
                [
                    add_title_bar(tensor_to_image(coarse[idx]), f"coarse {coarse_psnrs[idx].item():.2f}"),
                    add_title_bar(tensor_to_image(pred[idx]), f"final {psnrs[idx].item():.2f}"),
                    add_title_bar(tensor_to_image(clean[idx]), "target"),
                    add_title_bar(tensor_to_image(coarse_error), "err coarse x6"),
                    add_title_bar(tensor_to_image(error), "err final x6"),
                    add_title_bar(tensor_to_image(lap_coarse[0].mul(2.0)), "lap err coarse"),
                    add_title_bar(tensor_to_image(lap_final[0].mul(2.0)), "lap err final"),
                ]
            )
        else:
            panels.extend(
                [
                    add_title_bar(tensor_to_image(pred[idx]), f"pred {psnrs[idx].item():.2f} dB"),
                    add_title_bar(tensor_to_image(clean[idx]), "target"),
                    add_title_bar(tensor_to_image(error), "abs error x6"),
                ]
            )
        row_width = sum(panel.width for panel in panels)
        row_height = max(panel.height for panel in panels)
        row = Image.new("RGB", (row_width, row_height), (255, 255, 255))
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)

    gap = 10
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + gap
    sheet.save(path)


# -----------------------------------------------------------------------------
# Inference utilities used for validation
# -----------------------------------------------------------------------------


def pad_to_multiple(x: Tensor, multiple: int = 8) -> Tuple[Tensor, Tuple[int, int]]:
    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (height, width)


@torch.no_grad()
def simple_inference(model: nn.Module, x: Tensor, multiple: int = 8) -> Tensor:
    x_pad, (height, width) = pad_to_multiple(x, multiple)
    out = model(x_pad)
    return out[:, :, :height, :width]


def _build_weight(tile_h: int, tile_w: int, device: torch.device) -> Tensor:
    y = torch.hann_window(tile_h, periodic=False, device=device).clamp_min(1e-3)
    x = torch.hann_window(tile_w, periodic=False, device=device).clamp_min(1e-3)
    return torch.outer(y, x).view(1, 1, tile_h, tile_w)


@torch.no_grad()
def tiled_inference(model: nn.Module, x: Tensor, tile: int = 256, overlap: int = 48) -> Tensor:
    if tile <= 0:
        return simple_inference(model, x)
    _, _, height, width = x.shape
    if height <= tile and width <= tile:
        return simple_inference(model, x)

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile")

    out = torch.zeros_like(x)
    weight_sum = torch.zeros_like(x[:, :1])
    y_positions = list(range(0, max(height - tile, 0) + 1, stride))
    x_positions = list(range(0, max(width - tile, 0) + 1, stride))
    if y_positions[-1] != height - tile:
        y_positions.append(max(height - tile, 0))
    if x_positions[-1] != width - tile:
        x_positions.append(max(width - tile, 0))

    for top in y_positions:
        for left in x_positions:
            patch = x[:, :, top : top + tile, left : left + tile]
            patch_h, patch_w = patch.shape[-2:]
            pred = simple_inference(model, patch)
            weight = _build_weight(patch_h, patch_w, x.device)
            out[:, :, top : top + patch_h, left : left + patch_w] += pred * weight
            weight_sum[:, :, top : top + patch_h, left : left + patch_w] += weight
    return out / weight_sum.clamp_min(1e-8)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class EdgeLoss(nn.Module):
    """計算影像空間梯度的 L1 誤差，專門保護建築物與樹葉的輪廓"""

    def __init__(self) -> None:
        super().__init__()
        # 定義 Sobel 邊緣偵測卷積核
        k = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('weight_x', k.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('weight_y', k.t().view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_x = F.conv2d(pred, self.weight_x, padding=1, groups=3)
        pred_y = F.conv2d(pred, self.weight_y, padding=1, groups=3)
        target_x = F.conv2d(target, self.weight_x, padding=1, groups=3)
        target_y = F.conv2d(target, self.weight_y, padding=1, groups=3)
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


class LaplacianDetailLoss(nn.Module):
    """L1 loss on fixed RGB Laplacian responses for fine detail preservation."""

    def __init__(self) -> None:
        super().__init__()
        lap_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("weight", lap_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def laplacian(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight.to(dtype=x.dtype), padding=1, groups=3)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(self.laplacian(pred), self.laplacian(target))


class RainRegionLumaDetailLoss(nn.Module):
    """Luminance Laplacian loss focused on paired rain-corrupted regions."""

    def __init__(self) -> None:
        super().__init__()
        lap_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("weight", lap_kernel.view(1, 1, 3, 3))

    @staticmethod
    def luminance(x: Tensor) -> Tensor:
        if x.shape[1] == 1:
            return x
        coeffs = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x[:, :3] * coeffs).sum(dim=1, keepdim=True)

    def laplacian(self, x: Tensor) -> Tensor:
        return F.conv2d(self.luminance(x), self.weight.to(dtype=x.dtype), padding=1)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        degraded: Tensor,
        labels: Tensor,
        threshold: float = 0.055,
        sharpness: float = 45.0,
    ) -> Tensor:
        rain_weight = (labels == 0).to(dtype=pred.dtype).view(-1, 1, 1, 1)
        if float(rain_weight.sum().detach().cpu()) <= 0.0:
            return pred.new_tensor(0.0)
        paired_change = (degraded.detach() - target.detach()).abs().mean(dim=1, keepdim=True)
        region_weight = torch.sigmoid((paired_change - float(threshold)) * float(sharpness))
        weight = region_weight * rain_weight
        error = (self.laplacian(pred) - self.laplacian(target)).abs()
        return (error * weight).sum() / weight.sum().clamp_min(1e-6)


class FFTAmplitudeLoss(nn.Module):
    """Log-amplitude FFT L1 loss used as a light high-frequency stabilizer."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        return F.l1_loss(torch.log1p(pred_fft.abs()), torch.log1p(target_fft.abs()))


class SSIMLoss(nn.Module):
    """Small differentiable SSIM loss, returning 1 - SSIM."""

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)
        padding = self.window_size // 2
        mu_x = F.avg_pool2d(pred, self.window_size, stride=1, padding=padding)
        mu_y = F.avg_pool2d(target, self.window_size, stride=1, padding=padding)
        mu_x2 = mu_x.square()
        mu_y2 = mu_y.square()
        mu_xy = mu_x * mu_y
        sigma_x = F.avg_pool2d(pred.square(), self.window_size, stride=1, padding=padding) - mu_x2
        sigma_y = F.avg_pool2d(target.square(), self.window_size, stride=1, padding=padding) - mu_y2
        sigma_xy = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=padding) - mu_xy
        ssim = ((2.0 * mu_xy + self.c1) * (2.0 * sigma_xy + self.c2))
        ssim = ssim / ((mu_x2 + mu_y2 + self.c1) * (sigma_x + sigma_y + self.c2)).clamp_min(1e-8)
        return (1.0 - ssim).mean()


class HighFrequencyPhaseLoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    @staticmethod
    def _luminance(x: Tensor) -> Tensor:
        if x.shape[1] == 1:
            return x
        coeffs = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x[:, :3] * coeffs).sum(dim=1, keepdim=True)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        cutoff: float = 0.16,
        sharpness: float = 32.0,
        amp_power: float = 0.5,
    ) -> Tensor:
        pred_gray = self._luminance(pred).float()
        target_gray = self._luminance(target).float()
        _, _, height, width = pred_gray.shape

        fy = torch.fft.fftfreq(height, device=pred.device).view(height, 1)
        fx = torch.fft.rfftfreq(width, device=pred.device).view(1, width // 2 + 1)
        radius = torch.sqrt(fy.square() + fx.square())
        high_mask = torch.sigmoid((radius - cutoff) * sharpness).view(1, 1, height, width // 2 + 1)

        pred_fft = torch.fft.rfft2(pred_gray, norm="ortho")
        target_fft = torch.fft.rfft2(target_gray, norm="ortho")
        pred_phase = pred_fft / (pred_fft.abs() + self.eps)
        target_phase = target_fft / (target_fft.abs() + self.eps)

        amp_weight = target_fft.abs().clamp_min(self.eps).pow(amp_power)
        weight = high_mask * amp_weight
        phase_error = (pred_phase - target_phase).abs()
        return (phase_error * weight).sum() / weight.sum().clamp_min(self.eps)


class RestorationCriterion(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.phase_loss = HighFrequencyPhaseLoss()
        self.edge_loss = EdgeLoss()
        self.laplacian_detail_loss = LaplacianDetailLoss()
        self.rain_detail_loss = RainRegionLumaDetailLoss()
        self.fft_loss = FFTAmplitudeLoss()
        self.ssim_loss = SSIMLoss()
        self.current_epoch = 0
        self.total_epochs = 0
        self.configure(cfg)

    def configure(self, cfg: Dict[str, Any]) -> None:
        self.reconstruction = str(cfg.get("reconstruction", "charbonnier")).lower()
        anneal_cfg = cfg.get("anneal", {})
        self.anneal_enabled = bool(anneal_cfg.get("enabled", False))
        self.anneal_final_epochs = int(anneal_cfg.get("final_epochs", 0))
        self.anneal_reconstruction = str(anneal_cfg.get("reconstruction", "mse")).lower()
        self.anneal_auxiliary_weight_scale = float(anneal_cfg.get("auxiliary_weight_scale", 0.0))
        self.clamp_pred = bool(cfg.get("clamp_pred", False))
        self.weights = {
            "task": float(cfg.get("lambda_task", 0.05)),
            "edge": float(cfg.get("lambda_edge", 0.0)),
            "deep": float(cfg.get("lambda_deep_supervision", 0.0)),
            "stage1": float(cfg.get("lambda_stage1", 0.0)),
            "delta": float(cfg.get("lambda_delta", 0.0)),
            "chroma": float(cfg.get("lambda_chroma", 0.0)),
            "phase": float(cfg.get("lambda_phase", 0.0)),
            "identity": float(cfg.get("lambda_identity", 0.0)),
            "mask": float(cfg.get("lambda_mask", 0.0)),
            "laplacian_detail": float(cfg.get("lambda_laplacian_detail", 0.0)),
            "rain_detail": float(cfg.get("lambda_rain_detail", 0.0)),
            "rain_delta": float(cfg.get("lambda_rain_delta", 0.0)),
            "fft": float(cfg.get("lambda_fft", cfg.get("fft_weight", 0.0))),
            "ssim": float(cfg.get("lambda_ssim", cfg.get("ssim_weight", 0.0))),
            "detail_stage": float(cfg.get("lambda_detail_stage", 0.0)),
        }
        self.rain_reconstruction_weight = float(cfg.get("rain_reconstruction_weight", 1.0))
        self.rain_detail_threshold = float(cfg.get("rain_detail_threshold", 0.055))
        self.rain_detail_sharpness = float(cfg.get("rain_detail_sharpness", 45.0))
        # Paired-data priors for the routed residual/mask model.
        # identity: protect regions where degraded input already matches the clean target.
        # mask: supervise the predicted spatial corruption mask from paired residuals.
        self.identity_tau = float(cfg.get("identity_tau", 0.025))
        self.mask_target_scale = float(cfg.get("mask_target_scale", 0.08))
        self.mask_blur_kernel = int(cfg.get("mask_blur_kernel", 3))
        self.chroma_threshold = float(cfg.get("chroma_threshold", 0.04))
        self.chroma_sharpness = float(cfg.get("chroma_sharpness", 40.0))
        self.phase_cutoff = float(cfg.get("phase_cutoff", 0.16))
        self.phase_sharpness = float(cfg.get("phase_sharpness", 32.0))
        self.phase_amp_power = float(cfg.get("phase_amp_power", 0.5))
        self.deep_supervision_weights = [
            float(weight) for weight in cfg.get("deep_supervision_weights", [0.5, 0.25])
        ]

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def _in_anneal_stage(self) -> bool:
        if not self.anneal_enabled or self.anneal_final_epochs <= 0 or self.total_epochs <= 0:
            return False
        return self.current_epoch > self.total_epochs - self.anneal_final_epochs

    def _reconstruction_loss(self, pred: Tensor, target: Tensor, kind: str) -> Tensor:
        if kind in {"mse", "l2"}:
            return F.mse_loss(pred, target)
        if kind in {"l1", "mae"}:
            return F.l1_loss(pred, target)
        if kind in {"charbonnier", "charb"}:
            return self.charbonnier(pred, target)
        if kind in {"psnr", "psnrloss"}:
            mse = (pred.float() - target.float()).square().mean(dim=(1, 2, 3)).clamp_min(1e-10)
            return (10.0 * torch.log10(mse)).mean()
        raise ValueError(f"Unsupported reconstruction loss: {kind!r}")

    def _reconstruction_loss_per_sample(self, pred: Tensor, target: Tensor, kind: str) -> Tensor:
        diff = pred - target
        if kind in {"mse", "l2"}:
            return diff.square().mean(dim=(1, 2, 3))
        if kind in {"l1", "mae"}:
            return diff.abs().mean(dim=(1, 2, 3))
        if kind in {"charbonnier", "charb"}:
            return torch.sqrt(diff * diff + self.charbonnier.eps * self.charbonnier.eps).mean(dim=(1, 2, 3))
        if kind in {"psnr", "psnrloss"}:
            return 10.0 * torch.log10(diff.float().square().mean(dim=(1, 2, 3)).clamp_min(1e-10))
        raise ValueError(f"Unsupported reconstruction loss: {kind!r}")

    def _weighted_reconstruction_loss(
        self,
        pred: Tensor,
        target: Tensor,
        labels: Tensor,
        kind: str,
    ) -> Tensor:
        if abs(self.rain_reconstruction_weight - 1.0) < 1e-8:
            return self._reconstruction_loss(pred, target, kind)
        per_sample = self._reconstruction_loss_per_sample(pred, target, kind)
        sample_weights = torch.ones_like(per_sample)
        sample_weights = torch.where(
            labels == 0,
            sample_weights.new_full(sample_weights.shape, self.rain_reconstruction_weight),
            sample_weights,
        )
        return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)

    def _active_weights(self) -> Dict[str, float]:
        if not self._in_anneal_stage():
            return self.weights
        return {key: value * self.anneal_auxiliary_weight_scale for key, value in self.weights.items()}

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        labels: Tensor,
        aux: Dict[str, Any],
        degraded: Tensor | None = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        if self.clamp_pred:
            pred = pred.clamp(0.0, 1.0)
        anneal_stage = self._in_anneal_stage()
        recon_kind = self.anneal_reconstruction if anneal_stage else self.reconstruction
        weights = self._active_weights()
        recon = self._weighted_reconstruction_loss(pred, target, labels, recon_kind)
        loss = recon

        task = pred.new_tensor(0.0)
        deep = pred.new_tensor(0.0)
        stage1 = pred.new_tensor(0.0)
        delta = pred.new_tensor(0.0)
        chroma = pred.new_tensor(0.0)
        phase = pred.new_tensor(0.0)
        edge = pred.new_tensor(0.0)
        identity = pred.new_tensor(0.0)
        mask = pred.new_tensor(0.0)
        lap_detail = pred.new_tensor(0.0)
        rain_detail = pred.new_tensor(0.0)
        rain_delta = pred.new_tensor(0.0)
        fft = pred.new_tensor(0.0)
        ssim = pred.new_tensor(0.0)
        detail_stage = pred.new_tensor(0.0)
        prompt_aux = aux.get("prompt_aux") if isinstance(aux, dict) else None
        task_logits_items: List[Tensor] = []
        if prompt_aux:
            task_logits_items.extend(
                item["task_logits"]
                for item in prompt_aux
                if isinstance(item, dict) and "task_logits" in item
            )
        if isinstance(aux, dict) and "task_logits" in aux:
            task_logits_items.append(aux["task_logits"])
        if weights["task"] > 0 and task_logits_items:
            task_logits = torch.stack(task_logits_items, dim=0).mean(dim=0)
            task = F.cross_entropy(task_logits, labels)
            loss = loss + weights["task"] * task
        if weights["edge"] > 0:
            edge = self.edge_loss(pred, target)
            loss = loss + weights["edge"] * edge
        if weights["laplacian_detail"] > 0:
            lap_detail = self.laplacian_detail_loss(pred, target)
            loss = loss + weights["laplacian_detail"] * lap_detail
        if weights["rain_detail"] > 0 and degraded is not None:
            rain_detail = self.rain_detail_loss(
                pred,
                target,
                degraded,
                labels,
                threshold=self.rain_detail_threshold,
                sharpness=self.rain_detail_sharpness,
            )
            loss = loss + weights["rain_detail"] * rain_detail
        if weights["fft"] > 0:
            fft = self.fft_loss(pred, target)
            loss = loss + weights["fft"] * fft
        if weights["ssim"] > 0:
            ssim = self.ssim_loss(pred, target)
            loss = loss + weights["ssim"] * ssim
        if weights["deep"] > 0 and isinstance(aux, dict):
            deep_items: List[Tensor] = []
            for idx, key in enumerate(("pred_level2", "pred_level3")):
                if key not in aux:
                    continue
                aux_pred = aux[key]
                aux_target = F.interpolate(
                    target,
                    size=aux_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                weight = (
                    self.deep_supervision_weights[idx]
                    if idx < len(self.deep_supervision_weights)
                    else 1.0
                )
                deep_items.append(
                    aux_pred.new_tensor(weight)
                    * self._reconstruction_loss(aux_pred, aux_target, recon_kind)
                )
            if deep_items:
                deep = torch.stack(deep_items).sum()
                loss = loss + weights["deep"] * deep
        if weights["stage1"] > 0 and isinstance(aux, dict) and "pred_stage1" in aux:
            stage1 = self._reconstruction_loss(aux["pred_stage1"], target, recon_kind)
            loss = loss + weights["stage1"] * stage1
        if weights["detail_stage"] > 0 and isinstance(aux, dict) and "detail_stage_pred" in aux:
            detail_stage = self._reconstruction_loss(aux["detail_stage_pred"], target, recon_kind)
            loss = loss + weights["detail_stage"] * detail_stage
        if (
            weights["delta"] > 0
            and isinstance(aux, dict)
            and "pred_stage1" in aux
            and "delta" in aux
        ):
            delta_target = target - aux["pred_stage1"].detach()
            delta = self._reconstruction_loss(aux["delta"], delta_target, recon_kind)
            loss = loss + weights["delta"] * delta
        if (
            weights["rain_delta"] > 0
            and isinstance(aux, dict)
            and "rain_delta_pred" in aux
            and "coarse_pred" in aux
        ):
            rain_mask = (labels == 0).to(dtype=pred.dtype).view(-1, 1, 1, 1)
            if rain_mask.sum() > 0:
                delta_target = target - aux["coarse_pred"].detach()
                delta_error = (aux["rain_delta_pred"] - delta_target).abs()
                rain_delta = (delta_error * rain_mask).sum()
                rain_delta = rain_delta / rain_mask.expand_as(delta_error).sum().clamp_min(1.0)
                loss = loss + weights["rain_delta"] * rain_delta
        if weights["chroma"] > 0 and degraded is not None:
            degradation = (degraded.detach() - target.detach()).abs().mean(dim=1, keepdim=True)
            mask = torch.sigmoid((degradation - self.chroma_threshold) * self.chroma_sharpness)
            residual = pred - target
            residual_chroma = residual - residual.mean(dim=1, keepdim=True)
            chroma = (residual_chroma.abs() * mask).sum()
            chroma = chroma / mask.expand_as(residual_chroma).sum().clamp_min(1.0)
            loss = loss + weights["chroma"] * chroma
        if degraded is not None and weights["identity"] > 0:
            paired_change = (degraded.detach() - target.detach()).abs().mean(dim=1, keepdim=True)
            clean_weight = torch.exp(-paired_change / max(self.identity_tau, 1e-6))
            identity = ((pred - degraded).abs() * clean_weight).sum()
            identity = identity / clean_weight.expand_as(pred).sum().clamp_min(1.0)
            loss = loss + weights["identity"] * identity
        if degraded is not None and weights["mask"] > 0 and isinstance(aux, dict):
            paired_change = (degraded.detach() - target.detach()).abs().mean(dim=1, keepdim=True)
            mask_target = (paired_change / max(self.mask_target_scale, 1e-6)).clamp(0.0, 1.0)
            if self.mask_blur_kernel > 1:
                kernel = self.mask_blur_kernel if self.mask_blur_kernel % 2 == 1 else self.mask_blur_kernel + 1
                mask_target = F.avg_pool2d(mask_target, kernel_size=kernel, stride=1, padding=kernel // 2)
            if "corruption_mask_logits" in aux:
                mask = F.binary_cross_entropy_with_logits(aux["corruption_mask_logits"], mask_target)
            elif "corruption_mask" in aux:
                # Probability-space BCE is not autocast-safe. This fallback only exists for
                # older model variants that expose a sigmoid mask rather than mask logits.
                with autocast(pred.device.type, enabled=False):
                    predicted_mask = aux["corruption_mask"].float().clamp(1e-4, 1.0 - 1e-4)
                    mask = F.binary_cross_entropy(predicted_mask, mask_target.float())
            if mask.ndim == 0 and mask.numel() == 1:
                loss = loss + weights["mask"] * mask
        if weights["phase"] > 0:
            phase = self.phase_loss(
                pred,
                target,
                cutoff=self.phase_cutoff,
                sharpness=self.phase_sharpness,
                amp_power=self.phase_amp_power,
            )
            loss = loss + weights["phase"] * phase

        logs = {
            "loss": float(loss.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "edge": float(edge.detach().cpu()),
            "task": float(task.detach().cpu()),
            "deep": float(deep.detach().cpu()),
            "stage1": float(stage1.detach().cpu()),
            "delta": float(delta.detach().cpu()),
            "chroma": float(chroma.detach().cpu()),
            "identity": float(identity.detach().cpu()),
            "mask": float(mask.detach().cpu()),
            "lap_detail": float(lap_detail.detach().cpu()),
            "rain_detail": float(rain_detail.detach().cpu()),
            "rain_delta": float(rain_delta.detach().cpu()),
            "fft": float(fft.detach().cpu()),
            "ssim": float(ssim.detach().cpu()),
            "detail_stage": float(detail_stage.detach().cpu()),
            "phase": float(phase.detach().cpu()),
            "mse_stage": 1.0 if anneal_stage or recon_kind in {"mse", "l2"} else 0.0,
        }
        return loss, logs


# -----------------------------------------------------------------------------
# Model / data / checkpoint
# -----------------------------------------------------------------------------


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> Dict[str, Tensor]:
        return self.ema.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        self.ema.load_state_dict(state_dict)


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    module = importlib.import_module(f"model.{model_cfg['name']}")
    return module.build_model(model_cfg)


def model_forward(
    model: nn.Module,
    degraded: Tensor,
    labels: Tensor | None = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    try:
        out = model(degraded, return_aux=True, task_labels=labels)
    except TypeError:
        try:
            out = model(degraded, return_aux=True)
        except TypeError:
            return model(degraded), {}
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, {}


def load_state_dict_compatible(model: nn.Module, state_dict: Dict[str, Tensor]) -> Tuple[List[str], List[str]]:
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    incompatible = [
        key
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape != value.shape
    ]
    missing = [key for key in model_state.keys() if key not in compatible]
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=True)
    return missing, incompatible


def parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(param.numel() for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def has_allowed_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def set_detail_probe_trainability(model: nn.Module, detail_prefixes: Sequence[str], freeze_backbone: bool) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_(not freeze_backbone or has_allowed_prefix(name, detail_prefixes))


def initialize_model_from_checkpoint(
    model: nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any] | None:
    init_cfg = cfg.get("initialization", {}) or {}
    checkpoint_path = str(init_cfg.get("checkpoint", "") or "")
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"initialization.checkpoint does not exist: {checkpoint_path}. "
            "Replace the placeholder with the existing PromptIR-DeepMSFN best.pt path."
        )

    checkpoint = torch.load(path, map_location=device)
    use_ema = bool(init_cfg.get("use_ema", True))
    source_key = "ema" if use_ema and "ema" in checkpoint else "model"
    source_state = checkpoint[source_key]
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    incompatible = [
        key
        for key, value in source_state.items()
        if key in model_state and model_state[key].shape != value.shape
    ]
    missing = [key for key in model_state.keys() if key not in compatible]

    detail_prefixes = tuple(cfg.get("train", {}).get("detail_trainable_prefixes", [])) or (
        "detail_stem",
        "detail_blocks",
        "detail_sam",
        "detail_head",
    )
    allowed_new_prefixes = tuple(detail_prefixes) + ("laplacian_kernel",)
    unexpected_missing = [key for key in missing if not has_allowed_prefix(key, allowed_new_prefixes)]
    unexpected_incompatible = [key for key in incompatible if not has_allowed_prefix(key, allowed_new_prefixes)]
    if unexpected_missing or unexpected_incompatible:
        raise RuntimeError(
            "Initialization checkpoint is not compatible with the DeepMSFN backbone. "
            f"unexpected missing keys: {unexpected_missing[:10]}, "
            f"unexpected shape-mismatched keys: {unexpected_incompatible[:10]}"
        )

    model_state.update(compatible)
    model.load_state_dict(model_state, strict=True)

    max_abs_diff = None
    if bool(init_cfg.get("verify_equivalence", True)):
        baseline_cfg = deepcopy(checkpoint.get("config", cfg))
        if "model" in baseline_cfg:
            baseline_cfg["model"] = deepcopy(baseline_cfg["model"])
            baseline_cfg["model"]["name"] = "promptir_deep_msfn"
        baseline_model = build_model(baseline_cfg).to(device).eval()
        baseline_model.load_state_dict(source_state, strict=True)
        model.eval()
        with torch.no_grad():
            sample = torch.rand(1, 3, 128, 128, device=device)
            baseline_pred = baseline_model(sample)
            detailsam_pred = model(sample)
            max_abs_diff = float((baseline_pred - detailsam_pred).abs().max().item())
        del baseline_model
        if max_abs_diff >= 1e-6:
            raise RuntimeError(
                "DetailSAM zero-initialization equivalence check failed: "
                f"max_abs_diff={max_abs_diff:.8g} >= 1e-6"
            )

    print("Initialized PromptIR-DeepMSFN-DetailSAM from baseline checkpoint:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  source weights: {source_key}")
    print(f"  loaded keys: {len(compatible)}")
    print(f"  new DetailSAM keys: {len(missing)}")
    print(f"  shape-mismatched skipped keys: {len(incompatible)}")
    if max_abs_diff is not None:
        print(f"  baseline-equivalence max_abs_diff: {max_abs_diff:.8g}")
    return {
        "loaded": len(compatible),
        "missing": missing,
        "incompatible": incompatible,
        "max_abs_diff": max_abs_diff,
    }


def resolve_train_stage(cfg: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    stages = cfg.get("train", {}).get("progressive_stages", []) or []
    if not stages:
        return {}
    sorted_stages = sorted(stages, key=lambda item: int(item.get("until_epoch", 0)))
    for stage in sorted_stages:
        if epoch <= int(stage.get("until_epoch", 0)):
            return dict(stage)
    return dict(sorted_stages[-1])


def stage_data_config(cfg: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = deepcopy(cfg["data"])
    stage_data = stage.get("data", {})
    if isinstance(stage_data, dict) and stage_data:
        deep_update(data_cfg, stage_data)
    return data_cfg


def stage_loss_config(cfg: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, Any]:
    loss_cfg = deepcopy(cfg.get("loss", {}))
    stage_loss = stage.get("loss", {})
    if isinstance(stage_loss, dict) and stage_loss:
        deep_update(loss_cfg, stage_loss)
    return loss_cfg


def train_loader_signature(cfg: Dict[str, Any], stage: Dict[str, Any]) -> Tuple[int, int, str]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    crop_size = int(stage.get("crop_size", data_cfg.get("crop_size", 256)))
    batch_size = int(stage.get("batch_size", train_cfg.get("batch_size", 4)))
    stage_data_sig = yaml.safe_dump(stage.get("data", {}) or {}, sort_keys=True)
    return crop_size, batch_size, stage_data_sig


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def apply_warmup_lr(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], epoch: int) -> bool:
    train_cfg = cfg.get("train", {})
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0) or 0)
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return False
    base_lr = float(train_cfg.get("lr", 2e-4))
    warmup_lr = float(train_cfg.get("warmup_lr", base_lr))
    if warmup_epochs == 1:
        lr = base_lr
    else:
        progress = float(epoch - 1) / float(warmup_epochs - 1)
        lr = warmup_lr + (base_lr - warmup_lr) * progress
    set_optimizer_lr(optimizer, lr)
    return True


class LossAwareHardMining:
    """Maintain per-image error estimates for loss-aware resampling.

    The sampler only reuses the supplied paired training images.  During early
    epochs it records per-sample MSE from randomly cropped patches.  In later
    stages, high-error source images are sampled more often while preserving
    rain/snow balance and retaining a uniform component for generalization.
    """

    def __init__(self, size: int, momentum: float = 0.85) -> None:
        self.values = torch.ones(size, dtype=torch.float32)
        self.seen = torch.zeros(size, dtype=torch.bool)
        self.momentum = float(momentum)

    @torch.no_grad()
    def update(self, indices: Tensor, losses: Tensor) -> None:
        indices = indices.detach().cpu().long().flatten()
        losses = losses.detach().cpu().float().flatten().clamp_min(1e-8)
        for idx, loss in zip(indices.tolist(), losses.tolist()):
            if self.seen[idx]:
                self.values[idx] = self.momentum * self.values[idx] + (1.0 - self.momentum) * loss
            else:
                self.values[idx] = loss
                self.seen[idx] = True

    def sampling_weights(self, train_pairs: Sequence[Any], hard_cfg: Dict[str, Any]) -> Tensor:
        uniform_mix = float(hard_cfg.get("uniform_mix", 0.35))
        power = float(hard_cfg.get("power", 1.5))
        clip_ratio = float(hard_cfg.get("clip_ratio", 4.0))
        weights = torch.ones(len(train_pairs), dtype=torch.float64)
        labels = sorted({int(item.label) for item in train_pairs})
        for label in labels:
            group = torch.tensor(
                [
                    i
                    for i, item in enumerate(train_pairs)
                    if int(item.label) == label
                ],
                dtype=torch.long,
            )
            scores = self.values[group].clone()
            if (~self.seen[group]).any():
                observed = scores[self.seen[group]]
                fill = observed.median() if observed.numel() else torch.tensor(1.0)
                scores[~self.seen[group]] = fill
            median = scores.median().clamp_min(1e-8)
            relative = (scores / median).clamp(1.0 / clip_ratio, clip_ratio).pow(power)
            relative = uniform_mix * torch.ones_like(relative) + (1.0 - uniform_mix) * relative
            relative = relative / relative.sum().clamp_min(1e-8)
            weights[group] = relative.double() / max(len(labels), 1)
        return weights


def compute_pair_severities(train_pairs: Sequence[Any]) -> Tensor:
    """Compute paired degradation severity from the existing train pairs."""
    severities: List[float] = []
    for item in tqdm(train_pairs, desc="severity", leave=False):
        degraded = np.asarray(Image.open(item.degraded_path).convert("RGB"), dtype=np.float32) / 255.0
        clean = np.asarray(Image.open(item.clean_path).convert("RGB"), dtype=np.float32) / 255.0
        severities.append(float(np.mean(np.abs(degraded - clean))))
    return torch.tensor(severities, dtype=torch.float32).clamp_min(1e-8)


def severity_sampling_weights(
    train_pairs: Sequence[Any],
    severities: Tensor,
    cfg: Dict[str, Any],
) -> Tensor:
    """Oversample severe rain images while keeping some class balance."""
    uniform_mix = float(cfg.get("uniform_mix", 0.35))
    power = float(cfg.get("power", 1.4))
    clip_ratio = float(cfg.get("clip_ratio", 4.0))
    rain_class_weight = float(cfg.get("rain_class_weight", 1.35))
    snow_class_weight = float(cfg.get("snow_class_weight", 1.0))
    weights = torch.ones(len(train_pairs), dtype=torch.float64)
    labels = torch.tensor([int(item.label) for item in train_pairs], dtype=torch.long)
    class_mass = {
        0: rain_class_weight / max(rain_class_weight + snow_class_weight, 1e-8),
        1: snow_class_weight / max(rain_class_weight + snow_class_weight, 1e-8),
    }
    for label in (0, 1):
        group = torch.where(labels == label)[0]
        if group.numel() == 0:
            continue
        if label == 0:
            scores = severities[group].float()
            median = scores.median().clamp_min(1e-8)
            relative = (scores / median).clamp(1.0 / clip_ratio, clip_ratio).pow(power)
            relative = uniform_mix * torch.ones_like(relative) + (1.0 - uniform_mix) * relative
        else:
            relative = torch.ones(group.numel(), dtype=torch.float32)
        relative = relative / relative.sum().clamp_min(1e-8)
        weights[group] = relative.double() * class_mass[label]
    return weights


def build_train_loader(
    train_pairs: Sequence[Any],
    data_cfg: Dict[str, Any],
    crop_size: int,
    batch_size: int,
    sampling_weights: Tensor | None = None,
) -> DataLoader:
    train_set = RainSnowTrainDataset(
        train_pairs,
        crop_size=crop_size,
        augment_config=data_cfg.get("augment", {}),
        train=True,
    )
    num_workers = int(data_cfg.get("num_workers", 4))
    samples_per_epoch = int(data_cfg.get("samples_per_epoch", 0) or 0)
    sampler = None
    shuffle = True
    num_samples = samples_per_epoch if samples_per_epoch > 0 else len(train_set)
    if sampling_weights is not None:
        sampler = WeightedRandomSampler(sampling_weights, num_samples=num_samples, replacement=True)
        shuffle = False
    elif samples_per_epoch > 0:
        sampler = RandomSampler(train_set, replacement=True, num_samples=samples_per_epoch)
        shuffle = False
    loader_kwargs = {
        "dataset": train_set,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "drop_last": True,
        "persistent_workers": bool(data_cfg.get("persistent_workers", True)) and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    return DataLoader(**loader_kwargs)


def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, List[Any], Tuple[int, int, str]]:
    seed = int(cfg.get("project", {}).get("seed", 3407))
    data_cfg = cfg["data"]
    pairs = scan_train_pairs(data_cfg["root"])
    train_pairs, val_pairs = split_pairs(pairs, val_ratio=float(data_cfg.get("val_ratio", 0.05)), seed=seed)

    initial_stage = resolve_train_stage(cfg, 1)
    loader_sig = train_loader_signature(cfg, initial_stage)
    initial_data_cfg = stage_data_config(cfg, initial_stage)
    train_loader = build_train_loader(
        train_pairs,
        initial_data_cfg,
        crop_size=loader_sig[0],
        batch_size=loader_sig[1],
    )
    val_set = RainSnowTrainDataset(val_pairs, crop_size=None, augment_config={}, train=False)

    num_workers = int(data_cfg.get("num_workers", 4))
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    print(
        f"Train images: {len(train_pairs)} | Validation images: {len(val_set)} | "
        f"initial crop/batch: {loader_sig[0]}/{loader_sig[1]}"
    )
    return train_loader, val_loader, train_pairs, loader_sig


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_psnr: float,
    cfg: Dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
            "config": cfg,
        },
        path,
    )


def save_recent_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_psnr: float,
    cfg: Dict[str, Any],
    keep: int,
) -> None:
    if keep <= 0:
        return
    save_checkpoint(
        checkpoint_dir / f"recent_epoch_{epoch:04d}.pt",
        model,
        ema,
        optimizer,
        scheduler,
        epoch,
        best_psnr,
        cfg,
    )
    recent = sorted(checkpoint_dir.glob("recent_epoch_*.pt"))
    for old_path in recent[:-keep]:
        old_path.unlink(missing_ok=True)


def same_label_mixup(
    degraded: Tensor,
    clean: Tensor,
    labels: Tensor,
    aug_cfg: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor]:
    cfg = aug_cfg.get("same_label_mixup", {})
    if not cfg.get("enabled", False) or random.random() >= float(cfg.get("p", 0.0)):
        return degraded, clean, labels
    alpha = float(cfg.get("alpha", 0.2))
    if alpha <= 0:
        return degraded, clean, labels
    lam = float(np.random.beta(alpha, alpha))
    indices = torch.arange(labels.shape[0], device=labels.device)
    for label in labels.unique():
        mask = torch.where(labels == label)[0]
        if mask.numel() > 1:
            perm = mask[torch.randperm(mask.numel(), device=labels.device)]
            indices[mask] = perm
    degraded = lam * degraded + (1.0 - lam) * degraded[indices]
    clean = lam * clean + (1.0 - lam) * clean[indices]
    return degraded, clean, labels


def degradation_copy_paste(
    degraded: Tensor,
    clean: Tensor,
    labels: Tensor,
    aug_cfg: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor]:
    cfg = aug_cfg.get("degradation_copy_paste", {})
    if not cfg.get("enabled", False) or random.random() >= float(cfg.get("p", 0.0)):
        return degraded, clean, labels

    same_label = bool(cfg.get("same_label", True))
    scale_min = float(cfg.get("scale_min", 0.8))
    scale_max = float(cfg.get("scale_max", 1.2))
    batch = degraded.shape[0]
    if batch <= 1:
        return degraded, clean, labels

    out = degraded.clone()
    for idx in range(batch):
        if random.random() >= float(cfg.get("per_sample_p", 1.0)):
            continue
        if same_label:
            candidates = torch.where(labels == labels[idx])[0]
            candidates = candidates[candidates != idx]
        else:
            candidates = torch.arange(batch, device=labels.device)
            candidates = candidates[candidates != idx]
        if candidates.numel() == 0:
            continue
        src = candidates[torch.randint(candidates.numel(), (1,), device=labels.device)].item()
        noise = degraded[src] - clean[src]
        if bool(cfg.get("remove_mean", True)):
            noise = noise - noise.mean(dim=(1, 2), keepdim=True)
        max_abs = float(cfg.get("max_abs", 0.35) or 0.0)
        if max_abs > 0:
            noise = noise.clamp(-max_abs, max_abs)
        scale = random.uniform(scale_min, scale_max)
        out[idx] = (clean[idx] + noise * scale).clamp(0.0, 1.0)
    return out, clean, labels


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    writer: Any,
    epoch: int,
    visuals_dir: Path | None = None,
    visual_name: str = "ema",
) -> Dict[str, float]:
    model.eval()
    psnrs: List[float] = []
    rain_psnrs: List[float] = []
    snow_psnrs: List[float] = []
    laplacian_l1s: List[float] = []
    validation_cfg = cfg.get("validation", {})
    tile = int(validation_cfg.get("tile", 256))
    overlap = int(validation_cfg.get("overlap", 48))
    log_images_every = int(validation_cfg.get("log_images_every", 0))
    max_images = int(validation_cfg.get("max_images", 0) or 0)
    save_visuals = bool(validation_cfg.get("save_visuals", True))
    visuals_every = int(validation_cfg.get("visuals_every", 1))
    visuals_max_items = int(validation_cfg.get("visuals_max_items", 4))
    logged_image = False
    saved_visual = False

    for index, batch in enumerate(tqdm(val_loader, desc="val", leave=False), start=1):
        degraded = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        labels = batch.get("label")
        labels = labels.to(device, non_blocking=True) if labels is not None else None
        pred = tiled_inference(model, degraded, tile=tile, overlap=overlap)
        batch_psnr = tensor_psnr(pred, clean)
        psnrs.extend(batch_psnr.detach().cpu().tolist())
        laplacian_l1s.append(F.l1_loss(laplacian_response(pred), laplacian_response(clean)).item())
        if labels is not None:
            rain_mask = labels == 0
            snow_mask = labels == 1
            if rain_mask.any():
                rain_psnrs.extend(batch_psnr[rain_mask].detach().cpu().tolist())
            if snow_mask.any():
                snow_psnrs.extend(batch_psnr[snow_mask].detach().cpu().tolist())
        if writer is not None and log_images_every > 0 and epoch % log_images_every == 0 and not logged_image:
            grid = torch.cat([degraded[0], pred[0].clamp(0, 1), clean[0]], dim=2)
            writer.add_image("val/degraded_pred_clean", grid, epoch)
            logged_image = True
        if (
            save_visuals
            and visuals_dir is not None
            and visuals_every > 0
            and epoch % visuals_every == 0
            and not saved_visual
        ):
            visual_pred, visual_aux = model_forward(model, degraded, labels)
            save_visual_batch(
                degraded,
                visual_pred,
                clean,
                visuals_dir / f"epoch_{epoch:04d}_{visual_name}.png",
                max_items=visuals_max_items,
                aux=visual_aux,
            )
            saved_visual = True
        if max_images > 0 and index >= max_images:
            break
    overall = sum(psnrs) / len(psnrs)
    metrics = {
        "overall": overall,
        "rain": sum(rain_psnrs) / len(rain_psnrs) if rain_psnrs else overall,
        "snow": sum(snow_psnrs) / len(snow_psnrs) if snow_psnrs else overall,
        "laplacian_l1": sum(laplacian_l1s) / len(laplacian_l1s) if laplacian_l1s else 0.0,
    }
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    cfg = prepare_config()
    seed_everything(int(cfg.get("project", {}).get("seed", 3407)))
    dirs = create_run_dirs(cfg)
    save_yaml(cfg, dirs["run"] / "config_resolved.yaml")
    save_json(cfg, dirs["run"] / "config_resolved.json")
    print(f"Run directory: {dirs['run']}")

    writer = None
    if cfg.get("logging", {}).get("tensorboard", True):
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard is not installed. Run: pip install tensorboard")
        writer = SummaryWriter(log_dir=str(dirs["logs"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_pairs, active_loader_sig = build_loaders(cfg)
    model = build_model(cfg).to(device)
    print(f"Model: {cfg['model']['name']} | parameters: {count_parameters(model):.2f}M")
    initialize_model_from_checkpoint(model, cfg, device)

    criterion = RestorationCriterion(cfg.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 2e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["train"].get("epochs", 300)),
        eta_min=float(cfg["train"].get("min_lr", 1e-6)),
    )
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=amp_enabled)
    ema = ModelEMA(model, decay=float(cfg["train"].get("ema_decay", 0.999)))
    hard_miner = LossAwareHardMining(
        len(train_pairs),
        momentum=float(cfg.get("hard_mining", {}).get("momentum", 0.85)),
    )
    severity_scores: Tensor | None = None

    start_epoch = 1
    best_psnr = 0.0
    resume = str(cfg["train"].get("resume", "") or "")
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        strict_resume = bool(cfg["train"].get("strict_resume", True))
        if strict_resume:
            model.load_state_dict(checkpoint["model"], strict=True)
            if "ema" in checkpoint:
                ema.load_state_dict(checkpoint["ema"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_psnr = float(checkpoint.get("best_psnr", 0.0))
            print(f"Resumed from {resume}; next epoch = {start_epoch}")
        else:
            resume_weight = str(cfg["train"].get("resume_weight", "model") or "model")
            if resume_weight == "ema" and "ema" in checkpoint:
                source_state = checkpoint["ema"]
            else:
                source_state = checkpoint["model"]
                resume_weight = "model"
            missing, incompatible = load_state_dict_compatible(model, source_state)
            ema = ModelEMA(model, decay=float(cfg["train"].get("ema_decay", 0.999)))
            print(
                "Initialized compatible checkpoint weights; "
                f"source={resume_weight}, "
                f"missing/new keys: {len(missing)}, shape-mismatched skipped: {len(incompatible)}"
            )

    epochs = int(cfg["train"].get("epochs", 300))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    print_every = int(cfg.get("logging", {}).get("print_every", 20))
    val_every = int(cfg.get("validation", {}).get("every", 1))
    ckpt_cfg = cfg.get("checkpoint", {})

    global_step = 0
    freeze_state: str | None = None
    for epoch in range(start_epoch, epochs + 1):
        freeze_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0) or 0)
        detail_prefixes = tuple(cfg["train"].get("detail_trainable_prefixes", [])) or (
            "detail_stem",
            "detail_blocks",
            "detail_sam",
            "detail_head",
        )
        next_freeze_state = "detail" if epoch <= freeze_epochs else "all"
        if next_freeze_state != freeze_state:
            set_detail_probe_trainability(model, detail_prefixes, freeze_backbone=(next_freeze_state == "detail"))
            freeze_state = next_freeze_state
            trainable = parameter_count(model, trainable_only=True)
            total = parameter_count(model, trainable_only=False)
            if freeze_state == "detail":
                print(
                    f"Epoch {epoch}: backbone frozen; training DetailSAM only "
                    f"({trainable / 1e6:.2f}M / {total / 1e6:.2f}M trainable params)"
                )
            elif freeze_epochs > 0:
                print(
                    f"Epoch {epoch}: backbone unfrozen; joint fine-tuning begins "
                    f"({trainable / 1e6:.2f}M / {total / 1e6:.2f}M trainable params)"
                )
        stage = resolve_train_stage(cfg, epoch)
        data_cfg = stage_data_config(cfg, stage)
        criterion.configure(stage_loss_config(cfg, stage))
        stage_loader_sig = train_loader_signature(cfg, stage)
        hard_cfg = stage.get("hard_mining", {}) or {}
        hard_enabled = bool(hard_cfg.get("enabled", False))
        severity_cfg = stage.get("severity_mining", {}) or {}
        severity_enabled = bool(severity_cfg.get("enabled", False))
        sampling_weights = hard_miner.sampling_weights(train_pairs, hard_cfg) if hard_enabled else None
        if severity_enabled:
            if severity_scores is None:
                severity_scores = compute_pair_severities(train_pairs)
            severity_weights = severity_sampling_weights(train_pairs, severity_scores, severity_cfg)
            if sampling_weights is None:
                sampling_weights = severity_weights
            else:
                sampling_weights = sampling_weights.double() * severity_weights.double()
                sampling_weights = sampling_weights / sampling_weights.sum().clamp_min(1e-12)
        if stage_loader_sig != active_loader_sig or hard_enabled or severity_enabled:
            train_loader = build_train_loader(
                train_pairs,
                data_cfg,
                crop_size=stage_loader_sig[0],
                batch_size=stage_loader_sig[1],
                sampling_weights=sampling_weights,
            )
            if stage_loader_sig != active_loader_sig:
                active_loader_sig = stage_loader_sig
                print(
                    f"Epoch {epoch}: switched crop/batch to "
                    f"{active_loader_sig[0]}/{active_loader_sig[1]}"
                )
            previous_hard = (
                bool(
                    resolve_train_stage(cfg, epoch - 1)
                    .get("hard_mining", {})
                    .get("enabled", False)
                )
                if epoch > 1
                else False
            )
            if hard_enabled and not previous_hard:
                print("Loss-aware hard-image sampling enabled (class-balanced).")
            previous_severity = (
                bool(
                    resolve_train_stage(cfg, epoch - 1)
                    .get("severity_mining", {})
                    .get("enabled", False)
                )
                if epoch > 1
                else False
            )
            if severity_enabled and not previous_severity:
                print("Severity-aware rain sampling enabled.")
        stage_lr = stage.get("lr")
        if stage_lr is not None:
            set_optimizer_lr(optimizer, float(stage_lr))
            warmup_active = False
        else:
            warmup_active = apply_warmup_lr(optimizer, cfg, epoch)
        criterion.set_epoch(epoch, epochs)
        model.train()
        epoch_logs: List[Dict[str, float]] = []
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        accum_steps = int(stage.get("accum_steps", cfg["train"].get("accum_steps", 1)))

        for step, batch in enumerate(progress, start=1):
            global_step += 1
            degraded = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            degraded, clean, labels = degradation_copy_paste(degraded, clean, labels, data_cfg.get("augment", {}))
            degraded, clean, labels = same_label_mixup(degraded, clean, labels, data_cfg.get("augment", {}))

            with autocast(device.type, enabled=amp_enabled):
                pred, aux = model_forward(model, degraded, labels)
                loss, logs = criterion(pred, clean, labels, aux, degraded)
                scaled_loss = loss / accum_steps

            if "index" in batch:
                with torch.no_grad():
                    per_sample_mse = (
                        pred.detach().float().clamp(0.0, 1.0) - clean.float()
                    ).square().mean(dim=(1, 2, 3))
                    hard_miner.update(batch["index"], per_sample_mse)

            scaler.scale(scaled_loss).backward()
            should_step = step % accum_steps == 0 or step == len(train_loader)
            if should_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            epoch_logs.append(logs)
            if writer is not None:
                writer.add_scalar("train/loss_step", logs["loss"], global_step)
                writer.add_scalar("train/lr", current_lr(optimizer), global_step)
            if step % print_every == 0:
                progress.set_postfix(loss=f"{logs['loss']:.4f}", lr=f"{current_lr(optimizer):.2e}")

        if stage_lr is None and not warmup_active:
            scheduler.step()
        train_logs = average_logs(epoch_logs)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}: {format_logs(train_logs)} | time: {elapsed:.1f}s")
        if writer is not None:
            for key, value in train_logs.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            writer.add_scalar("train/epoch_time_sec", elapsed, epoch)

        val_psnr = None
        if val_every > 0 and epoch % val_every == 0:
            validate_ema = bool(cfg.get("validation", {}).get("validate_ema", True))
            validate_model = bool(cfg.get("validation", {}).get("validate_model", False))
            val_scores: Dict[str, float] = {}
            if validate_ema:
                ema_metrics = validate(ema.ema, val_loader, device, cfg, writer, epoch, dirs["visuals"], "ema")
                val_scores["ema"] = ema_metrics["overall"]
                print(
                    f"Epoch {epoch}: EMA validation PSNR = {ema_metrics['overall']:.4f} dB "
                    f"(rain {ema_metrics['rain']:.4f}, snow {ema_metrics['snow']:.4f}, "
                    f"lap {ema_metrics['laplacian_l1']:.5f})"
                )
                if writer is not None:
                    writer.add_scalar("val/psnr_overall_ema", ema_metrics["overall"], epoch)
                    writer.add_scalar("val/psnr_rain_ema", ema_metrics["rain"], epoch)
                    writer.add_scalar("val/psnr_snow_ema", ema_metrics["snow"], epoch)
                    writer.add_scalar("val/laplacian_l1_ema", ema_metrics["laplacian_l1"], epoch)
            if validate_model:
                model_metrics = validate(model, val_loader, device, cfg, writer, epoch, dirs["visuals"], "model")
                val_scores["model"] = model_metrics["overall"]
                print(
                    f"Epoch {epoch}: model validation PSNR = {model_metrics['overall']:.4f} dB "
                    f"(rain {model_metrics['rain']:.4f}, snow {model_metrics['snow']:.4f}, "
                    f"lap {model_metrics['laplacian_l1']:.5f})"
                )
                if writer is not None:
                    writer.add_scalar("val/psnr_overall_model", model_metrics["overall"], epoch)
                    writer.add_scalar("val/psnr_rain_model", model_metrics["rain"], epoch)
                    writer.add_scalar("val/psnr_snow_model", model_metrics["snow"], epoch)
                    writer.add_scalar("val/laplacian_l1_model", model_metrics["laplacian_l1"], epoch)
            if not val_scores:
                raise ValueError("At least one of validation.validate_ema or validation.validate_model must be true")
            val_psnr = max(val_scores.values())
            if writer is not None:
                for name, score in val_scores.items():
                    writer.add_scalar(f"val/psnr_{name}", score, epoch)
                writer.add_scalar("val/psnr_best_score", val_psnr, epoch)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                if ckpt_cfg.get("save_best", True):
                    save_checkpoint(
                        dirs["checkpoints"] / "best.pt",
                        model,
                        ema,
                        optimizer,
                        scheduler,
                        epoch,
                        best_psnr,
                        cfg,
                    )
                    print(f"Saved best checkpoint: {best_psnr:.4f} dB")

        if ckpt_cfg.get("save_last", True):
            save_checkpoint(dirs["checkpoints"] / "last.pt", model, ema, optimizer, scheduler, epoch, best_psnr, cfg)
        save_every = int(ckpt_cfg.get("save_every", 0) or 0)
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                dirs["checkpoints"] / f"epoch_{epoch:04d}.pt",
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
                cfg,
            )
        save_recent_k = int(ckpt_cfg.get("save_recent_k", 0) or 0)
        if save_recent_k > 0:
            save_recent_checkpoint(
                dirs["checkpoints"],
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
                cfg,
                save_recent_k,
            )

    if writer is not None:
        writer.close()
    print(f"Training done. Best validation PSNR: {best_psnr:.4f} dB")
    print(f"Artifacts saved under: {dirs['run']}")


if __name__ == "__main__":
    main()
