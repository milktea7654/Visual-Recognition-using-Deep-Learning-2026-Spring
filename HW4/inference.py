"""Generate pred.npz and pred.zip for HW4 submission."""

from __future__ import annotations

import argparse
import importlib
import os
import random
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RainSnowTestDataset


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
        return yaml.safe_load(handle) or {}


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
    cur = cfg
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW4 inference.")
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--data_root", type=str, default="", help="Override data.root.")
    parser.add_argument("--checkpoint", type=str, default="", help="Override inference.checkpoint.")
    parser.add_argument("--output_npz", type=str, default="", help="Override inference.output_npz.")
    parser.add_argument("--output_zip", type=str, default="", help="Override inference.output_zip.")
    parser.add_argument("--set", action="append", default=[], help="Override config with dotted keys.")
    return parser.parse_args()


def prepare_config() -> Dict[str, Any]:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        set_by_dotted_key(cfg, "data.root", args.data_root)
    if args.checkpoint:
        set_by_dotted_key(cfg, "inference.checkpoint", args.checkpoint)
    if args.output_npz:
        set_by_dotted_key(cfg, "inference.output_npz", args.output_npz)
    if args.output_zip:
        set_by_dotted_key(cfg, "inference.output_zip", args.output_zip)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        key, raw_value = item.split("=", 1)
        set_by_dotted_key(cfg, key, yaml.safe_load(raw_value))
    return cfg


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    module = importlib.import_module(f"model.{model_cfg['name']}")
    return module.build_model(model_cfg)


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


def _gravity_aware_tta_transforms(x: Tensor) -> List[Tensor]:
    return [
        x,
        torch.flip(x, dims=[-1]),
    ]


def _gravity_aware_tta_inverse(y: Tensor, index: int) -> Tensor:
    if index == 0:
        return y
    if index == 1:
        return torch.flip(y, dims=[-1])
    raise ValueError(index)


def _eight_way_tta_transforms(x: Tensor) -> List[Tensor]:
    transforms: List[Tensor] = []
    for k in range(4):
        rotated = torch.rot90(x, k=k, dims=(-2, -1))
        transforms.append(rotated)
        transforms.append(torch.flip(rotated, dims=[-1]))
    return transforms


def _eight_way_tta_inverse(y: Tensor, index: int) -> Tensor:
    k = index // 2
    is_flip = index % 2 == 1
    if is_flip:
        y = torch.flip(y, dims=[-1])
    return torch.rot90(y, k=-k, dims=(-2, -1))


@torch.no_grad()
def geometric_tta_inference(
    model: nn.Module,
    x: Tensor,
    tile: int = 256,
    overlap: int = 48,
    mode: str = "gravity",
) -> Tensor:
    outputs = []
    mode = str(mode).lower()
    if mode in {"eight", "8", "8way", "eight_way"}:
        if not getattr(geometric_tta_inference, "_warned_eight_way", False):
            print(
                "Warning: tta_mode=eight applies rotations/vertical flips. "
                "For rain/snow checkpoints trained only with hflip, tta_mode=gravity is usually safer."
            )
            setattr(geometric_tta_inference, "_warned_eight_way", True)
        transforms = _eight_way_tta_transforms(x)
        inverse = _eight_way_tta_inverse
    elif mode in {"gravity", "hflip", "horizontal"}:
        transforms = _gravity_aware_tta_transforms(x)
        inverse = _gravity_aware_tta_inverse
    else:
        raise ValueError(f"Unsupported tta_mode: {mode!r}")
    for idx, aug in enumerate(transforms):
        pred = tiled_inference(model, aug, tile=tile, overlap=overlap)
        outputs.append(inverse(pred, idx))
    return torch.stack(outputs, dim=0).mean(dim=0)


@torch.no_grad()
def tta_inference(
    model: nn.Module,
    x: Tensor,
    tile: int = 256,
    overlap: int = 48,
    scales: List[float] | None = None,
    scale_weights: List[float] | None = None,
    mode: str = "gravity",
) -> Tensor:
    scales = scales or [1.0]
    scale_weights = scale_weights or [1.0 for _ in scales]
    if len(scales) != len(scale_weights):
        raise ValueError("tta_scales and tta_scale_weights must have the same length")
    _, _, height, width = x.shape
    outputs = []
    weights = []
    for scale, weight in zip(scales, scale_weights):
        if scale <= 0:
            raise ValueError(f"Invalid TTA scale: {scale}")
        scaled = x
        if abs(scale - 1.0) > 1e-6:
            scaled = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        pred = geometric_tta_inference(model, scaled, tile=tile, overlap=overlap, mode=mode)
        if pred.shape[-2:] != (height, width):
            pred = F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=False)
        outputs.append(pred)
        weights.append(float(weight))
    weight_tensor = x.new_tensor(weights).view(-1, 1, 1, 1, 1)
    stacked = torch.stack(outputs, dim=0)
    return (stacked * weight_tensor).sum(dim=0) / weight_tensor.sum().clamp_min(1e-8)


def tensor_to_uint8_chw(tensor: Tensor) -> np.ndarray:
    tensor = tensor.detach().float().clamp(0.0, 1.0).cpu()
    return (tensor.squeeze(0).numpy() * 255.0).round().astype(np.uint8)


def checkpoint_paths_from_config(cfg: Dict[str, Any]) -> List[str]:
    paths = cfg["inference"].get("checkpoints", None)
    if paths:
        return [str(path) for path in paths]
    return [str(cfg["inference"]["checkpoint"])]


def checkpoint_weight_key(checkpoint: Dict[str, Any], use_ema: bool) -> str:
    return "ema" if use_ema and "ema" in checkpoint else "model"


def remap_legacy_singleton_stage_keys(state: Dict[str, Tensor], model: nn.Module) -> Dict[str, Tensor]:
    """Load older NAF/MSFN checkpoints saved when every stage was wrapped in Sequential."""
    model_keys = set(model.state_dict().keys())
    prefixes = (
        "encoder_level1",
        "encoder_level2",
        "encoder_level3",
        "latent",
        "decoder_level3",
        "decoder_level2",
        "decoder_level1",
        "refinement",
    )
    remapped: Dict[str, Tensor] = {}
    changed = False
    for name, tensor in state.items():
        new_name = name
        for prefix in prefixes:
            token = f"{prefix}.0."
            if name.startswith(token):
                candidate = f"{prefix}.{name[len(token):]}"
                if candidate in model_keys:
                    new_name = candidate
                    changed = True
                break
        remapped[new_name] = tensor
    return remapped if changed else state


def load_state_dict_compat(model: nn.Module, state: Dict[str, Tensor]) -> None:
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        remapped = remap_legacy_singleton_stage_keys(state, model)
        if remapped is state:
            raise
        model.load_state_dict(remapped, strict=True)
        print("Loaded checkpoint with legacy singleton-stage key remapping.")


def load_weights(checkpoint_path: str | Path, model: nn.Module, device: torch.device, use_ema: bool) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    key = checkpoint_weight_key(checkpoint, use_ema)
    load_state_dict_compat(model, checkpoint[key])
    return checkpoint


def load_averaged_weights(
    checkpoint_paths: List[str],
    model: nn.Module,
    device: torch.device,
    use_ema: bool,
) -> Dict[str, Any]:
    if len(checkpoint_paths) == 1:
        return load_weights(checkpoint_paths[0], model, device, use_ema=use_ema)

    averaged: Dict[str, Tensor] = {}
    first_checkpoint: Dict[str, Any] | None = None
    for idx, path in enumerate(checkpoint_paths):
        checkpoint = torch.load(path, map_location="cpu")
        if first_checkpoint is None:
            first_checkpoint = checkpoint
        state = checkpoint[checkpoint_weight_key(checkpoint, use_ema)]
        for name, tensor in state.items():
            tensor = tensor.detach()
            if tensor.dtype.is_floating_point:
                value = tensor.float() / len(checkpoint_paths)
                averaged[name] = value if idx == 0 else averaged[name] + value
            elif idx == 0:
                averaged[name] = tensor.clone()
    load_state_dict_compat(model, averaged)
    return first_checkpoint or {}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    cfg = prepare_config()
    seed_everything(int(cfg.get("project", {}).get("seed", 3407)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_paths = checkpoint_paths_from_config(cfg)
    checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
    if isinstance(checkpoint, dict) and "config" in checkpoint and "model" in checkpoint["config"]:
        requested_model_name = cfg.get("model", {}).get("name", checkpoint["config"]["model"].get("name"))
        cfg["model"] = dict(checkpoint["config"]["model"])
        cfg["model"]["name"] = requested_model_name

    model = build_model(cfg).to(device)
    use_ema = bool(cfg["inference"].get("use_ema", True))
    load_averaged_weights(checkpoint_paths, model, device, use_ema=use_ema)
    model.eval()

    dataset = RainSnowTestDataset(cfg["data"]["root"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["inference"].get("batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg["inference"].get("num_workers", 2)),
        pin_memory=True,
        drop_last=False,
    )

    tile = int(cfg["inference"].get("tile", 256))
    overlap = int(cfg["inference"].get("overlap", 48))
    use_tta = bool(cfg["inference"].get("tta", True))
    tta_mode = str(cfg["inference"].get("tta_mode", "gravity"))
    tta_scales = [float(scale) for scale in cfg["inference"].get("tta_scales", [1.0])]
    tta_scale_weights = [
        float(weight) for weight in cfg["inference"].get("tta_scale_weights", [1.0 for _ in tta_scales])
    ]
    results: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="infer"):
            degraded = batch["degraded"].to(device, non_blocking=True)
            filenames = batch["filename"]
            pred = (
                tta_inference(
                    model,
                    degraded,
                    tile=tile,
                    overlap=overlap,
                    scales=tta_scales,
                    scale_weights=tta_scale_weights,
                    mode=tta_mode,
                )
                if use_tta
                else tiled_inference(model, degraded, tile=tile, overlap=overlap)
            )
            for idx, filename in enumerate(filenames):
                results[filename] = tensor_to_uint8_chw(pred[idx : idx + 1])

    output_npz = Path(cfg["inference"].get("output_npz", "outputs/pred.npz"))
    mkdir(output_npz.parent)
    np.savez_compressed(output_npz, **results)
    print(f"Saved {len(results)} restored images to {output_npz}")

    output_zip = cfg["inference"].get("output_zip", "")
    if output_zip:
        output_zip = Path(output_zip)
        mkdir(output_zip.parent)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_npz, arcname="pred.npz")
        print(f"Saved submission zip to {output_zip}. It contains pred.npz at the root.")


if __name__ == "__main__":
    main()
