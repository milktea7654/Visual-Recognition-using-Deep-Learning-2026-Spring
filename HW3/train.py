"""
train.py – Train ConvNeXt-Base + FPN + Mask R-CNN with SOTA tricks.

Improvements:
* Class-balanced sampling (WeightedRandomSampler)
* EMA model weights for evaluation / best.pt
* Cosine LR schedule + linear warmup
* Mixed precision + grad clip

Output layout:
    output/YYYYMMDD_HHMMSS/
    ├── checkpoints/   (best.pt, last.pt – contains EMA state if enabled)
    ├── logs/          (TensorBoard)
    └── result/

Usage:
    python train.py
    python train.py --resume output/.../checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import io
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data as data
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import (
    CellInstanceDataset,
    collate_fn,
    compute_class_balanced_weights,
    make_train_val_split,
)
from model import build_maskrcnn, build_cascade_maskrcnn, count_parameters


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Mask R-CNN")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch",  type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr",     type=float, default=config.LR)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--imgsz",  type=int, default=None,
                   help="Ignored (image size is set via config.CROP_SIZE)")
    args, _ = p.parse_known_args()  # ignore unknown args gracefully
    return args


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class ModelEMA:
    """Track exponential moving average of model parameters and buffers."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self._step = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self._step += 1
        # Adaptive decay: ramp from ~0.5 up to config.EMA_DECAY over first 2000 steps.
        # Prevents the EMA from being stuck near random init early in training.
        d = min(self.decay, (1 + self._step) / (10 + self._step))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)


# ---------------------------------------------------------------------------
# COCO eval
# ---------------------------------------------------------------------------


def _build_coco_gt(dataset: CellInstanceDataset) -> COCO:
    images, annotations = [], []
    ann_id = 1
    for idx in range(len(dataset)):
        _img, target = dataset[idx]
        h, w = _img.shape[-2:]
        images.append({"id": idx, "height": h, "width": w})

        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        masks = target["masks"].numpy()

        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            rle = coco_mask.encode(np.asfortranarray(masks[i].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": int(labels[i]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
                "segmentation": rle,
            })
            ann_id += 1

    categories = [{"id": c, "name": f"class{c}"} for c in range(1, config.NUM_CLASSES + 1)]
    coco_dict = {"images": images, "annotations": annotations, "categories": categories}

    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    return coco_gt


@torch.no_grad()
def evaluate(model, loader, device, coco_gt: COCO, score_thr: float) -> float:
    model.eval()
    # Temporarily lower the model's internal score threshold so low-confidence
    # predictions are still returned and counted by COCOeval.
    old_score_thresh = getattr(getattr(model, 'roi_heads', None), 'score_thresh', None)
    if old_score_thresh is not None:
        model.roi_heads.score_thresh = min(old_score_thresh, 0.01)
    results = []
    try:
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for tgt, out in zip(targets, outputs):
                img_id = int(tgt["image_id"].item())
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                masks = (out["masks"].squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
                boxes = out["boxes"].cpu().numpy()

                for i in range(scores.shape[0]):
                    if scores[i] < score_thr:
                        continue
                    rle = coco_mask.encode(np.asfortranarray(masks[i]))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    x1, y1, x2, y2 = boxes[i].tolist()
                    results.append({
                        "image_id": img_id,
                        "category_id": int(labels[i]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(scores[i]),
                        "segmentation": rle,
                    })

        if not results:
            return 0.0

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(results)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        return float(coco_eval.stats[1])  # AP@0.50
    finally:
        # Always restore original threshold
        if old_score_thresh is not None:
            model.roi_heads.score_thresh = old_score_thresh


# ---------------------------------------------------------------------------
# Sliding-window evaluation (matches test-time inference conditions)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_sliding_window(model, val_ids: List[str], device, score_thr: float) -> float:
    """Evaluate on full-resolution val images with sliding window.

    Mirrors predict_image() in inference.py so that the val metric reflects
    the same distribution shift the model sees at test time (full images,
    padding of small images, sliding window for large images).
    Only used when USE_FULL_TRAIN=False and SW_EVAL_EVERY > 0.
    """
    from dataset import _load_image_tif, _load_mask_tif
    import torchvision.transforms.functional as TF_func
    import torchvision.ops as tv_ops

    model.eval()
    old_thresh = getattr(getattr(model, "roi_heads", None), "score_thresh", None)
    if old_thresh is not None:
        model.roi_heads.score_thresh = min(old_thresh, 0.01)

    tile = config.CROP_SIZE
    step = int(tile * (1 - config.SLIDING_WINDOW_OVERLAP))
    images_info: List = []
    annotations_info: List = []
    results: List = []
    ann_id = 1

    try:
        for img_idx, sid in enumerate(val_ids):
            img_path = Path(config.DATA_ROOT) / sid / "image.tif"
            if not img_path.exists():
                continue
            img_np = _load_image_tif(img_path)   # (H, W, 3) uint8
            H, W = img_np.shape[:2]
            images_info.append({"id": img_idx, "height": H, "width": W})

            # GT in full-image coordinates
            for cls in range(1, config.NUM_CLASSES + 1):
                mp = Path(config.DATA_ROOT) / sid / f"class{cls}.tif"
                if not mp.exists():
                    continue
                mask_arr = _load_mask_tif(mp)
                inst_ids = np.unique(mask_arr)
                inst_ids = inst_ids[inst_ids > 0]
                for iid in inst_ids:
                    binary = (mask_arr == iid).astype(np.uint8)
                    rle = coco_mask.encode(np.asfortranarray(binary))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    ys_b, xs_b = np.where(binary)
                    x1, y1 = int(xs_b.min()), int(ys_b.min())
                    x2, y2 = int(xs_b.max()) + 1, int(ys_b.max()) + 1
                    annotations_info.append({
                        "id": ann_id, "image_id": img_idx, "category_id": cls,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": 0, "segmentation": rle,
                    })
                    ann_id += 1

            # Build tile list (pad small images to tile×tile)
            if H <= tile and W <= tile:
                padded = np.zeros((tile, tile, 3), dtype=img_np.dtype)
                padded[:H, :W] = img_np
                tile_list = [(padded, 0, 0)]
            else:
                oys = list(range(0, max(1, H - tile + 1), step))
                if H > tile and oys[-1] != H - tile:
                    oys.append(H - tile)
                oxs = list(range(0, max(1, W - tile + 1), step))
                if W > tile and oxs[-1] != W - tile:
                    oxs.append(W - tile)
                tile_list = []
                for oy in oys:
                    for ox in oxs:
                        patch = img_np[oy:oy + tile, ox:ox + tile]
                        ph, pw = patch.shape[:2]
                        if ph < tile or pw < tile:
                            p2 = np.zeros((tile, tile, 3), dtype=patch.dtype)
                            p2[:ph, :pw] = patch
                            patch = p2
                        tile_list.append((patch, oy, ox))

            # Forward pass per tile
            all_boxes: List = []
            all_scores: List = []
            all_labels_l: List = []
            all_placements: List = []
            for patch_np, oy, ox in tile_list:
                img_t = TF_func.to_tensor(patch_np).unsqueeze(0).to(device)
                out = model(img_t)[0]
                boxes = out["boxes"].cpu().numpy()
                sc = out["scores"].cpu().numpy()
                lb = out["labels"].cpu().numpy()
                ms = (out["masks"].squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
                del out
                torch.cuda.empty_cache()
                for i in range(sc.shape[0]):
                    if sc[i] < score_thr:
                        continue
                    gx1 = max(0.0, min(float(boxes[i][0]) + ox, W))
                    gy1 = max(0.0, min(float(boxes[i][1]) + oy, H))
                    gx2 = max(0.0, min(float(boxes[i][2]) + ox, W))
                    gy2 = max(0.0, min(float(boxes[i][3]) + oy, H))
                    if gx2 <= gx1 or gy2 <= gy1:
                        continue
                    all_boxes.append([gx1, gy1, gx2, gy2])
                    all_scores.append(float(sc[i]))
                    all_labels_l.append(int(lb[i]))
                    all_placements.append((oy, ox, ms[i]))

            if not all_boxes:
                continue

            # Per-class NMS
            boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
            scores_t2 = torch.tensor(all_scores, dtype=torch.float32)
            labels_t2 = torch.tensor(all_labels_l, dtype=torch.int64)
            keep_indices: List[int] = []
            for cls in range(1, config.NUM_CLASSES + 1):
                cls_m = labels_t2 == cls
                if cls_m.sum() == 0:
                    continue
                idx = torch.where(cls_m)[0]
                k = tv_ops.nms(boxes_t[idx], scores_t2[idx], 0.5)
                keep_indices.extend(idx[k].tolist())

            for ki in keep_indices:
                oy, ox, small_mask = all_placements[ki]
                th, tw = small_mask.shape
                full_mask = np.zeros((H, W), dtype=np.uint8)
                ey = min(oy + th, H)
                ex = min(ox + tw, W)
                full_mask[oy:ey, ox:ex] = small_mask[:ey - oy, :ex - ox]
                rle2 = coco_mask.encode(np.asfortranarray(full_mask))
                rle2["counts"] = rle2["counts"].decode("utf-8")
                b = all_boxes[ki]
                results.append({
                    "image_id": img_idx,
                    "category_id": all_labels_l[ki],
                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                    "score": all_scores[ki],
                    "segmentation": rle2,
                })

        if not results:
            return 0.0

        categories = [{"id": c, "name": f"class{c}"} for c in range(1, config.NUM_CLASSES + 1)]
        coco_dict2 = {"images": images_info, "annotations": annotations_info, "categories": categories}
        coco_gt_sw = COCO()
        coco_gt_sw.dataset = coco_dict2
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt_sw.createIndex()
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt_sw.loadRes(results)
            coco_eval2 = COCOeval(coco_gt_sw, coco_dt, iouType="segm")
            coco_eval2.evaluate()
            coco_eval2.accumulate()
            coco_eval2.summarize()
        return float(coco_eval2.stats[1])   # AP@0.50
    finally:
        if old_thresh is not None:
            model.roi_heads.score_thresh = old_thresh


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def cosine_warmup_lambda(epoch: int, total: int) -> float:
    """Linear warmup → cosine decay to LR_MIN/LR ratio."""
    if epoch < config.WARMUP_EPOCHS:
        return config.WARMUP_FACTOR + (1 - config.WARMUP_FACTOR) * epoch / max(1, config.WARMUP_EPOCHS)
    progress = (epoch - config.WARMUP_EPOCHS) / max(1, total - config.WARMUP_EPOCHS)
    progress = min(max(progress, 0.0), 1.0)
    min_ratio = config.LR_MIN / config.LR
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model, ema, optimizer, loader, device, scaler, grad_clip: float,
) -> Dict[str, float]:
    model.train()
    sums: Dict[str, float] = {}
    n_batches = 0
    skipped_empty_loss = 0
    skipped_nonfinite = 0

    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [
            {name: value.to(device, non_blocking=True) for name, value in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                if not loss_dict:
                    skipped_empty_loss += 1
                    continue
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                continue
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            if not loss_dict:
                skipped_empty_loss += 1
                continue
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        for k, v in loss_dict.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach())
        sums["total"] = sums.get("total", 0.0) + float(loss.detach())
        n_batches += 1

    if n_batches == 0:
        return {
            "total": float("nan"),
            "_updates": 0.0,
            "_skipped_empty_loss": float(skipped_empty_loss),
            "_skipped_nonfinite": float(skipped_nonfinite),
        }

    out = {k: v / n_batches for k, v in sums.items()}
    out["_updates"] = float(n_batches)
    if skipped_empty_loss:
        out["_skipped_empty_loss"] = float(skipped_empty_loss)
    if skipped_nonfinite:
        out["_skipped_nonfinite"] = float(skipped_nonfinite)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.OUTPUT_DIR) / timestamp
    ckpt_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    result_dir = output_dir / "result"
    for d in (ckpt_dir, logs_dir, result_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Output → {output_dir}")

    # ----- Dataset -----
    train_ids, val_ids = make_train_val_split()

    if config.USE_FULL_TRAIN:
        # Use all images for final submission — no val split.
        # Early stopping is disabled; model saves every SW_EVAL_EVERY epochs.
        all_ids = val_ids + train_ids   # same deterministic order
        train_ids = all_ids
        val_ids = []
        print(f"USE_FULL_TRAIN=True → {len(train_ids)} images, no val evaluation")
    else:
        print(f"Split: train={len(train_ids)}  val={len(val_ids)}")

    train_set = CellInstanceDataset(config.DATA_ROOT, train_ids, train=True)

    # Only build val infrastructure when we have val images
    val_loader = None
    coco_gt = None
    if val_ids:
        val_set = CellInstanceDataset(config.DATA_ROOT, val_ids, train=False)
        val_loader = data.DataLoader(
            val_set, batch_size=1, shuffle=False,
            num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        )
        print("Building COCO ground-truth for crop-level validation...")
        coco_gt = _build_coco_gt(val_set)

    # Class-balanced sampler
    if config.USE_CLASS_BALANCED_SAMPLER:
        weights = compute_class_balanced_weights(train_set.majority_classes)
        sampler = data.WeightedRandomSampler(
            weights, num_samples=len(train_set), replacement=True,
        )
        shuffle = False
        print("Using WeightedRandomSampler for class balance")
    else:
        sampler = None
        shuffle = True

    train_loader = data.DataLoader(
        train_set, batch_size=args.batch, sampler=sampler, shuffle=shuffle,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )

    # ----- Model -----
    if getattr(config, "USE_CASCADE", False):
        model = build_cascade_maskrcnn(pretrained_backbone=True, multi_scale=True).to(device)
        print("Architecture: Cascade Mask R-CNN (3 IoU stages: 0.5→0.6→0.7)")
    else:
        model = build_maskrcnn(pretrained_backbone=True, multi_scale=True).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params / 1e6:.2f} M  (limit 200 M)")
    assert n_params < 200 * 1e6, "Parameter budget exceeded!"

    ema = ModelEMA(model, decay=config.EMA_DECAY) if config.USE_EMA else None
    if ema:
        ema.module.to(device)
        print(f"EMA enabled (decay={config.EMA_DECAY})")

    # ----- Optimizer / Scheduler -----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: cosine_warmup_lambda(e, args.epochs),
    )
    scaler = torch.amp.GradScaler("cuda") if config.USE_AMP and device.type == "cuda" else None

    # ----- Resume -----
    start_epoch = 0
    best_ap50 = -1.0
    best_loss = float("inf")
    epochs_no_improve = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if ema and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt.get("epoch", 0)
        best_ap50 = ckpt.get("best_ap50", -1.0)
        best_loss = ckpt.get("best_loss", float("inf"))
        resumed_base_lrs = list(getattr(scheduler, "base_lrs", []))
        resumed_lrs = [pg.get("lr", args.lr) for pg in optimizer.param_groups]
        if resumed_base_lrs and any(not math.isclose(base_lr, args.lr, rel_tol=0.0, abs_tol=1e-12) for base_lr in resumed_base_lrs):
            scaled_lrs = []
            for lr, base_lr in zip(resumed_lrs, resumed_base_lrs):
                factor = lr / base_lr if base_lr > 0 else 1.0
                scaled_lrs.append(args.lr * factor)
            for pg, new_lr in zip(optimizer.param_groups, scaled_lrs):
                pg["initial_lr"] = args.lr
                pg["lr"] = new_lr
            scheduler.base_lrs = [args.lr for _ in optimizer.param_groups]
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = [pg["lr"] for pg in optimizer.param_groups]
            print(f"Overriding resumed LR schedule to base lr={args.lr:.2e}")
        if not math.isfinite(best_loss) or best_loss <= 0.0:
            print(f"Ignoring invalid best_loss={best_loss:.4f} from checkpoint")
            best_loss = float("inf")
        print(f"Resumed from {args.resume} (epoch={start_epoch}, best AP50={best_ap50:.4f}, best loss={best_loss:.4f})")

    tb = SummaryWriter(log_dir=str(logs_dir))
    pbar = tqdm(total=args.epochs, initial=start_epoch,
                desc=f"Epoch[{start_epoch + 1}/{args.epochs}]",
                unit="epoch", file=sys.stdout, leave=True)

    try:
        for epoch in range(start_epoch, args.epochs):
            pbar.set_description(f"Epoch[{epoch + 1}/{args.epochs}]")
            t0 = time.time()
            losses = train_one_epoch(
                model, ema, optimizer, train_loader, device, scaler, config.GRAD_CLIP,
            )
            updates = int(losses.get("_updates", 0.0))
            skipped_nonfinite = int(losses.get("_skipped_nonfinite", 0.0))
            attempted_batches = updates + skipped_nonfinite
            if updates == 0:
                tqdm.write("Stopping: epoch produced no finite updates; training has diverged.")
                tqdm.write("Resume from an earlier checkpoint with a lower LR, e.g. epoch_40.pt and --lr 5e-5.")
                break
            if attempted_batches > 0 and skipped_nonfinite / attempted_batches > 0.5:
                tqdm.write(
                    f"Stopping: skipped_nonfinite={skipped_nonfinite}/{attempted_batches}; "
                    "training is numerically unstable."
                )
                tqdm.write("Resume from an earlier checkpoint with a lower LR, e.g. epoch_40.pt and --lr 5e-5.")
                break
            scheduler.step()
            elapsed = time.time() - t0

            # ---- Evaluation ----
            # USE_FULL_TRAIN: no val set → skip evaluation, use periodic saving
            # Otherwise: fast crop-level eval every epoch + SW eval every SW_EVAL_EVERY epochs
            ap50 = 0.0
            sw_ap50 = None
            if not config.USE_FULL_TRAIN and val_loader is not None:
                ap50 = evaluate(model, val_loader, device, coco_gt, config.VAL_SCORE_THRESHOLD)
                sw_every = getattr(config, "SW_EVAL_EVERY", 0)
                if sw_every > 0 and (epoch + 1) % sw_every == 0:
                    tqdm.write(f"  Running sliding-window eval (epoch {epoch + 1})...")
                    sw_ap50 = evaluate_sliding_window(
                        model, val_ids, device, config.VAL_SCORE_THRESHOLD
                    )
                    # SW eval is the primary metric for best-checkpoint selection
                    ap50 = sw_ap50

            loss_str = "  ".join(
                f"{k}={v:.4f}" for k, v in losses.items()
                if k != "total" and not k.startswith("_")
            )
            train_stats = "  ".join(
                f"{k[1:]}={int(v)}" for k, v in losses.items()
                if k.startswith("_")
            )
            ap_info = f"AP50={ap50:.4f}" if not config.USE_FULL_TRAIN else "AP50=N/A"
            if sw_ap50 is not None:
                ap_info = f"AP50(crop)={ap50:.4f}  AP50(SW)={sw_ap50:.4f}"
            cur_total = float(losses.get("total", float("nan")))
            train_info = f"{loss_str}  {train_stats}" if train_stats else loss_str
            tqdm.write(f"Epoch[{epoch + 1}/{args.epochs}]")
            tqdm.write(f"  total={cur_total:.4f}  {train_info}  {ap_info}  best_loss={best_loss:.4f}  ({elapsed:.1f}s)")

            for k, v in losses.items():
                tb.add_scalar(f"train/{k.lstrip('_')}", v, epoch + 1)
            tb.add_scalar("val/AP50", ap50, epoch + 1)
            if sw_ap50 is not None:
                tb.add_scalar("val/AP50_SW", sw_ap50, epoch + 1)
            tb.add_scalar("train/best_loss", best_loss, epoch + 1)
            tb.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
            tb.flush()

            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "ema":   ema.state_dict() if ema else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "best_ap50": best_ap50,
                "best_loss": best_loss,
                "config_backbone": config.BACKBONE_NAME,
            }
            torch.save(ckpt, ckpt_dir / "last.pt")

            # Save best-loss checkpoint (works regardless of USE_FULL_TRAIN)
            if math.isfinite(cur_total) and cur_total < best_loss:
                best_loss = cur_total
                ckpt["best_loss"] = best_loss
                torch.save(ckpt, ckpt_dir / "best_loss.pt")
                tqdm.write(f"  -> best_loss.pt saved (loss={best_loss:.4f})")

            if config.USE_FULL_TRAIN:
                # No val metric → save every SW_EVAL_EVERY epochs as periodic checkpoint
                every = max(1, getattr(config, "SW_EVAL_EVERY", 10))
                if (epoch + 1) % every == 0:
                    torch.save(ckpt, ckpt_dir / f"epoch_{epoch + 1}.pt")
                    tqdm.write(f"  -> Checkpoint saved epoch={epoch + 1}")
            else:
                if ap50 > best_ap50:
                    best_ap50 = ap50
                    ckpt["best_ap50"] = best_ap50
                    torch.save(ckpt, ckpt_dir / "best.pt")
                    tqdm.write(f"  -> Best model saved (AP50={ap50:.4f})")
                    epochs_no_improve = 0
                else:
                    # Only count patience once the model has produced a non-zero AP50.
                    if best_ap50 > 0.0:
                        epochs_no_improve += 1

                if config.SAVE_EVERY > 0 and (epoch + 1) % config.SAVE_EVERY == 0:
                    torch.save(ckpt, ckpt_dir / f"epoch_{epoch + 1}.pt")

                if best_ap50 > 0.0 and epochs_no_improve >= config.EARLY_STOP_PATIENCE:
                    tqdm.write(f"Early stopping: no AP50 improvement for {config.EARLY_STOP_PATIENCE} epochs.")
                    break

            pbar.update(1)

    finally:
        pbar.close()
        tb.close()
        print(f"\nTraining complete. Best AP50 = {best_ap50:.4f}  Best loss = {best_loss:.4f}")
        print(f"Outputs → {output_dir}")


if __name__ == "__main__":
    main()
