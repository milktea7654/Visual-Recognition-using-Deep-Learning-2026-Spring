# NYCU Computer Vision 2026 HW3

- **Student ID:** 112550200
- **Name:** Zheng Wu Qian

## Introduction

This repository contains the implementation for NYCU Computer Vision 2026 HW3. The goal is to perform instance segmentation on a medical cell dataset (4 foreground classes). The core architecture is **Mask R-CNN** with a **ConvNeXt-V2-Base** backbone and **FPN**, featuring sliding-window (SAHI) inference, Test-Time Augmentation (TTA), and Soft-NMS / Weighted Box Fusion for robust predictions on high-resolution TIFF images.

Key improvements over the baseline:
- ConvNeXt-V2-Base backbone (ImageNet-1K weights via HuggingFace `transformers`)
- Feature Pyramid Network (FPN) with tuned anchor sizes for small cells
- Higher-resolution mask head (56×56 logits via 28×28 RoIAlign)
- Copy-Paste and Albumentations augmentation pipeline
- Optional Cascade Mask R-CNN (IoU thresholds 0.5 → 0.6 → 0.7)

## Environment Setup

```bash
pip install -r requirements.txt
```

## Usage

### Training

Default configurations are set in `config.py`.

```bash
python train.py
```

### Inference

```bash
python inference.py --checkpoint output/<run>/checkpoints/best.pt
```

## Performance Snapshot

![Leaderboard Snapshot](leaderboard.png)
