# NYCU Computer Vision 2026 HW2

- **Student ID:** 112550200
- **Name:** Zheng Wu Qian

## Introduction

This repository contains the implementation for NYCU Computer Vision 2026 HW2. The goal is to perform object detection on a multiple-digit dataset. The core architecture relies on **DINO (DEtection TRansformer with Improved DeNoising)**, utilizing a ResNet-50 backbone. 

## Environment Setup

How to install dependencies.

```bash
pip install -r requirements.txt
```

## Usage

### Training

How to train your model. (Default configurations are set in `config.py`)

```bash
python train.py
```

### Inference
```bash
python inference.py --checkpoint output/20260421_015205/best.pth --batch_size 8
```

## Performance Snapshot

![Leaderboard Snapshot](assets/leaderboard.png)
