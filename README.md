# NYCU Computer Vision 2026 Homework

This repository contains the homework submissions for NYCU Computer Vision
2026 Spring.

The final project is maintained in a separate repository:
[Visual Recognition using Deep Learning 2026 Spring Final Project](https://github.com/milktea7654/Visual-Recognition-using-Deep-Learning-2026-Spring-final-project).

## Repository Layout

```text
.
├── HW1/
│   ├── README.md
│   ├── requirements.txt
│   ├── config.py
│   ├── train.py
│   ├── inference.py
│   ├── utils.py
│   ├── assets/
│   └── docs/
├── HW2/
│   ├── README.md
│   ├── requirements.txt
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── assets/
│   └── docs/
├── HW3/
│   ├── README.md
│   ├── requirements.txt
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── assets/
│   └── docs/
└── HW4/
    ├── README.md
    ├── requirements.txt
    ├── configs/
    ├── model/
    ├── dataset.py
    ├── train.py
    ├── inference.py
    ├── assets/
    └── docs/
```

Each homework folder is self-contained. Enter the target homework directory
before installing dependencies, training, or running inference.

## File Management

- `assets/`: tracked images used by README files, such as leaderboard screenshots
  and training curves.
- `docs/`: assignment slides and supporting documents.
- `data/`: local datasets, ignored by Git.
- `output/`, `runs/`, `logs/`, `checkpoints/`: generated training artifacts,
  ignored by Git.

This keeps the repository focused on source code, configs, documentation, and
small result snapshots while leaving large local artifacts outside version
control.
