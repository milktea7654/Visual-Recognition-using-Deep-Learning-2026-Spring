# Image and Video Generation

NYCU Image and Video Generation coursework repository.

## Repository Structure

```text
.
├── HW1/
├── HW2/
├── HW3/
├── HW4/
├── final-project/
├── .gitignore
└── .gitmodules
```

## Submodules

Each lab and the final project is managed as an independent git repository and linked here as a submodule.

```bash
git submodule update --init --recursive
```

## Folder Convention

- `README.md`: task overview, setup, commands, and submission notes.
- `requirements.txt`: Python dependencies.
- `assets/`: README images and visual results.
- `docs/`: assignment PDFs, result summaries, and written notes.
- `data/`: local datasets.
- `output/`, `outputs/`, `results/`: generated files, logs, and checkpoints.
