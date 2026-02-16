Project: Fire detection with EfficientNet on multiband TIFFs

What it does

- Trains and evaluates an EfficientNet classifier for a binary dataset (`Fire`, `No_Fire`) using multiband TIFF inputs. Features:
  - Loading and preparing ImageFolder-style datasets
  - Multiband TIFF reading (11 bands padded to 13)
  - Training-time augmentations
  - Training using Hugging Face `Trainer`
  - Saving checkpoints, metrics, and plots (confusion matrix, loss curves)
  - Saving misclassified images and generating Grad-CAM visualizations

Repository structure (summary)

- `train_efficientnet.py` — Main training script for multiband EfficientNet.
- `data/` — loaders, augmentations and collators:
  - `dataset_loader.py` — `load_imagefolder` helper to build datasets.
  - `augmentations.py` — training augmentations (uses `torchvision.transforms.v2`).
  - `collators_efficientnet.py` — collator for precomputed `pixel_values`.
  - `multiband_tiff.py` — TIFF reader, padding to target channels, and RGB preview helper.
- `models/` — model loaders (`model_loader.py`).
- `training/` — training utilities (`metrics.py`, `trainer_args.py`).
- `utils/` — utilities for Grad-CAM, saving errors, plotting, listing false positives.
- `resultados_efficientnet/` — output folder where runs and artifacts are saved.
- `train/`, `val/`, `test/` — expected ImageFolder data layout with two subfolders: `Fire/` and `No_Fire/`.
- `requirements.txt` — Python dependencies.

Installation and dependencies

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

How to use (basic flow)

1. Prepare dataset: place `train/`, `val/`, and `test/` folders with subfolders `Fire/` and `No_Fire/` under the repo root, or change `DATA_PATH` in `train_efficientnet.py`.
2. Select model: change the `MODEL_NAME` constant at the top of `train_efficientnet.py`.
3. Adjust training arguments: edit `training/trainer_args.py` or pass different args to `get_training_args(...)` from the script.
4. Adjust augmentations: edit `data/augmentations.py` to change training transforms.
5. Run training: `python train_efficientnet.py`.

Example command

```bash
python train_efficientnet.py
```

Notes and tips

- Quick testing: `train_efficientnet.py` contains a commented block that reduces the dataset size (`shuffle().select(range(...))`) for fast trials.
- Resume training: to resume from a checkpoint, enable `resume_from_checkpoint` in the `trainer.train(...)` call.
- Outputs: run artifacts are saved under `resultados_efficientnet/` and include `all_results.json`, `eval_results.json`, `classification_report.txt`, `misclassified/`, and the saved model.
- GPU: Hugging Face `Trainer` uses GPU if available (and configured). Ensure you have a compatible `torch` + CUDA installation.
- Multiband inputs: `train_efficientnet.py` loads 11-band TIFFs, pads to 13 channels, and expands the first conv layer to match.

Common parameters to change

- `DATA_PATH`: dataset root path.
- `MODEL_NAME`: top of the training script.
- `training/trainer_args.py`: learning rate, epochs, batch size, etc.
- `data/augmentations.py`: input size, crop, flips, color jitter, etc.
- `train_efficientnet.py`: `TARGET_CHANNELS` for multiband inputs (default 13).

## Automated setup and run script

This repository includes a convenience script `setup_and_run.sh` that automates common setup tasks and launches training. Summary:

- What it does (high level):
  - updates the OS packages and installs basic tools (`wget`, `git`, `unzip`)
  - installs Miniconda (if not present) and creates/activates a Conda env named `vit_env`
  - installs Python dependencies (PyTorch, Transformers, Datasets, Pillow, etc.) — the script uses a sentinel file `.dependencies_installed` to avoid reinstalling on subsequent runs
  - synchronizes the dataset from a GCS bucket using `gsutil rsync` (idempotent)
  - runs the training command (`python train_efficientnet.py`)

- Prerequisites and notes:
  - The script uses `sudo` for system package operations; run it as a user with sudo privileges.
  - `gsutil` must be available and authenticated for the dataset sync to work (or change the script to use another copy mechanism).
  - The script installs Miniconda under `$HOME/miniconda` if missing. If you prefer a different Conda installation, edit the `CONDA_PATH` variable.
  - It pins the Conda environment name to `vit_env`. Modify `ENV_NAME` in the script to change this.
  - The dataset bucket path (`gs://training_data_v1_new/dataset/`) is the default in the script — change it if your data lives elsewhere.
  - The script installs PyTorch via the CUDA-specific wheel URL in the file. Verify the CUDA version matches your hardware or change the pip install line.

- How to run:

```bash
# Make the script executable (once)
chmod +x setup_and_run.sh

# Run it
./setup_and_run.sh
```

- Customization tips:
  - To only install dependencies without running training, comment out the final `python` line.
  - To use a different dataset source, replace the `gsutil rsync` command with your copy/sync command.
