Project: Fire detection with ViT and EfficientNet

What it does
- A set of scripts to train and evaluate image classification models (Vision Transformer and EfficientNet) on a binary dataset (`Fire`, `No_Fire`). Features:
	- Loading and preparing ImageFolder-style datasets
	- Training-time augmentations
	- Training using Hugging Face `Trainer`
	- Saving checkpoints, metrics, and plots (confusion matrix, loss curves)
	- Saving misclassified images and generating Grad-CAM visualizations

Repository structure (summary)
- `train_vit.py` — Train a ViT model (uses `models/vit_factory.py`).
- `train_efficientnet.py` — Train an EfficientNet model (uses `models/model_loader.py`).
- `data/` — loaders, augmentations and collators:
	- `dataset_loader.py` — `load_imagefolder` helper to build datasets.
	- `augmentations.py` — training augmentations implemented with `torchvision` (PIL).
	- `collators.py`, `collators_efficientnet.py` — collators that batch images and call the model `processor`.
- `models/` — model factories/loaders and processors (`vit_factory.py`, `model_loader.py`).
- `training/` — training utilities (`metrics.py`, `trainer_args.py`).
- `utils/` — utilities for Grad-CAM, saving errors, plotting, listing false positives.
- `resultados_vit/` and `resultados_efficientnet/` — output folders where runs and artifacts are saved.
- `test/`, `train/`, `val/` — expected ImageFolder data layout with two subfolders: `Fire/` and `No_Fire/`.
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

Note: this project relies on `transformers`, `datasets`, `torch`, `torchvision` and other common libraries — check `requirements.txt` for exact versions.

How to use (basic flow)
1. Prepare dataset: place `train/`, `val/`, and `test/` folders with subfolders `Fire/` and `No_Fire/` under the repo root, or change `DATA_PATH` in the training scripts.
2. Select model: change the `MODEL_NAME` constant at the top of `train_vit.py` or `train_efficientnet.py` (or modify factory calls).
3. Adjust training arguments: edit `training/trainer_args.py` or pass different args to `get_training_args(...)` from the scripts.
4. Adjust augmentations: edit `data/augmentations.py` to change training transforms.
5. Collator / processor: collators (`data/collators*.py`) call the model `processor` to convert PIL images into tensors; you normally don't need to edit them unless you change the pipeline.

Example commands

Train ViT:

```bash
python train_vit.py
```

Train EfficientNet:

```bash
python train_efficientnet.py
```

Notes and tips
- Quick testing: `train_efficientnet.py` contains a commented block that reduces the dataset size (`shuffle().select(range(...))`) for fast trials.
- Resume training: to resume from a checkpoint, enable `resume_from_checkpoint` in the `trainer.train(...)` call (see `train_efficientnet.py`).
- Outputs: run artifacts are saved under `resultados_vit/` or `resultados_efficientnet/` and include `all_results.json`, `eval_results.json`, `classification_report.txt`, `misclassified/`, and the saved model.
- GPU: Hugging Face `Trainer` uses GPU if available (and configured). Ensure you have a compatible `torch` + CUDA installation.
- Normalization: model processors already apply resizing and normalization; augmentations in `data/augmentations.py` operate on PIL images before the processor step.

Common parameters to change
- `DATA_PATH`: dataset root path.
- `MODEL_NAME`: top of each training script.
- `training/trainer_args.py`: learning rate, epochs, batch size, etc.
- `data/augmentations.py`: input size, crop, flips, color jitter, etc.

Suggested improvements
- Add a CLI wrapper to choose model and override args at runtime.
- Integrate experiment tracking (Weights & Biases or TensorBoard).

---
See `train_vit.py` and `train_efficientnet.py` for concrete configuration examples and inline notes.

Automated setup and run script
--------------------------------
This repository includes a convenience script `setup_and_run.sh` that automates common setup tasks and launches training. Use it when you want a reproducible, one-command setup on a clean machine. Summary:

- What it does (high level):
	- updates the OS packages and installs basic tools (`wget`, `git`, `unzip`)
	- installs Miniconda (if not present) and creates/activates a Conda env named `vit_env`
	- installs Python dependencies (PyTorch, Transformers, Datasets, Pillow, etc.) — the script uses a sentinel file `.dependencies_installed` to avoid reinstalling on subsequent runs
	- synchronizes the dataset from a GCS bucket using `gsutil rsync` (idempotent)
	- runs the training command (`python train_vit.py` by default; `train_efficientnet.py` is available but commented)

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
	- To run `train_efficientnet.py` instead, uncomment that line and comment `train_vit.py`.
	- To use a different dataset source, replace the `gsutil rsync` command with your copy/sync command.

If you want, I can add a safer CLI wrapper around the script so you can select steps (install-only, sync-only, train-only) interactively.
