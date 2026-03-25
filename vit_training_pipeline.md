# ViT Training Pipeline Documentation

This document outlines the steps involved in training a Vision Transformer (ViT) model in this project, referencing the relevant methods and scripts used throughout the pipeline.

## 1. Dataset Loading

- **Script/Module:** `data/dataset_loader.py`
- **Key Methods:**
  - `load_dataset()` (or similar function for loading datasets)
- **Description:**
  - Loads and prepares the dataset for training and validation. Handles data splits and preprocessing.

## 2. Data Augmentation & Collation

- **Script/Module:** `data/augmentations.py`, `data/collators.py`
- **Key Methods:**
  - Augmentation functions (e.g., random crops, flips, normalization)
  - Collator classes/functions for ViT
- **Description:**
  - Applies data augmentation techniques and prepares batches for the model.

## 3. Model Loading

- **Script/Module:** `models/model_loader.py`
- **Key Methods:**
  - `load_vit_model()` (or similar)
- **Description:**
  - Loads the ViT architecture, optionally with pretrained weights.

## 4. Training Setup

- **Script/Module:** `train_vit.py`, `training/trainer_args.py`
- **Key Methods:**
  - Argument parsing for training configuration
  - Training loop setup
- **Description:**
  - Configures training parameters (epochs, batch size, learning rate, etc.).

## 5. Training Loop

- **Script/Module:** `train_vit.py`, `training/metrics.py`
- **Key Methods:**
  - Training and validation steps
  - Metric calculation (accuracy, loss, etc.)
- **Description:**
  - Runs the main training loop, evaluates on validation set, and tracks metrics.

## 6. Model Evaluation & Testing

- **Script/Module:** `train_vit.py`, `training/metrics.py`
- **Key Methods:**
  - Final evaluation on test set
  - Metric reporting
- **Description:**
  - Evaluates the trained model on the test set and reports final metrics.

## 7. Saving Results

- **Script/Module:** `utils/save_errors.py`, `utils/loss_plotter.py`, `utils/plots.py`
- **Key Methods:**
  - Functions for saving error cases, plotting loss curves, and visualizing results
- **Description:**
  - Saves model checkpoints, error cases, and generates plots for analysis.

---

**Note:** For detailed usage, refer to the docstrings and comments within each script/module. The main entry point for training is typically `train_vit.py`.
