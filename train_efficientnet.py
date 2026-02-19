import os
import argparse
import torch
from datetime import datetime
from transformers import Trainer


# --- módulos propios ---
from models.model_loader import load_hf_model
from data.dataset_loader import load_imagefolder
from data.augmentations import train_augmentations_multiband, eval_augmentations_multiband
from data.multiband_tiff import load_multiband_tiff
from data.collators_efficientnet import EfficientNetCollator
from training.metrics import compute_metrics
from training.trainer_args import get_training_args
from utils.efficientnet_helpers import build_multiband_transforms, apply_effnet_transforms
from utils.save_errors import save_misclassified_images
from utils.grad_cam_efficientnet import create_gradcam_for_misclassified
from utils.loss_plotter import plot_learning_curves
from utils.plots import plot_confusion, save_classification_report
from utils.list_FP import inspect_fp
from utils.threshold_tuning import find_optimal_threshold, apply_threshold

# ---------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------

# Modelo a elegir:
# - "torchvision/efficientnet_v2_s" (EfficientNetV2-S)
# - "google/efficientnet-b4" (switch back to HF EfficientNet-B4)
MODEL_NAME = "torchvision/efficientnet_v2_s"
RESULTS_BASE = "./resultados_efficientnet" #directorio para guardar resultados

def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet with optional checkpoint resume.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Checkpoint directory to resume from (e.g. .../checkpoint-1200) or run directory "
            "containing checkpoints. If omitted, starts a new run."
        ),
    )
    resume_group.add_argument(
        "--auto_resume_last",
        action="store_true",
        help=(
            "Automatically resume from the latest run directory under resultados_efficientnet "
            "(picks latest checkpoint-* inside that run)."
        ),
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Optional output run directory. Useful to continue writing into an existing run folder.",
    )
    return parser.parse_args()


def find_latest_checkpoint(path):
    if not os.path.isdir(path):
        return None

    direct_trainer_checkpoint_files = {
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }

    if direct_trainer_checkpoint_files.issubset(set(os.listdir(path))):
        return path

    checkpoints = []
    for entry in os.listdir(path):
        if not entry.startswith("checkpoint-"):
            continue
        checkpoint_path = os.path.join(path, entry)
        if not os.path.isdir(checkpoint_path):
            continue
        step_str = entry.replace("checkpoint-", "")
        if step_str.isdigit():
            checkpoints.append((int(step_str), checkpoint_path))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def find_latest_run_dir(results_base):
    if not os.path.isdir(results_base):
        return None

    run_dirs = []
    for entry in os.listdir(results_base):
        full_path = os.path.join(results_base, entry)
        if os.path.isdir(full_path) and entry.startswith("efficientnet_run_"):
            run_dirs.append(full_path)

    if not run_dirs:
        return None

    run_dirs.sort(key=lambda path: os.path.getmtime(path))
    return run_dirs[-1]


args = parse_args()

resume_source = args.resume_from
if args.auto_resume_last:
    latest_run = find_latest_run_dir(RESULTS_BASE)
    if latest_run is None:
        raise ValueError(
            f"No run directories found in '{RESULTS_BASE}'. "
            "Expected folders like efficientnet_run_YYYY-mm-dd_HH-MM-SS."
        )
    resume_source = latest_run

RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if args.run_dir:
    RUN_DIR = args.run_dir
elif resume_source:
    resume_abs = os.path.abspath(resume_source)
    if os.path.basename(resume_abs).startswith("checkpoint-"):
        RUN_DIR = os.path.dirname(resume_abs)
    else:
        RUN_DIR = resume_abs
else:
    RUN_DIR = os.path.join(RESULTS_BASE, f"efficientnet_run_{RUN_ID}")

os.makedirs(RUN_DIR, exist_ok=True)

RESUME_CHECKPOINT = None
if resume_source:
    RESUME_CHECKPOINT = find_latest_checkpoint(resume_source)
    if RESUME_CHECKPOINT is None:
        raise ValueError(
            f"No valid checkpoint found in '{resume_source}'. "
            "Pass a checkpoint dir or a run dir containing checkpoint-* folders."
        )

print(f"\n=== Entrenamiento EfficientNet-V2 ===")
print(f"Modelo base: {MODEL_NAME}")
print(f"Guardando resultados en: {RUN_DIR}\n")
if RESUME_CHECKPOINT:
    print(f"Reanudando desde checkpoint: {RESUME_CHECKPOINT}\n")

# ---------------------------------------------------------------------
# CARGA DEL DATASET
# ---------------------------------------------------------------------

DATA_PATH = "/srv/train_project/Gcloud_tests/dataset"
TARGET_CHANNELS = 11
USE_TORCH_COMPILE = False

ds = load_imagefolder(DATA_PATH)

"""
# =================================================================
# PRUEBA RÁPIDA CON DATASET REDUCIDO
# =================================================================
NUM_SAMPLES = 100
NUM_VAL_SAMPLES = 20
print(f"!!! EJECUTANDO PRUEBA RÁPIDA: Reduciendo datasets a {NUM_SAMPLES} train y {NUM_VAL_SAMPLES} val !!!")

# Crear subconjuntos pequeños (aseguramos que sea aleatorio y reproducible con shuffle)
ds["train"] = ds["train"].shuffle(seed=42).select(range(NUM_SAMPLES))
ds['validation'] = ds['validation'].shuffle(seed=42).select(range(NUM_VAL_SAMPLES))
"""

# ---------------------------------------------------------------------
# MAPEOS DE CLASES
# ---------------------------------------------------------------------
labels = ds["train"].features["label"].names # nombres de las clases en el orden del dataset
print("labels (dataset order):", labels)

id2label = {i: label for i, label in enumerate(labels)} # mapeo id a label ({0: 'Fire', 1: 'No_Fire'})
label2id = {label: i for i, label in enumerate(labels)} # mapeo label a id ({"Fire": 0, "No_Fire": 1})

print("id2label:", id2label)
print("label2id:", label2id)

fire_index = labels.index("Fire")
no_fire_index = labels.index("No_Fire")

# ---------------------------------------------------------------------
# CARGA DEL MODELO
# ---------------------------------------------------------------------
print("\nCargando modelo y processor...")
model, processor = load_hf_model(
    MODEL_NAME,
    num_labels=len(id2label), # número de clases
    id2label=id2label,
    label2id=label2id,
    in_channels=TARGET_CHANNELS,
)
if USE_TORCH_COMPILE and hasattr(torch, "compile"):
    model = torch.compile(model)

sample = load_multiband_tiff(ds['validation'][0]["path"], target_channels=TARGET_CHANNELS)
print("Sample multiband shape:", sample.shape)

# ---------------------------------------------------------------------
# COLLATOR (procesamiento por batch)
# ---------------------------------------------------------------------
print("Creando data collator...")
collator = EfficientNetCollator(processor=None)

# -------------------------------------------------------------------------
#TRANSFORMS 
# -------------------------------------------------------------------------
print("Definiendo transforms...")
train_transform_effnet, eval_transform_effnet = build_multiband_transforms(
    TARGET_CHANNELS,
    train_augmentations_multiband,
    eval_augmentations_multiband,
    load_multiband_tiff,
)
ds = apply_effnet_transforms(ds, train_transform_effnet, eval_transform_effnet)
# ---------------------------------------------------------------------
# TRAINING ARGUMENTS
# ---------------------------------------------------------------------
print("Definiendo training arguments...")
training_args = get_training_args(
    output_dir=RUN_DIR
)

# ---------------------------------------------------------------------
# TRAINER
# ---------------------------------------------------------------------
print("Creando Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds['validation'],
    data_collator=collator,
    compute_metrics=compute_metrics,
    processing_class=processor
)

# ---------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------
print("\n=== Iniciando entrenamiento ===")

train_start = datetime.now()
if RESUME_CHECKPOINT:
    train_output = trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)
else:
    train_output = trainer.train()
trainer.save_model(os.path.join(RUN_DIR, "best_model"))

# Curva de pérdidas
plot_learning_curves(trainer.state.log_history, RUN_DIR)

print("\n=== Evaluación final ===")
metrics = trainer.evaluate()
print(metrics)

# -------------------------------------------------------------------------
# THRESHOLD TUNING (en val set)
# -------------------------------------------------------------------------
print("\n=== Sintonizando umbral (threshold) en validación ===")
val_preds = trainer.predict(ds['validation'])
val_logits = val_preds.predictions
val_labels = val_preds.label_ids

optimal_threshold, threshold_metrics = find_optimal_threshold(
    val_logits,
    val_labels,
    fire_index=fire_index,
    metric="f1_weighted",
    beta=2.0,  # Enfatizar recall (reducir FN) 2x más que precision
)
print(f"Umbral óptimo: {optimal_threshold:.4f}")
print(f"  - Precisión: {threshold_metrics['precision']:.4f}")
print(f"  - Recall: {threshold_metrics['recall']:.4f}")
print(f"  - F1: {threshold_metrics['f1']:.4f}")
print(f"  - TP: {threshold_metrics['true_positives']}, FP: {threshold_metrics['false_positives']}")
print(f"  - FN: {threshold_metrics['false_negatives']}, TN: {threshold_metrics['true_negatives']}\n")

# -------------------------------------------------------------------------
# EVALUACIÓN
# -------------------------------------------------------------------------
trainer.save_metrics("eval", metrics)

# -------------------------------------------------------------------------
# IMÁGENES MAL CLASIFICADAS
# -------------------------------------------------------------------------
fp_count, fn_count, fp_paths, fn_paths = save_misclassified_images(
    model, processor, ds['validation'], output_dir=f"{RUN_DIR}/misclassified", fire_index=fire_index, no_fire_index=no_fire_index
)

create_gradcam_for_misclassified(
    model, processor, fp_paths, fn_paths, output_dir=f"{RUN_DIR}/misclassified"
)

# -------------------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------------------
# Usar predicciones con umbral óptimo en lugar de argmax
y_pred_threshold = apply_threshold(val_logits, fire_index, optimal_threshold)
y_true = val_labels
plot_confusion(y_true, y_pred_threshold, labels, RUN_DIR)
print('confusion done')
save_classification_report(y_true, y_pred_threshold, labels, RUN_DIR)
print('report done')
"""fps = inspect_fp(model, processor, ds['validation'], labels)
print("FOUND FP:", len(fps))
for r in fps[:10]:
    print(r)
"""
plot_learning_curves(trainer.state.log_history, RUN_DIR)
print('learning curves done')

# -------------------------------------------------------------------------
# GUARDAR UMBRAL ÓPTIMO
# -------------------------------------------------------------------------
import json
threshold_info = {
    "optimal_threshold": float(optimal_threshold),
    "fire_index": int(fire_index),
    "metrics": threshold_metrics,
}
with open(os.path.join(RUN_DIR, "optimal_threshold.json"), "w") as f:
    json.dump(threshold_info, f, indent=2)
print(f"Umbral guardado en: {os.path.join(RUN_DIR, 'optimal_threshold.json')}\n")

training_end = datetime.now()
training_duration = training_end - train_start
total_seconds = int(training_duration.total_seconds())
hrs, rem = divmod(total_seconds, 3600)
mins, secs = divmod(rem, 60)
print(f"Entrenamiento completado. Duración del entrenamiento: {hrs}h {mins}m {secs}s")
