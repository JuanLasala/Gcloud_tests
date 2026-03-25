import os
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from torchvision import transforms
from collections import Counter
from transformers import EarlyStoppingCallback
import argparse

# --- módulos propios ---
from models.model_loader import load_hf_model
from data.dataset_loader import load_imagefolder
from data.augmentations import train_augmentations
from data.collators_efficientnet import EfficientNetCollator
from training.metrics import compute_metrics
from training.trainer_args import get_training_args
from utils.save_errors import save_misclassified_images
from utils.grad_cam_efficientnet import create_gradcam_for_misclassified
from utils.loss_plotter import plot_learning_curves
from utils.plots import plot_confusion, save_classification_report
from utils.list_FP import inspect_fp

# ---------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT with optional checkpoint resume.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument

    parser.add_argument(
            "--test-run",
            action="store_true",
            help="Run a quick test with reduced train/validation subsets.",
        )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Use RGB mode (3 channels) instead of multiband mode.",
    )
    return parser.parse_args()

args = parse_args()

if args.rgb:
    DATA_PATH = "/srv/train_project/Gcloud_tests/dataset_rgb"
    TARGET_CHANNELS = 3
else:
    DATA_PATH = "/srv/train_project/Gcloud_tests/dataset"
    TARGET_CHANNELS = 6

MODEL_NAME = "google/efficientnet-b4"
#MODEL_NAME = "google/efficientnet-v2-s" #modelo a elegir (en este caso EfficientNet-V2-S)
RESULTS_BASE = "./resultados_efficientnet" #directorio para guardar resultados

RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = os.path.join(RESULTS_BASE, f"old_efficientnet_run_{RUN_ID}")
os.makedirs(RUN_DIR, exist_ok=True)

print(f"\n=== Entrenamiento EfficientNet-V2 ===")
print(f"Modelo base: {MODEL_NAME}")
print(f"Guardando resultados en: {RUN_DIR}\n")

# ---------------------------------------------------------------------
# CARGA DEL DATASET
# ---------------------------------------------------------------------


DATA_PATH = "/srv/train_project/Gcloud_tests/dataset_rgb"


ds = load_imagefolder(DATA_PATH)


# =================================================================
# PRUEBA RÁPIDA CON DATASET REDUCIDO
# =================================================================
if args.test_run and ds is not None:
    NUM_SAMPLES = 100
    NUM_VAL_SAMPLES = 20
    print(f"!!! EJECUTANDO PRUEBA RÁPIDA: Reduciendo datasets a {NUM_SAMPLES} train y {NUM_VAL_SAMPLES} val !!!")

    # Crear subconjuntos pequeños (aseguramos que sea aleatorio y reproducible con shuffle)
    ds["train"] = ds["train"].shuffle(seed=42).select(range(NUM_SAMPLES))
    ds['validation'] = ds['validation'].shuffle(seed=42).select(range(NUM_VAL_SAMPLES))
    ds['test'] = ds['test'].shuffle(seed=42).select(range(NUM_VAL_SAMPLES))

# ---------------------------------------------------------------------
# MAPEOS DE CLASES
# ---------------------------------------------------------------------
labels = ds["train"].features["label"].names # nombres de las clases en el orden del dataset
print("labels (dataset order):", labels)

id2label = {i: label for i, label in enumerate(labels)} # mapeo id a label ({0: 'Fire', 1: 'No_Fire'})
label2id = {label: i for i, label in enumerate(labels)} # mapeo label a id ({"Fire": 0, "No_Fire": 1})

print("id2label:", id2label)
print("label2id:", label2id)

labels_lower = [l.lower() for l in labels]
fire_index = labels_lower.index("fire")
no_fire_index = labels_lower.index("no_fire")

# ---------------------------------------------------------------------
# CARGA DEL MODELO
# ---------------------------------------------------------------------
print("\nCargando modelo y processor...")
model, processor = load_hf_model(
    MODEL_NAME,
    num_labels=len(id2label), # número de clases
    id2label=id2label,
    label2id=label2id 
)


sample = ds["val"][0]["image"].convert("RGB") #toma la primera imagen del dataset de validación y la convierte a RGB
inputs = processor(images=sample, return_tensors="pt") # procesamiento de la imagen por el processor
print("Processor output shape:", inputs["pixel_values"].shape) 

# ---------------------------------------------------------------------
# COLLATOR (procesamiento por batch)
# ---------------------------------------------------------------------
print("Creando data collator...")
collator = EfficientNetCollator(processor) #

# -------------------------------------------------------------------------
#TRANSFORMS 
# -------------------------------------------------------------------------
print("Definiendo transforms...")
# normalización para efficientnet
effnet_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
def train_transform_effnet(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    images = [train_augmentations(img) for img in images]   # SOLO PIL transforms
    return {"image": images, "label": batch["label"], "path": batch["path"]}

def eval_transform_effnet(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    return {"image": images, "label": batch["label"], "path": batch["path"]}
ds = {
    "train": ds["train"].with_transform(train_transform_effnet),
    "validation": ds["val"].with_transform(eval_transform_effnet),
    "test": ds["test"].with_transform(eval_transform_effnet),
}
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
    processing_class=processor,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=8,
        early_stopping_threshold=0.001,)]
)
# ---------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------
print("\n=== Iniciando entrenamiento ===")

#PATH_TO_RESUME = "./resultados_efficientnet/efficientnet_run_2025-12-09_19-51-14"
#train_output = trainer.train(resume_from_checkpoint=PATH_TO_RESUME)
train_start = datetime.now()
train_output = trainer.train()
trainer.save_model(os.path.join(RUN_DIR, "best_model"))

# Curva de pérdidas
plot_learning_curves(trainer.state.log_history, RUN_DIR)

print("\n=== Evaluación final ===")
metrics = trainer.evaluate()
print(metrics)

# -------------------------------------------------------------------------
# EVALUACIÓN
# -------------------------------------------------------------------------

# Final evaluation on test set
print("\n=== Evaluación final en test set ===")
test_metrics = trainer.evaluate(ds["test"])
print(test_metrics)

# Save test metrics and generate plots/reports for test set
trainer.save_metrics("test", test_metrics)

# IMÁGENES MAL CLASIFICADAS (test set)
fp_count, fn_count, fp_paths, fn_paths = save_misclassified_images(
    model, processor, ds["test"], output_dir=f"{RUN_DIR}/misclassified_test", fire_index=fire_index, no_fire_index=no_fire_index
)
create_gradcam_for_misclassified(
    model, processor, fp_paths, fn_paths, output_dir=f"{RUN_DIR}/misclassified_test"
)

# PLOTS (test set)
test_preds = trainer.predict(ds["test"])
y_pred_test = test_preds.predictions.argmax(axis=1)
y_true_test = test_preds.label_ids
plot_confusion(y_true_test, y_pred_test, labels, RUN_DIR, suffix='_test')
save_classification_report(y_true_test, y_pred_test, labels, RUN_DIR, suffix='_test')
plot_learning_curves(trainer.state.log_history, RUN_DIR)
print('Test confusion, report, and learning curves done')


# -------------------------------------------------------------------------
# IMÁGENES MAL CLASIFICADAS
# -------------------------------------------------------------------------

# Only use validation for early stopping/model selection, not for final plots/logs
training_end = datetime.now()
training_duration = training_end - train_start
total_seconds = int(training_duration.total_seconds())
hrs, rem = divmod(total_seconds, 3600)
mins, secs = divmod(rem, 60)
print(f"Entrenamiento completado. Duración del entrenamiento: {hrs}h {mins}m {secs}s")