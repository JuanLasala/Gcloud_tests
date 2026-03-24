import os
import torch
from datetime import datetime

from data.dataset_loader import load_imagefolder
from data.augmentations import train_augmentations
from data.collators import ImageCollator

from models.vit_factory import build_vit
from training.metrics import compute_metrics
from training.trainer_args import get_training_args

from utils.save_errors import save_misclassified_images
from utils.grad_cam_vit import create_gradcam_for_misclassified

from utils.loss_plotter import plot_learning_curves
from utils.plots import plot_confusion, save_classification_report
from utils.threshold_tuning import find_optimal_threshold, apply_threshold
import numpy as np
import json
import threading
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix

from transformers import Trainer, EarlyStoppingCallback
import torch.nn.functional as F


# ==========================================
# CARGAR DATASET
# ==========================================
DATA_PATH = "/home/jlasala/ViT tests"
ds = load_imagefolder(DATA_PATH)

# ==========================================
# CREAR MODELO VIT + PROCESSOR
# ==========================================
labels = ds["train"].features["label"].names # nombres de las clases en el orden del dataset
print("labels (dataset order):", labels)

id2label = {i: label for i, label in enumerate(labels)} # Mapeo ID a label ({0: 'Fire', 1: 'No_Fire'})
label2id = {label: i for i, label in enumerate(labels)} # Mapeo LABEL A ID ({"Fire": 0, "No_Fire": 1})

fire_index = labels.index("Fire")
no_fire_index = labels.index("No_Fire")

model, processor = build_vit(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels), # número de clases
    id2label=id2label,
    label2id=label2id
)

# ==========================================
# TRANSFORMS (SE USAN CON .WITH_TRANSFORM)
# ==========================================
def train_transform(batch):
    images = [train_augmentations(img.convert("RGB")) for img in batch["image"]]
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = batch["label"]
    return inputs

def eval_transform(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = batch["label"]
    return inputs

ds_transf = {
    "train": ds["train"].with_transform(train_transform),
    "val": ds["val"].with_transform(eval_transform),
    "test": ds["test"].with_transform(eval_transform),
}

# ==========================================
# TRAINER
# ==========================================


run_name = datetime.now().strftime("vit_run_%Y-%m-%d_%H-%M-%S")
output_dir = f"./resultados_vit/{run_name}"

training_args = get_training_args(output_dir)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_transf["train"],
    eval_dataset=ds_transf["val"],
    data_collator=ImageCollator(),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
)

# ==========================================
# ENTRENAR
# ==========================================
train_results = trainer.train()
trainer.save_model()

# ==========================================
# EVALUAR
# ==========================================
metrics = trainer.evaluate(ds_transf["val"])
trainer.save_metrics("eval", metrics)

# ==========================================
# GUARDAR IMÁGENES MAL CLASIFICADAS
# ==========================================
fp_count, fn_count, fp_paths, fn_paths = save_misclassified_images(
    model, processor, ds["val"], output_dir=f"{output_dir}/misclassified", fire_index=fire_index, no_fire_index=no_fire_index
)

create_gradcam_for_misclassified(
    model, processor, fp_paths, fn_paths, output_dir=f"{output_dir}/misclassified"
)


# ==========================================
# THRESHOLD TUNING (en val set)
# ==========================================
print("\n=== Sintonizando umbral (threshold) en validación ===")
val_preds = trainer.predict(ds_transf["val"])
val_logits = val_preds.predictions
val_labels = val_preds.label_ids

optimal_threshold, threshold_metrics = find_optimal_threshold(
    val_logits,
    val_labels,
    fire_index=fire_index,
    metric="f1_weighted",
    beta=2.0,
)
print(f"Umbral óptimo: {optimal_threshold:.4f}")
print(f"  - Precisión: {threshold_metrics['precision']:.4f}")
print(f"  - Recall: {threshold_metrics['recall']:.4f}")
print(f"  - F1: {threshold_metrics['f1']:.4f}")
print(f"  - TP: {threshold_metrics['true_positives']}, FP: {threshold_metrics['false_positives']}")
print(f"  - FN: {threshold_metrics['false_negatives']}, TN: {threshold_metrics['true_negatives']}\n")


# ==========================================
# (Removed repeated threshold-based evaluation on validation set)
# Only do final evaluation and plots on test set below
plot_learning_curves(trainer.state.log_history, output_dir)

# ==========================================
# GUARDAR UMBRAL ÓPTIMO
# ==========================================
threshold_info = {
    "optimal_threshold": float(optimal_threshold),
    "fire_index": int(fire_index),
    "metrics": threshold_metrics,
}
file_lock = threading.Lock()
with file_lock:
    with open(os.path.join(output_dir, "optimal_threshold.json"), "w") as f:
        json.dump(threshold_info, f, indent=2)
print(f"Umbral guardado en: {os.path.join(output_dir, 'optimal_threshold.json')}\n")

# ==========================================
# FINAL TEST EVALUATION (using optimal threshold from validation)
# ==========================================
print("\n=== Evaluación final en test set ===")
if "test" in ds_transf:
    test_preds = trainer.predict(ds_transf["test"])
    test_logits = test_preds.predictions
    test_labels = test_preds.label_ids
    y_test_pred = apply_threshold(test_logits, fire_index, optimal_threshold)
    y_test_true = test_labels

    # --- Compute and save confusion details and misclassified names ---
    test_cm = confusion_matrix(y_test_true, y_test_pred, labels=[fire_index, no_fire_index])
    tn, fp, fn, tp = 0, 0, 0, 0
    if test_cm.shape == (2,2):
        tn, fp, fn, tp = test_cm.ravel()
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_test_true, y_test_pred)):
        if true != pred:
            misclassified.append(ds["test"][i]["path"])
    confusion_details = {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'misclassified': misclassified
    }
    with open(os.path.join(output_dir, "test_confusion_details.json"), "w") as f:
        json.dump(confusion_details, f, indent=2)
    print(f"Test confusion details saved in: {os.path.join(output_dir, 'test_confusion_details.json')}")

    plot_confusion(y_test_true, y_test_pred, labels, output_dir)
    print('Test confusion matrix saved')
    save_classification_report(y_test_true, y_test_pred, labels, output_dir)
    print('Test classification report saved')
    test_precision = precision_score(y_test_true, y_test_pred, average='weighted')
    test_recall = recall_score(y_test_true, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test_true, y_test_pred, average='weighted')
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    try:
        if len(set(y_test_true)) == 2:
            if test_logits.ndim > 1 and test_logits.shape[1] == 2:
                row_sums = np.sum(test_logits, axis=1)
                if np.allclose(row_sums, 1.0, atol=1e-3):
                    print("[WARNING] test_logits rows sum to ~1. Probabilities may already be normalized. Skipping softmax.")
                    test_probs = test_logits[:, fire_index]
                else:
                    test_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()[:, fire_index]
            elif test_logits.ndim == 1:
                test_probs = 1 / (1 + np.exp(-test_logits))
            else:
                test_probs = test_logits[:, fire_index]
            test_roc_auc = roc_auc_score(y_test_true, test_probs)
            print("USED CORRECT ROC_AUC CALCULATION FOR BINARY")
        else:
            test_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()
            test_roc_auc = roc_auc_score(y_test_true, test_probs, multi_class='ovr')
    except Exception as e:
        test_roc_auc = None
    test_metrics = {
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'accuracy': test_accuracy,
        'roc_auc': test_roc_auc,
    }
    file_lock = threading.Lock()
    with file_lock:
        with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
    print(f"Test metrics saved in: {os.path.join(output_dir, 'test_metrics.json')}")
else:
    print("No test split found in dataset. Skipping test evaluation.")

print("Entrenamiento completado.")
