import os
import argparse
import json
import torch
from datetime import datetime
from PIL import Image
from torchvision.transforms import functional as tvf
from transformers import Trainer
from transformers import EarlyStoppingCallback
import threading
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import torch.nn.functional as F


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
    parser.add_argument(
        "--generate_config_only",
        action="store_true",
        help=(
            "Generate config.json and preprocessor_config.json in best_model directory and exit "
            "without loading model weights or running training."
        ),
    )
    parser.add_argument(
        "--config_output_dir",
        type=str,
        default=None,
        help=(
            "Optional output directory for generated config artifacts. "
            "If omitted, uses <run_dir>/best_model."
        ),
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help=(
            "Optional comma-separated labels in class-index order (e.g. 'Fire,No_Fire'). "
            "If omitted, labels are read from dataset."
        ),
    )
    epochs_group = parser.add_mutually_exclusive_group()
    epochs_group.add_argument(
        "--resume_additional_epochs",
        type=float,
        default=0.0,
        help=(
            "Extra epochs to add when resuming. Useful if resumed checkpoint already reached "
            "num_train_epochs and would otherwise skip training."
        ),
    )
    epochs_group.add_argument(
        "--resume_to_total_epochs",
        type=float,
        default=None,
        help=(
            "Set an absolute total epoch target when resuming (e.g. 20). "
            "If checkpoint is already at/above this value, no extra training is run."
        ),
    )
    return parser.parse_args()


def find_latest_checkpoint(path):
    if not os.path.isdir(path):
        return None

    def is_resumeable_checkpoint(checkpoint_path):
        required_files = {
            "trainer_state.json",
            "optimizer.pt",
        }
        has_required = all(
            os.path.isfile(os.path.join(checkpoint_path, file_name))
            for file_name in required_files
        )
        has_model_weights = (
            os.path.isfile(os.path.join(checkpoint_path, "model.safetensors"))
            or os.path.isfile(os.path.join(checkpoint_path, "pytorch_model.bin"))
        )
        return has_required and has_model_weights

    if is_resumeable_checkpoint(path):
        return path

    checkpoints = []
    for entry in os.listdir(path):
        if not entry.startswith("checkpoint-"):
            continue
        checkpoint_path = os.path.join(path, entry)
        if not os.path.isdir(checkpoint_path):
            continue
        if not is_resumeable_checkpoint(checkpoint_path):
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


def read_checkpoint_progress(checkpoint_dir):
    state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.isfile(state_file):
        return None, None

    with open(state_file, "r") as state_handle:
        trainer_state = json.load(state_handle)

    epoch = trainer_state.get("epoch")
    global_step = trainer_state.get("global_step")

    epoch_value = float(epoch) if epoch is not None else None
    step_value = int(global_step) if global_step is not None else None
    return epoch_value, step_value


def parse_labels_csv(labels_csv):
    labels = [label.strip() for label in labels_csv.split(",") if label.strip()]
    if len(labels) < 2:
        raise ValueError("--labels_csv must include at least two labels (e.g. 'Fire,No_Fire').")
    return labels


def load_image_tensor(path: str, rgb_mode: bool) -> torch.Tensor:
    if not rgb_mode:
        return load_multiband_tiff(path)

    with Image.open(path) as pil_img:
        pil_img = pil_img.convert("RGB")
        return tvf.pil_to_tensor(pil_img).float() / 255.0


def ensure_inference_artifacts(best_model_dir, model, processor, model_name, target_channels, id2label, label2id):
    os.makedirs(best_model_dir, exist_ok=True)

    config_path = os.path.join(best_model_dir, "config.json")
    if not os.path.isfile(config_path):
        model_config = getattr(model, "config", None) if model is not None else None
        if model is not None:
            architecture_name = model.__class__.__name__
        elif model_name == "torchvision/efficientnet_v2_s":
            architecture_name = "TorchvisionEfficientNetForClassification"
        else:
            architecture_name = "AutoModelForImageClassification"

        config_payload = {
            "model_type": "efficientnet",
            "architectures": [architecture_name],
            "model_name": model_name,
            "num_labels": int(len(id2label)),
            "id2label": {str(index): str(label) for index, label in id2label.items()},
            "label2id": {str(label): int(index) for label, index in label2id.items()},
            "num_channels": int(target_channels),
            "problem_type": "single_label_classification",
        }

        if model_config is not None:
            for key in ["model_type", "num_labels", "num_channels"]:
                value = getattr(model_config, key, None)
                if value is not None:
                    config_payload[key] = value

            cfg_id2label = getattr(model_config, "id2label", None)
            if isinstance(cfg_id2label, dict) and cfg_id2label:
                config_payload["id2label"] = {
                    str(index): str(label) for index, label in cfg_id2label.items()
                }

            cfg_label2id = getattr(model_config, "label2id", None)
            if isinstance(cfg_label2id, dict) and cfg_label2id:
                config_payload["label2id"] = {
                    str(label): int(index) for label, index in cfg_label2id.items()
                }

        with open(config_path, "w") as config_handle:
            json.dump(config_payload, config_handle, indent=2)
        print(f"Config guardado en: {config_path}")

    preprocessor_config_path = os.path.join(best_model_dir, "preprocessor_config.json")
    if not os.path.isfile(preprocessor_config_path):
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(best_model_dir)
            print(f"Preprocessor guardado en: {preprocessor_config_path}")
        else:
            preprocessor_payload = {
                "processor_class": "CustomMultibandPreprocessor",
                "model_name": model_name,
                "num_channels": int(target_channels),
                "note": "This pipeline uses custom torchvision transforms instead of a HF processor.",
            }
            with open(preprocessor_config_path, "w") as preprocessor_handle:
                json.dump(preprocessor_payload, preprocessor_handle, indent=2)
            print(f"Preprocessor config guardado en: {preprocessor_config_path}")


def save_trainer_configuration_txt(
    best_model_dir,
    training_args,
    trainer,
    model_name,
    target_channels,
    use_rgb,
    resume_checkpoint,
):
    os.makedirs(best_model_dir, exist_ok=True)
    config_path = os.path.join(best_model_dir, "trainer_configuration.txt")

    def serialize_callback(callback):
        callback_info = {
            "class_name": callback.__class__.__name__,
        }

        for attr_name in [
            "early_stopping_patience",
            "early_stopping_threshold",
        ]:
            if hasattr(callback, attr_name):
                callback_info[attr_name] = getattr(callback, attr_name)

        return callback_info

    if hasattr(training_args, "to_dict"):
        training_args_dict = training_args.to_dict()
    else:
        training_args_dict = vars(training_args)

    trainer_config = {
        "model_name": model_name,
        "model_class": trainer.model.__class__.__name__,
        "best_model_dir": best_model_dir,
        "resume_checkpoint": resume_checkpoint,
        "use_rgb": bool(use_rgb),
        "target_channels": int(target_channels),
        "train_dataset_size": len(trainer.train_dataset) if trainer.train_dataset is not None else None,
        "eval_dataset_size": len(trainer.eval_dataset) if trainer.eval_dataset is not None else None,
        "data_collator": trainer.data_collator.__class__.__name__ if trainer.data_collator is not None else None,
        "compute_metrics": trainer.compute_metrics.__name__ if trainer.compute_metrics is not None else None,
        "callbacks": [serialize_callback(callback) for callback in trainer.callback_handler.callbacks],
    }

    with open(config_path, "w") as config_file:
        config_file.write("=== TrainingArguments ===\n")
        config_file.write(json.dumps(training_args_dict, indent=2, sort_keys=True, default=str))
        config_file.write("\n\n=== Trainer Configuration ===\n")
        config_file.write(json.dumps(trainer_config, indent=2, sort_keys=True, default=str))
        config_file.write("\n")

    print(f"Trainer config guardado en: {config_path}")


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
            "Pass a checkpoint dir or a run dir containing checkpoint-* folders with "
            "at least model weights, optimizer.pt, and trainer_state.json."
        )

print(f"\n=== Entrenamiento EfficientNet-V2 ===")
print(f"Modelo base: {MODEL_NAME}")
print(f"Guardando resultados en: {RUN_DIR}\n")
if RESUME_CHECKPOINT:
    print(f"Reanudando desde checkpoint: {RESUME_CHECKPOINT}\n")

# ---------------------------------------------------------------------
# CARGA DEL DATASET
# ---------------------------------------------------------------------

if args.rgb:
    DATA_PATH = "/srv/train_project/Gcloud_tests/dataset_rgb"
    TARGET_CHANNELS = 3
else:
    DATA_PATH = "/srv/train_project/Gcloud_tests/dataset"
    TARGET_CHANNELS = 6

USE_TORCH_COMPILE = False

print(f"Modo {'RGB' if args.rgb else 'multibanda'} activado. DATA_PATH={DATA_PATH}, canales={TARGET_CHANNELS}")

if args.labels_csv:
    labels = parse_labels_csv(args.labels_csv)
    ds = None
else:
    ds = load_imagefolder(DATA_PATH)

    train_columns = set(ds["train"].column_names)
    if "label" not in train_columns and "labels" in train_columns:
        print("[INFO] Renombrando columna 'labels' -> 'label'.")
        for split_name in ds.keys():
            split_columns = set(ds[split_name].column_names)
            if "labels" in split_columns and "label" not in split_columns:
                ds[split_name] = ds[split_name].rename_column("labels", "label")

    if "label" not in ds["train"].column_names:
        raise KeyError(
            "No se encontró columna de clases. Columnas disponibles en train: "
            f"{ds['train'].column_names}"
        )


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


# ---------------------------------------------------------------------
# MAPEOS DE CLASES
# ---------------------------------------------------------------------
if ds is not None:
    labels = ds["train"].features["label"].names # nombres de las clases en el orden del dataset
print("labels (dataset order):", labels)

id2label = {i: label for i, label in enumerate(labels)} # mapeo id a label ({0: 'Fire', 1: 'No_Fire'})
label2id = {label: i for i, label in enumerate(labels)} # mapeo label a id ({"Fire": 0, "No_Fire": 1})

print("id2label:", id2label)
print("label2id:", label2id)

if args.generate_config_only:
    if args.config_output_dir:
        best_model_dir = os.path.abspath(args.config_output_dir)
    else:
        best_model_dir = os.path.join(RUN_DIR, "best_model")

    ensure_inference_artifacts(
        best_model_dir=best_model_dir,
        model=None,
        processor=None,
        model_name=MODEL_NAME,
        target_channels=TARGET_CHANNELS,
        id2label=id2label,
        label2id=label2id,
    )
    print(f"Config-only mode completado. Artefactos generados en: {best_model_dir}")
    raise SystemExit(0)

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
    label2id=label2id,
    in_channels=TARGET_CHANNELS,
    use_rgb=args.rgb,
)
#if USE_TORCH_COMPILE and hasattr(torch, "compile"): # // DISABLE TORCH.COMPILE FOR NOW
    #model = torch.compile(model, mode="reduce-overhead")

sample = load_image_tensor(ds['validation'][0]["path"], args.rgb)
print("Sample tensor shape:", sample.shape)

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
    lambda p: load_image_tensor(p, args.rgb),
)
ds = apply_effnet_transforms(ds, train_transform_effnet, eval_transform_effnet)
# ---------------------------------------------------------------------
# TRAINING ARGUMENTS
# ---------------------------------------------------------------------
print("Definiendo training arguments...")
training_args = get_training_args(
    output_dir=RUN_DIR
)

if RESUME_CHECKPOINT:
    current_epoch, current_step = read_checkpoint_progress(RESUME_CHECKPOINT)
    if current_epoch is not None:
        print(f"Progreso del checkpoint: epoch={current_epoch:.4f}, global_step={current_step}")

    original_epochs = float(training_args.num_train_epochs)
    if args.resume_to_total_epochs is not None:
        target_total = float(args.resume_to_total_epochs)
        training_args.num_train_epochs = target_total
        print(
            "Usando objetivo total de epochs para reanudar: "
            f"num_train_epochs -> {training_args.num_train_epochs}"
        )
        if current_epoch is not None and current_epoch >= target_total:
            print(
                "[WARN] El checkpoint ya alcanzó/superó resume_to_total_epochs; "
                "entrenamiento adicional será 0."
            )
    elif args.resume_additional_epochs > 0:
        training_args.num_train_epochs = original_epochs + float(args.resume_additional_epochs)
        print(
            "Extendiendo num_train_epochs para continuar entrenamiento: "
            f"{original_epochs} -> {training_args.num_train_epochs}"
        )
    elif current_epoch is not None and current_epoch >= original_epochs:
        print(
            "[WARN] El checkpoint ya alcanzó num_train_epochs; entrenamiento adicional será 0. "
            "Usa --resume_additional_epochs N o --resume_to_total_epochs M para seguir entrenando."
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

save_trainer_configuration_txt(
    best_model_dir=os.path.join(RUN_DIR, "best_model"),
    training_args=training_args,
    trainer=trainer,
    model_name=MODEL_NAME,
    target_channels=TARGET_CHANNELS,
    use_rgb=args.rgb,
    resume_checkpoint=RESUME_CHECKPOINT,
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
best_model_dir = os.path.join(RUN_DIR, "best_model")
trainer.save_model(best_model_dir)
ensure_inference_artifacts(
    best_model_dir=best_model_dir,
    model=model,
    processor=processor,
    model_name=MODEL_NAME,
    target_channels=TARGET_CHANNELS,
    id2label=id2label,
    label2id=label2id,
)

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

# Usar exactamente las mismas predicciones threshold-based para métricas, plots y errores
y_pred_threshold = apply_threshold(val_logits, fire_index, optimal_threshold)
y_true = val_labels

# -------------------------------------------------------------------------
# IMÁGENES MAL CLASIFICADAS
# -------------------------------------------------------------------------
fp_count, fn_count, fp_paths, fn_paths = save_misclassified_images(
    model,
    ds['validation'],
    output_dir=f"{RUN_DIR}/misclassified",
    fire_index=fire_index,
    no_fire_index=no_fire_index,
    pred_labels=y_pred_threshold,
    true_labels=y_true,
    use_rgb=args.rgb,
)

create_gradcam_for_misclassified(
    model, processor, fp_paths, fn_paths, output_dir=f"{RUN_DIR}/misclassified", use_rgb=args.rgb
)

# -------------------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------------------
# Usar predicciones con umbral óptimo en lugar de argmax
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
threshold_info = {
    "optimal_threshold": float(optimal_threshold),
    "fire_index": int(fire_index),
    "metrics": threshold_metrics,
}
file_lock = threading.Lock()
with file_lock:
    with open(os.path.join(RUN_DIR, "optimal_threshold.json"), "w") as f:
        json.dump(threshold_info, f, indent=2)
print(f"Umbral guardado en: {os.path.join(RUN_DIR, 'optimal_threshold.json')}\n")


# -------------------------------------------------------------------------
# FINAL TEST EVALUATION (using optimal threshold from validation)
# ------------------------------------------------------------------------- 
print("\n=== Evaluación final en test set ===")
if 'test' in ds:
    test_preds = trainer.predict(ds['test'])
    test_logits = test_preds.predictions
    test_labels = test_preds.label_ids
    y_test_pred = apply_threshold(test_logits, fire_index, optimal_threshold)
    y_test_true = test_labels

    # --- Compute and save confusion details and misclassified names ---
    from sklearn.metrics import confusion_matrix
    test_cm = confusion_matrix(y_test_true, y_test_pred, labels=[fire_index, no_fire_index])
    # test_cm: rows=true, cols=pred; for binary: [[TP, FN], [FP, TN]] if order is [fire, no_fire]
    # But sklearn confusion_matrix returns: [[TN, FP], [FN, TP]] for binary if labels=[0,1]
    # Let's map indices for clarity
    tn, fp, fn, tp = 0, 0, 0, 0
    if test_cm.shape == (2,2):
        tn, fp, fn, tp = test_cm.ravel()
    # Find misclassified image names
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_test_true, y_test_pred)):
        if true != pred:
            # ds['test'][i]['path'] should exist due to load_imagefolder
            misclassified.append(ds['test'][i]['path'])
    # Save to JSON
    confusion_details = {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'misclassified': misclassified
    }
    with open(os.path.join(RUN_DIR, "test_confusion_details.json"), "w") as f:
        json.dump(confusion_details, f, indent=2)
    print(f"Test confusion details saved in: {os.path.join(RUN_DIR, 'test_confusion_details.json')}")

    # Métricas y reportes en test
    plot_confusion(y_test_true, y_test_pred, labels, RUN_DIR)
    print('Test confusion matrix saved')
    save_classification_report(y_test_true, y_test_pred, labels, RUN_DIR)
    print('Test classification report saved')
    # Guardar métricas principales
    test_precision = precision_score(y_test_true, y_test_pred, average='weighted')
    test_recall = recall_score(y_test_true, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test_true, y_test_pred, average='weighted')
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    # Compute ROC AUC (only if binary or multilabel with proper shape)
    try:
        if len(set(y_test_true)) == 2:
            # For binary, use probabilities for positive class
            if test_logits.ndim > 1 and test_logits.shape[1] == 2:
                # Check if logits are already normalized (sum close to 1)
                row_sums = np.sum(test_logits, axis=1)
                if np.allclose(row_sums, 1.0, atol=1e-3):
                    print("[WARNING] test_logits rows sum to ~1. Probabilities may already be normalized. Skipping softmax.")
                    test_probs = test_logits[:, fire_index]
                else:
                    # Apply softmax to get probabilities
                    test_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()[:, fire_index]
            elif test_logits.ndim == 1:
                # If already 1D, apply sigmoid
                test_probs = 1 / (1 + np.exp(-test_logits))
            else:
                test_probs = test_logits[:, fire_index]  # fallback
            test_roc_auc = roc_auc_score(y_test_true, test_probs)
            print("USED CORRECT ROC_AUC CALCULATION FOR BINARY")
        else:
            # For multiclass, use softmax probabilities
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
    # Lock file writing to avoid tqdm interference
    file_lock = threading.Lock()
    with file_lock:
        with open(os.path.join(RUN_DIR, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
    print(f"Test metrics saved in: {os.path.join(RUN_DIR, 'test_metrics.json')}")
else:
    print("No test split found in dataset. Skipping test evaluation.")

training_end = datetime.now()
training_duration = training_end - train_start
total_seconds = int(training_duration.total_seconds())
hrs, rem = divmod(total_seconds, 3600)
mins, secs = divmod(rem, 60)
print(f"Entrenamiento completado. Duración del entrenamiento: {hrs}h {mins}m {secs}s")