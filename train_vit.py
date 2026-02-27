import os
import argparse
from datetime import datetime

from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from transformers import Trainer, EarlyStoppingCallback

from data.dataset_loader import load_imagefolder
from data.multiband_tiff import load_multiband_tiff
from data.collators_efficientnet import EfficientNetCollator
from models.model_loader import load_hf_model
from training.metrics import compute_metrics
from training.trainer_args import get_training_args
from utils.efficientnet_helpers import build_multiband_transforms, apply_effnet_transforms
from utils.save_errors import save_misclassified_images
from utils.loss_plotter import plot_learning_curves
from utils.plots import plot_confusion, save_classification_report


MODEL_NAME = "google/vit-base-patch16-384"
RESULTS_BASE = "./resultados_vit"
DATA_PATH = "/srv/train_project/Gcloud_tests/dataset"
TARGET_CHANNELS = 6


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT with multiband TIFF preprocessing.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run a quick test with reduced train/validation subsets.",
    )
    return parser.parse_args()


args = parse_args()
ds = load_imagefolder(DATA_PATH)

labels = ds["train"].features["label"].names # nombres de las clases en el orden del dataset
print("labels (dataset order):", labels)

id2label = {i: label for i, label in enumerate(labels)} # Mapeo ID a label ({0: 'Fire', 1: 'No_Fire'})
label2id = {label: i for i, label in enumerate(labels)} # Mapeo LABEL A ID ({"Fire": 0, "No_Fire": 1})

fire_index = labels.index("Fire")
no_fire_index = labels.index("No_Fire")

if args.test_run:
    num_samples = 100
    num_val_samples = 20
    print(f"!!! EJECUTANDO PRUEBA RÁPIDA: Reduciendo datasets a {num_samples} train y {num_val_samples} val !!!")
    ds["train"] = ds["train"].shuffle(seed=42).select(range(num_samples))
    ds["validation"] = ds["validation"].shuffle(seed=42).select(range(num_val_samples))

model, processor = load_hf_model(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    in_channels=TARGET_CHANNELS,
)

train_augmentations_multiband_vit = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),
    v2.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR, fill=0),
])

eval_augmentations_multiband_vit = v2.Compose([
    v2.Resize((384, 384), interpolation=InterpolationMode.BILINEAR, antialias=True),
])

train_transform_vit, eval_transform_vit = build_multiband_transforms(
    TARGET_CHANNELS,
    train_augmentations_multiband_vit,
    eval_augmentations_multiband_vit,
    load_multiband_tiff,
    force_output_size=(384, 384),
)
ds_transf = apply_effnet_transforms(ds, train_transform_vit, eval_transform_vit)

run_name = datetime.now().strftime("vit_run_%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(RESULTS_BASE, run_name)
os.makedirs(output_dir, exist_ok=True)

training_args = get_training_args(output_dir)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_transf["train"],
    eval_dataset=ds_transf["validation"],
    data_collator=EfficientNetCollator(processor=None),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
)

# ==========================================
# ENTRENAR
# ==========================================
train_results = trainer.train()
trainer.save_model(os.path.join(output_dir, "best_model"))

# ==========================================
# EVALUAR
# ==========================================
metrics = trainer.evaluate(ds_transf["validation"])
trainer.save_metrics("eval", metrics)

# ==========================================
# GUARDAR IMÁGENES MAL CLASIFICADAS
# ==========================================
fp_count, fn_count, fp_paths, fn_paths = save_misclassified_images(
    model, ds["validation"], output_dir=f"{output_dir}/misclassified", fire_index=fire_index, no_fire_index=no_fire_index
)

# ==========================================
# PLOTS
# ==========================================
preds = trainer.predict(ds_transf["validation"])
y_pred = preds.predictions.argmax(axis=1)
y_true = preds.label_ids

plot_confusion(y_true, y_pred, labels, output_dir)
save_classification_report(y_true, y_pred, labels, output_dir)
plot_learning_curves(trainer.state.log_history, output_dir)

print("Entrenamiento completado.")
