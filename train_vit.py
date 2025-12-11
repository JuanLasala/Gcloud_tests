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

from transformers import Trainer, EarlyStoppingCallback
import torch.nn.functional as F


# ==========================================
# 1. Cargar dataset
# ==========================================
DATA_PATH = "/home/jlasala/ViT tests"
ds = load_imagefolder(DATA_PATH)

# ==========================================
# 2. Crear modelo ViT + processor
# ==========================================
labels = ds["train"].features["label"].names
print("labels (dataset order):", labels)

id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

print("id2label:", id2label)
print("label2id:", label2id)

fire_index = labels.index("Fire")
no_fire_index = labels.index("No_Fire")

model, processor = build_vit(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# ==========================================
# 3. Transforms (se usan con .with_transform)
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
# 4. Trainer
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
# 5. Entrenar
# ==========================================
train_results = trainer.train()
trainer.save_model()

# ==========================================
# 6. Evaluar
# ==========================================
metrics = trainer.evaluate(ds_transf["val"])
trainer.save_metrics("eval", metrics)

# ==========================================
# 7. Guardar imágenes mal clasificadas
# ==========================================
fp_count, fn_count, fp_paths, fn_paths = save_misclassified_images(
    model, processor, ds["val"], output_dir=f"{output_dir}/misclassified", fire_index=fire_index, no_fire_index=no_fire_index
)

create_gradcam_for_misclassified(
    model, processor, fp_paths, fn_paths, output_dir=f"{output_dir}/misclassified"
)

# ==========================================
# 8. Plots
# ==========================================
preds = trainer.predict(ds_transf["val"])
y_pred = preds.predictions.argmax(axis=1)
y_true = preds.label_ids

plot_confusion(y_true, y_pred, labels, output_dir)
save_classification_report(y_true, y_pred, labels, output_dir)
plot_learning_curves(trainer.state.log_history, output_dir)

print("Entrenamiento completado.")
