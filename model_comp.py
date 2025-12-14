import time
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar   # pip install statsmodels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIG: cambialo por tus paths y loaders ---
#PATH_VIT = "resultados_vit/vit_run_2025-12-13_14-14-13"    # carpeta o checkpoint HF
PATH_VIT = "resultados_vit/vit_run_2025-12-13_17-02-25"    # carpeta o checkpoint HF

#PATH_EFF = "resultados_efficientnet/efficientnet_run_2025-12-12_13-01-16/best_model"    # carpeta o checkpoint HF/timm
PATH_EFF = "resultados_efficientnet/efficientnet_run_2025-12-11_22-56-59/best_model"    # carpeta o checkpoint HF/timm
TEST_DATA_DIR = "/home/jlasala/ViT tests/val"  # o ds["test"]
LABELS = ["Fire", "No_Fire"]
fire_index = LABELS.index("Fire")
no_fire_index = LABELS.index("No_Fire")
BATCH_SIZE = 16

# --- Helpers: cargar dataset (HuggingFace imagefolder) ---
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir=TEST_DATA_DIR)["train"]  # test split if present

def evaluate_model_hf(model, processor, dataset, batch_size=BATCH_SIZE):
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    probs_all = []
    paths = []
    for i in range(len(dataset)):
        item = dataset[i]
        img = item["image"].convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0].cpu().numpy()
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            pred = int(probs.argmax())
        y_true.append(int(item["label"]))
        y_pred.append(pred)
        probs_all.append(probs.tolist())
        paths.append(item.get("path", f"no_path_idx_{i}"))
    return np.array(y_true), np.array(y_pred), np.array(probs_all), paths

# --- Función para medir latencia (simple) ---
def measure_latency(model, processor, dataset, n_images=200):
    model.to(device); model.eval()
    t0 = time.time()
    cnt = 0
    for i in range(min(n_images, len(dataset))):
        item = dataset[i]
        img = item["image"].convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
        cnt += 1
    total = time.time() - t0
    return total / cnt  # seconds per image

# --- Cargar modelos (ejemplo HF AutoModelForImageClassification) ---
from transformers import AutoModelForImageClassification, AutoImageProcessor

print("Loading ViT...")
processor_vit = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model_vit = AutoModelForImageClassification.from_pretrained(PATH_VIT)
print("Evaluating ViT...")
y_true_vit, y_pred_vit, probs_vit, paths_vit = evaluate_model_hf(model_vit, processor_vit, ds)
lat_vit = measure_latency(model_vit, processor_vit, ds)

print("Loading EfficientNet...")
processor_eff = AutoImageProcessor.from_pretrained("google/efficientnet-b4")
model_eff = AutoModelForImageClassification.from_pretrained(PATH_EFF)
print("Evaluating EfficientNet...")
y_true_eff, y_pred_eff, probs_eff, paths_eff = evaluate_model_hf(model_eff, processor_eff, ds)
lat_eff = measure_latency(model_eff, processor_eff, ds)

# --- Guardar predicciones ---
out = {
    "vit": {"y_true": y_true_vit.tolist(), "y_pred": y_pred_vit.tolist(), "probs": probs_vit.tolist()},
    "eff": {"y_true": y_true_eff.tolist(), "y_pred": y_pred_eff.tolist(), "probs": probs_eff.tolist()},
    "paths": paths_vit
}
with open("comparison_preds.json", "w") as f:
    json.dump(out, f)

# --- Métricas básicas ---
print("ViT report:")
print(classification_report(y_true_vit, y_pred_vit, target_names=LABELS))
print("Eff report:")
print(classification_report(y_true_eff, y_pred_eff, target_names=LABELS))

# --- McNemar test (pares de errores) ---
# Construimos la tabla b, c: b = vit correcto, eff wrong; c = vit wrong, eff correct
b = int(((y_pred_vit == y_true_vit) & (y_pred_eff != y_true_eff)).sum())
c = int(((y_pred_vit != y_true_vit) & (y_pred_eff == y_true_eff)).sum())
print("disagree counts b,c:", b, c)
table = [[0, b],[c, 0]]
res = mcnemar(table, exact=False, correction=True)
print("McNemar statistic=%.3f, pvalue=%.5f" % (res.statistic, res.pvalue))

# --- Bootstrap CI for accuracy difference (Eff - ViT) ---
def bootstrap_ci(y_true, a_pred, b_pred, n=2000):
    n_samples = len(y_true)
    diffs = []
    idx = np.arange(n_samples)
    for _ in range(n):
        s = np.random.choice(idx, n_samples, replace=True)
        a = (a_pred[s] == y_true[s]).mean()
        b = (b_pred[s] == y_true[s]).mean()
        diffs.append(b - a)  # eff - vit
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi

ci = bootstrap_ci(y_true_vit, y_pred_vit, y_pred_eff, n=1000)
print("Bootstrap CI (Eff - ViT) acc:", ci)

# --- Latency comparison ---
print("Latency (s/img) ViT:", lat_vit, "Eff:", lat_eff)
