import os
import csv
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForImageClassification, AutoImageProcessor
import re

# --------------------------------------------------
# CONFIGURACIÓN
# Rutas y dispositivo: aquí se define qué modelo cargar,
# de dónde leer las imágenes y dónde guardar las predicciones.
# Cambia `MODEL_PATH` si quieres usar otro experimento.
# --------------------------------------------------
MODEL_PATH = "./resultados_efficientnet/efficientnet_run_2025-12-11_22-56-59/best_model"
# Alternativa para usar un ViT entrenado:
# MODEL_PATH = "./resultados_vit/vit_run_2025-12-13_14-14-13"

# Carpeta con imágenes externas a evaluar
IMAGES_DIR = "./external images"
# Archivo CSV de salida con las predicciones
OUTPUT_CSV = "./predictions/predictions_external_images.csv"

# Detecta GPU si está disponible, si no usa CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# CARGA DEL MODELO Y DEL PROCESADOR
# - `model`: red neuronal para clasificación de imágenes.
# - `processor`: preprocesa imágenes (resize/normalización) igual que en entrenamiento.
# Se cargan desde `MODEL_PATH` usando las utilidades de Hugging Face.
# --------------------------------------------------
print("Loading model...")
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

# Mover el modelo al dispositivo (GPU/CPU) y poner en modo evaluación
model.to(DEVICE)
model.eval()

# Mapas entre índices y etiquetas (útiles para interpretar salidas)
id2label = model.config.id2label
label2id = model.config.label2id

print("Labels:", id2label)

# --------------------------------------------------
# INFERENCIA (predicción sobre imágenes externas)
# - Recorre la carpeta `IMAGES_DIR`, procesa cada imagen y obtiene probabilidades
# - Guarda por fila: nombre de archivo, etiqueta predicha, confianza y probabilidades por clase
# --------------------------------------------------
rows = []

# Filtrar sólo archivos de imagen comunes
image_files = [
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Ordenar numéricamente por el primer número presente en el nombre de archivo.
# Si no hay número, quedará al final y se ordenará lexicográficamente.
# Esto mantiene un orden intuitivo para nombres como img_1.jpg, img_2.jpg, img_10.jpg
# en lugar del orden lexicográfico incorrecto (1,10,2).

def _extract_number(s):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else float('inf')

image_files = sorted(image_files, key=lambda x: (_extract_number(x), x))

print(f"Found {len(image_files)} images (sorted numerically)")

for fname in tqdm(image_files):
    img_path = os.path.join(IMAGES_DIR, fname)

    # Abrir la imagen con PIL y forzar RGB (algunos PNG pueden tener alpha)
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        # Si no se puede abrir la imagen, saltarla y continuar
        print(f"Skipping {fname}: {e}")
        continue

    # Preprocesado: obtiene tensores listos para el modelo
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Inferencia sin cálculo de gradientes para ahorrar memoria
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        # Convertir logits a probabilidades (softmax)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Índice y etiqueta predicha, y su confianza
    pred_idx = int(np.argmax(probs))
    pred_label = id2label[pred_idx]
    confidence = float(probs[pred_idx])

    # Construir fila con información útil para análisis posterior
    row = {
        "filename": fname,
        "prediction": pred_label,
        "confidence": confidence,
        # Extrae probabilidades explicitas para las clases esperadas
        "prob_fire": float(probs[label2id["Fire"]]),
        "prob_no_fire": float(probs[label2id["No_Fire"]]),
    }

    rows.append(row)

# --------------------------------------------------
# GUARDAR RESULTADOS EN CSV
# - Escribe un archivo CSV con todas las filas generadas en la inferencia.
# - Las columnas se definen explícitamente para mantener el orden.
# --------------------------------------------------
print(f"Saving CSV to {OUTPUT_CSV}")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "prediction",
            "confidence",
            "prob_fire",
            "prob_no_fire",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("Done.")
