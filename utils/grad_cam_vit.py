import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# =========================================================
# Función: Extraer Grad-CAM de la atención de la última capa
# =========================================================

def generate_vit_gradcam(model, processor, image_path, output_path):
    """
    Genera un mapa de calor (Heatmap) Grad-CAM para un Vision Transformer (ViT) 
    Args:
        model: ViTForImageClassification ya entrenado (fine-tuned) de HuggingFace.
        processor: ViTImageProcessor correspondiente para el preprocesamiento de la imagen.
        image_path: Ruta a la imagen original a analizar.
        output_path: Archivo donde guardar la imagen final con el heatmap superpuesto.
    """

    # 1. Modo de Evaluación
    model.eval()
    
    # ----------------------
    # 2. Cargar y Preprocesar Imagen
    # ----------------------
    img_pil = Image.open(image_path).convert("RGB")
    # El processor redimensiona, normaliza y convierte la imagen a un tensor de PyTorch (batch size 1)
    inputs = processor(images=img_pil, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ----------------------
    # 3. Preparar Interceptación de Activaciones y Gradientes (Hooks)
    # ----------------------
    # Se usarán diccionarios para almacenar el tensor capturado por cada hook
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        # El hook de forward captura la salida (output) de la capa, que son las activaciones.
        # Para ViT, es la matriz de atención de la última capa.
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        # El hook de backward captura el gradiente (grad_output) que fluye hacia atrás
        # a través de la capa de atención. Seleccionamos el primer elemento (el gradiente del output).
        gradients["value"] = grad_output[0]

    # Registramos los hooks en la capa de auto-atención (Self-Attention) del último bloque del Transformer.
    # Esta es la característica clave para Grad-CAM en ViT.
    last_block = model.vit.encoder.layer[-1].output
    forward_handle = last_block.register_forward_hook(forward_hook)
    backward_handle = last_block.register_backward_hook(backward_hook)

    # ----------------------
    # 4. Pase Forward y Determinación de la Clase Predicha
    # ----------------------
    # Ejecutamos el modelo con la imagen para obtener las predicciones
    outputs = model(**inputs)
    # Obtenemos el índice de la clase con el logit más alto (la predicción final)
    pred_class = outputs.logits.argmax().item()

    # ----------------------
    # 5. Pase Backward (Backpropagation)
    # ----------------------
    model.zero_grad()
    # Calculamos los gradientes de la PREDICCIÓN ELEGIDA (pred_class) respecto a los pesos.
    # Esto propaga el "interés" de la clase objetivo hacia atrás.
    outputs.logits[0, pred_class].backward()

    # ----------------------
    # 6. Cálculo del Mapa de Calor (Grad-CAM)
    # ----------------------
    # Leemos activaciones (Activ) y gradientes (Grads) capturados por los hooks.
    # Ambas tienen forma (1, num_heads, seq_len, seq_len), donde seq_len es 1 (CLS) + num_patches.
    activ = activations["value"] 
    grads = gradients["value"] 

    # Grad-CAM: Calculamos los "pesos de importancia" para cada cabeza.
    # Esto se hace promediando los gradientes sobre la dimensión de las cabezas (dim=1).
    weights = grads.mean(dim=1, keepdim=True) 
    
    # El mapa CAM es la suma ponderada de las activaciones (Atención) por los pesos (Gradientes).
    # Multiplicamos pesos por activaciones y sumamos a través de las cabezas.
    # Forma resultante: (1, 1, seq_len, seq_len)
    cam = (weights * activ).sum(dim=-1) 

    # ----------------------
    # 7. Post-procesamiento del CAM (ViT específico)
    # ----------------------
    # Seleccionamos la primera (y única) muestra, el primer canal y excluimos el token CLS (posición 0).
    cam = cam[0, 1:] 
    
    # Calculamos la dimensión de los parches (la raíz cuadrada de la longitud de la secuencia de parches).
    num_patches = int(cam.shape[0] ** 0.5)
    # Redimensionamos el mapa a una matriz 2D de parches (ej. 14x14).
    cam = cam.reshape(num_patches, num_patches).detach().cpu().numpy()

    # Normalización Min-Max para escalar los valores entre 0 y 1
    cam -= cam.min()
    cam /= cam.max()

    # ----------------------
    # 8. Visualización
    # ----------------------
    # Redimensionar el mapa de parches a la resolución original de la imagen (ej. 224x224).
    heatmap = cv2.resize(cam, img_pil.size)

    # Convertir el mapa de calor a una imagen en color (colormap JET)
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET # Colores cálidos (rojo) indican alta relevancia, fríos (azul) baja.
    )[:, :, ::-1]    # Convertir de BGR (formato OpenCV) a RGB

    # Mezclar el heatmap de color con la imagen original (Superposición - Overlay)
    image_np = np.array(img_pil)
    # Fórmula de superposición: 40% Heatmap + 60% Imagen Original
    overlay = (0.4 * heatmap_color + 0.6 * image_np).astype(np.uint8)

    # ----------------------
    # 9. Guardar y Limpiar
    # ----------------------
    # Guardar el resultado. Se convierte de RGB a BGR antes de guardar, que es el formato esperado por cv2.imwrite.
    cv2.imwrite(output_path, overlay[:, :, ::-1]) 

    # Es crucial borrar los hooks para evitar consumir memoria y asegurar que no afecten
    # a las siguientes ejecuciones o inferencias del modelo.
    forward_handle.remove()
    backward_handle.remove()

    return pred_class, image_np, overlay

def create_gradcam_for_misclassified(model, processor, fp_paths, fn_paths, output_dir):
    fp_gradcam_dir = os.path.join(output_dir, "false_positives_gradcam")
    fn_gradcam_dir = os.path.join(output_dir, "false_negatives_gradcam")
    # Crear carpetas si no existen
    os.makedirs(fp_gradcam_dir, exist_ok=True)
    os.makedirs(fn_gradcam_dir, exist_ok=True)

    # Grad-CAM para falsos positivos
    for img_path in fp_paths:
        filename = os.path.basename(img_path)
        save_path = os.path.join(fp_gradcam_dir, f"GC_{filename}") 
        generate_vit_gradcam(model, processor, img_path, save_path) 

    # Grad-CAM para falsos negativos
    for img_path in fn_paths:
        filename = os.path.basename(img_path)
        save_path = os.path.join(fn_gradcam_dir, f"GC_{filename}")
        generate_vit_gradcam(model, processor, img_path, save_path)

    print("\n✔ Grad-CAM generado para FP y FN.\n")
