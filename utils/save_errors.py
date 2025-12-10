import os
from PIL import Image
import torch

fp_paths = []
fn_paths = []

def save_misclassified_images(model, processor, dataset, output_dir, fire_index, no_fire_index): 
    """
    Guarda imágenes mal clasificadas (FP y FN) en carpetas separadas.
    Funciona tanto si dataset[i] contiene 'path' como si no.
    """

    # Carpetas de salida
    fp_dir = os.path.join(output_dir, "false_positives")
    fn_dir = os.path.join(output_dir, "false_negatives")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    print("\n>>> Buscando errores de clasificación...\n")

    model.eval()
    device = next(model.parameters()).device

    fp_count = 0
    fn_count = 0

    for i in range(len(dataset)):
        item = dataset[i]

        # --------------------------
        # 1) Obtener la imagen
        # --------------------------
        if "path" in item:  
            # CASO 1: ImageFolder de torchvision
            image_path = item["path"]
            image = Image.open(image_path).convert("RGB")
        else:
            # CASO 2: Dataset HuggingFace load_imagefolder (tu caso)
            image = item["image"].convert("RGB")
            image_path = None  # No existe ruta original

        true_label = int(item["label"])

        # --------------------------
        # 2) Procesar imagen
        # --------------------------
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = outputs.logits.argmax(dim=-1).item()
        # --------------------------
        # EXTRA: extraer país y número desde el nombre original
        # --------------------------
        pais = "unknown"
        num = "unknown"

        if image_path is not None:
            base = os.path.basename(image_path).replace(".jpg", "")
            parts = base.split("_")
            if len(parts) >= 2:
                pais = parts[0]
                num = parts[1]

        # --------------------------
        # 3) Chequear error
        # --------------------------
        if pred_label != true_label:
            # filename final
            filename = (f"{pais}_{num}_true-{true_label}_pred-{pred_label}.jpg"
                if image_path is None
                else os.path.basename(image_path).replace(".jpg", f"_true-{true_label}_pred-{pred_label}.jpg")
            )

            #if true_label == 0 and pred_label == 1:
            if true_label == no_fire_index and pred_label == fire_index:
                save_path = os.path.join(fp_dir, filename)
                image.save(save_path)
                fp_paths.append(save_path)
                fp_count += 1

            #elif true_label == 1 and pred_label == 0:
            elif true_label == fire_index and pred_label == no_fire_index:
                save_path = os.path.join(fn_dir, filename)
                image.save(save_path)
                fn_paths.append(save_path)
                fn_count += 1

    print(f"✔ Errores encontrados: {fp_count + fn_count}")
    print(f"   - Falsos Positivos: {fp_count}")
    print(f"   - Falsos Negativos: {fn_count}")
    print(f"\nGuardados en:\n{output_dir}\n")

    return fp_count, fn_count, fp_paths, fn_paths
