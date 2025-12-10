import torch
import numpy as np
import cv2
import os
from PIL import Image


# =====================================================
# Obtener última capa convolucional de EfficientNet
# (para calcular Grad-CAM)
# =====================================================
def get_last_conv_layer(model):
    """
    Obtiene la última capa convolucional REAL de EfficientNet,
    válida para Grad-CAM. Funciona para todos los efficientnet-b0..b7.
    """
    try:
        # última capa MBConv
        return model.efficientnet.blocks[-1].layers[-1].conv
    except Exception:
        # fallback seguro: buscar la última Conv2d VÁLIDA
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv


# =====================================================
# Grad-CAM para EfficientNet
# =====================================================
def generate_efficientnet_gradcam(model, processor, image_path, output_path):
    """
    Genera un Grad-CAM clásico para EfficientNet.
    """
    model.eval()
    
    # ----- cargar imagen -----
    img_pil = Image.open(image_path).convert("RGB")
    inputs = processor(images=img_pil, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ----- obtener última capa conv -----
    target_layer = get_last_conv_layer(model)
    if target_layer is None:
        raise ValueError("No se encontró una capa Conv2d en EfficientNet.")

    activations = {}
    gradients = {}

    # forward hook
    def forward_hook(module, input, output):
        activations["value"] = output

    # backward hook
    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    # ----- forward -----
    outputs = model(**inputs)
    pred_class = outputs.logits.argmax().item()

    # ----- backward -----
    model.zero_grad()
    outputs.logits[0, pred_class].backward()

    # ----- extraer datos -----
    activ = activations["value"]        # (1, C, H, W)
    grads = gradients["value"]          # (1, C, H, W)

    # pesos = promedio por canal
    weights = grads.mean(dim=(2, 3), keepdim=True)

    # mapa Grad-CAM
    cam = (weights * activ).sum(dim=1).squeeze().detach().cpu().numpy()

    # normalizar
    cam = np.maximum(cam, 0)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    # ----- overlay -----
    w, h = img_pil.size
    if w == 0 or h == 0:
        print(f"[ERROR] Imagen inválida (size == 0) en {image_path}")
        return
    dsize = (int(w), int(h))

    if cam.size == 0 or cam.ndim != 2:
        print(f"[ERROR] CAM inválido para {image_path}, shape={cam.shape}")
    return

    cam_resized = cv2.resize(cam, dsize)


    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )[:, :, ::-1]

    image_np = np.array(img_pil)
    overlay = (0.4 * heatmap + 0.6 * image_np).astype(np.uint8)

    cv2.imwrite(output_path, overlay[:, :, ::-1])

    # limpiar hooks
    fh.remove()
    bh.remove()

    return pred_class, image_np, overlay


# =====================================================
# Crear Grad-CAM para falsos positivos y negativos
# =====================================================
def create_gradcam_for_misclassified(model, processor, fp_paths, fn_paths, output_dir):
    fp_gradcam_dir = os.path.join(output_dir, "false_positives_gradcam")
    fn_gradcam_dir = os.path.join(output_dir, "false_negatives_gradcam")

    os.makedirs(fp_gradcam_dir, exist_ok=True)
    os.makedirs(fn_gradcam_dir, exist_ok=True)

    print("\n>>> Generando Grad-CAM para EfficientNet...")

    # falsos positivos
    for img_path in fp_paths:
        filename = os.path.basename(img_path)
        save_path = os.path.join(fp_gradcam_dir, f"GC_{filename}")
        generate_efficientnet_gradcam(model, processor, img_path, save_path)

    # falsos negativos
    for img_path in fn_paths:
        filename = os.path.basename(img_path)
        save_path = os.path.join(fn_gradcam_dir, f"GC_{filename}")
        generate_efficientnet_gradcam(model, processor, img_path, save_path)

    print("✔ Grad-CAM listo para FP y FN.\n")
