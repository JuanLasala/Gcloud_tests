import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

def generate_vit_gradcam(model, processor, image_path, output_path):
    """Generate a Grad-CAM heatmap for a Vision Transformer (ViT).

    Args:
        model: a fine-tuned `ViTForImageClassification` model.
        processor: corresponding `ViTImageProcessor` for preprocessing.
        image_path: path to the input image.
        output_path: path where the overlay image will be saved.
    Returns:
        pred_class (int), original_image_np (np.ndarray), overlay_np (np.ndarray)
    """

    model.eval()  # set model to evaluation mode

    img_pil = Image.open(image_path).convert("RGB")  # load input image as RGB
    inputs = processor(images=img_pil, return_tensors="pt")  # preprocess (resize, normalize, tensor)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}  # move inputs to model device

    activations = {}  # store activations from forward hook
    gradients = {}    # store gradients from backward hook

    def forward_hook(module, input, output):
        activations["value"] = output  # capture forward output (attention activations)

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]  # capture gradient of the output

    # register hooks on the last transformer's output block (self-attention output)
    last_block = model.vit.encoder.layer[-1].output
    forward_handle = last_block.register_forward_hook(forward_hook)
    backward_handle = last_block.register_backward_hook(backward_hook)

    outputs = model(**inputs)  # forward pass
    pred_class = outputs.logits.argmax().item()  # predicted class index

    model.zero_grad()
    outputs.logits[0, pred_class].backward()  # backward pass for the predicted class

    activ = activations["value"]  # (1, num_heads, seq_len, seq_len)
    grads = gradients["value"]     # same shape as activ

    weights = grads.mean(dim=1, keepdim=True)  # importance weights per head (average over heads)
    cam = (weights * activ).sum(dim=-1)  # weighted sum over attention

    cam = cam[0, 1:]  # remove batch dim and CLS token (position 0)
    num_patches = int(cam.shape[0] ** 0.5)
    cam = cam.reshape(num_patches, num_patches).detach().cpu().numpy()  # to 2D patch grid

    cam -= cam.min()
    cam /= cam.max()  # normalize to [0, 1]

    heatmap = cv2.resize(cam, img_pil.size)  # upsample to original image size
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB

    image_np = np.array(img_pil)
    overlay = (0.4 * heatmap_color + 0.6 * image_np).astype(np.uint8)  # overlay (40% heatmap)

    cv2.imwrite(output_path, overlay[:, :, ::-1])  # save as BGR for OpenCV

    # remove hooks to free resources and avoid side effects
    forward_handle.remove()
    backward_handle.remove()

    return pred_class, image_np, overlay


def create_gradcam_for_misclassified(model, processor, fp_paths, fn_paths, output_dir):
    fp_gradcam_dir = os.path.join(output_dir, "false_positives_gradcam")  # dir for false positives
    fn_gradcam_dir = os.path.join(output_dir, "false_negatives_gradcam")  # dir for false negatives
    os.makedirs(fp_gradcam_dir, exist_ok=True)
    os.makedirs(fn_gradcam_dir, exist_ok=True)

    for img_path in fp_paths:  # generate Grad-CAM for false positives
        filename = os.path.basename(img_path)
        save_path = os.path.join(fp_gradcam_dir, f"GC_{filename}")
        generate_vit_gradcam(model, processor, img_path, save_path)

    for img_path in fn_paths:  # generate Grad-CAM for false negatives
        filename = os.path.basename(img_path)
        save_path = os.path.join(fn_gradcam_dir, f"GC_{filename}")
        generate_vit_gradcam(model, processor, img_path, save_path)

    print("\n✔ Grad-CAM generated for false positives and false negatives.\n")
