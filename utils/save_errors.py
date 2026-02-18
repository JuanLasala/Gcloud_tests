import os
from PIL import Image
import torch

from data.multiband_tiff import load_multiband_tiff, make_rgb_preview

fp_paths = []
fn_paths = []


def _get_model_input_channels(model, default=13):
    if hasattr(model, "config") and hasattr(model.config, "num_channels"):
        return int(model.config.num_channels)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.in_channels)
    return default


def save_misclassified_images(model, processor, dataset, output_dir, fire_index, no_fire_index): 
    """
    Save misclassified images (FP and FN) in separate folders.
    """

    # Output directories
    fp_dir = os.path.join(output_dir, "false_positives")
    fn_dir = os.path.join(output_dir, "false_negatives")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    print("\n>>> Looking for misclassified images...\n")

    model.eval()
    device = next(model.parameters()).device
    target_channels = _get_model_input_channels(model)

    fp_count = 0
    fn_count = 0

    for i in range(len(dataset)):
        item = dataset[i]

        # --------------------------
        # 1) Obtain image
        # --------------------------
        if "path" in item:
            image_path = item["path"]
            tensor = load_multiband_tiff(image_path, target_channels=target_channels)
            image = Image.fromarray(make_rgb_preview(tensor))
        else:
            image = item["image"].convert("RGB")
            image_path = None

        label_key = "label" if "label" in item else "labels"
        true_label = int(item[label_key])

        # --------------------------
        # 2) Process image
        # --------------------------
        if image_path is not None:
            inputs = {"pixel_values": tensor.unsqueeze(0)}
        else:
            inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = outputs.logits.argmax(dim=-1).item()
        # --------------------------
        # EXTRA: extract country and id number from original filename
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
        # 3) Check error
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

    print(f"✔ Errors found: {fp_count + fn_count}")
    print(f"   - False Positives: {fp_count}")
    print(f"   - False Negatives: {fn_count}")
    print(f"\nSaved in:\n{output_dir}\n")

    return fp_count, fn_count, fp_paths, fn_paths
