import os
from PIL import Image
import torch

from data.multiband_tiff import load_multiband_tiff, make_rgb_preview

fp_paths = []
fn_paths = []


def save_misclassified_images(model, dataset, output_dir, fire_index, no_fire_index):
    """
    Collect paths of misclassified images (FP and FN)
    and save them to text files.
    """

    # Output directories
    fp_dir = os.path.join(output_dir, "false_positives")
    fn_dir = os.path.join(output_dir, "false_negatives")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    print("\n>>> Looking for misclassified images...\n")

    model.eval()
    device = next(model.parameters()).device

    fp_count = 0
    fn_count = 0
    fp_paths = []
    fn_paths = []

    for i in range(len(dataset)):
        item = dataset[i]

        # --------------------------
        # 1) Load multiband image
        # --------------------------
        image_path = item["path"]
        tensor = load_multiband_tiff(image_path)

        label_key = "label" if "label" in item else "labels"
        true_label = int(item[label_key])

        # --------------------------
        # 2) Forward pass
        # --------------------------
        inputs = {"pixel_values": tensor.unsqueeze(0).to(device)}

        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = outputs.logits.argmax(dim=-1).item()

        # --------------------------
        # 3) Check error
        # --------------------------
        if pred_label != true_label:

            if true_label == no_fire_index and pred_label == fire_index:
                fp_paths.append(image_path)
                fp_count += 1

            elif true_label == fire_index and pred_label == no_fire_index:
                fn_paths.append(image_path)
                fn_count += 1

    # --------------------------
    # 4) Save path lists to disk
    # --------------------------
    fp_txt = os.path.join(fp_dir, "false_positives.txt")
    fn_txt = os.path.join(fn_dir, "false_negatives.txt")

    with open(fp_txt, "w") as f:
        for path in fp_paths:
            f.write(path + "\n")

    with open(fn_txt, "w") as f:
        for path in fn_paths:
            f.write(path + "\n")

    print(f"✔ Errors found: {fp_count + fn_count}")
    print(f"   - False Positives: {fp_count}")
    print(f"   - False Negatives: {fn_count}")
    print(f"\nSaved lists in:\n{output_dir}\n")

    return fp_count, fn_count, fp_paths, fn_paths

