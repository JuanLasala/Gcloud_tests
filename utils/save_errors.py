import os
import torch
import torch.nn.functional as F
import numpy as np

from data.multiband_tiff import load_multiband_tiff, make_rgb_preview

fp_paths = []
fn_paths = []


def _get_model_target_size(model):
    image_size = getattr(getattr(model, "config", None), "image_size", None)
    if image_size is None:
        return None

    if isinstance(image_size, int):
        return image_size, image_size

    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])

    return None


def save_misclassified_images(
    model,
    dataset,
    output_dir,
    fire_index,
    no_fire_index,
    threshold=None,
    pred_labels=None,
    true_labels=None,
):
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
    target_size = _get_model_target_size(model)

    use_precomputed = pred_labels is not None and true_labels is not None
    if use_precomputed:
        pred_labels = np.asarray(pred_labels).astype(int)
        true_labels = np.asarray(true_labels).astype(int)
        if len(pred_labels) != len(true_labels):
            raise ValueError("pred_labels and true_labels must have the same length.")
        if len(pred_labels) != len(dataset):
            raise ValueError(
                f"Length mismatch: dataset={len(dataset)}, pred_labels={len(pred_labels)}, true_labels={len(true_labels)}"
            )

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
        if "pixel_values" in item:
            tensor = item["pixel_values"]
        else:
            tensor = load_multiband_tiff(image_path)

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)

        if target_size is not None and tensor.shape[-2:] != target_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if use_precomputed:
            true_label = int(true_labels[i])
            pred_label = int(pred_labels[i])
        else:
            label_key = "label" if "label" in item else "labels"
            true_label = int(item[label_key])

            # --------------------------
            # 2) Forward pass
            # --------------------------
            inputs = {"pixel_values": tensor.unsqueeze(0).to(device)}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                if threshold is None:
                    pred_label = logits.argmax(dim=-1).item()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    fire_prob = probs[0, fire_index].item()
                    pred_label = fire_index if fire_prob >= float(threshold) else no_fire_index

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