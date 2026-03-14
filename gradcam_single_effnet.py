import argparse
import os

import torch

from models.model_loader import load_hf_model
from predict_directory_effnet import (
    find_weights_file,
    labels_from_config,
    load_config,
    load_state_dict,
    parse_labels_csv,
    resolve_model_paths,
)
from utils.grad_cam_efficientnet import generate_efficientnet_gradcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM for a single TIFF image using a trained EfficientNet model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to run dir (contains best_model), best_model dir, or direct weights file "
            "(model.safetensors / pytorch_model.bin)."
        ),
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to a single TIFF file (.tif/.tiff).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path for Grad-CAM overlay (.png recommended).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model_name from config.json (if present).",
    )
    parser.add_argument(
        "--target_channels",
        type=int,
        default=None,
        help="Override num_channels from config.json (if present).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Override labels as comma-separated values in index order (e.g. Fire,No_Fire).",
    )
    return parser.parse_args()


def default_output_path(input_file: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.abspath(os.path.join("predictions", f"gradcam_{base_name}.png"))


def main() -> None:
    args = parse_args()

    input_file = os.path.abspath(args.input_file)
    if not os.path.isfile(input_file):
        raise ValueError(f"Input file does not exist: {input_file}")

    lower_name = input_file.lower()
    if not (lower_name.endswith(".tif") or lower_name.endswith(".tiff")):
        raise ValueError("--input_file must point to a .tif or .tiff file.")

    run_dir, model_dir, explicit_weights_file = resolve_model_paths(args.model_path)
    config_data = load_config(model_dir)

    model_name = args.model_name or str(config_data.get("model_name") or "torchvision/efficientnet_v2_s")

    if args.target_channels is not None:
        target_channels = int(args.target_channels)
    else:
        target_channels = int(config_data.get("num_channels", 6))

    if args.labels:
        labels = parse_labels_csv(args.labels)
    else:
        labels = labels_from_config(config_data)
        if labels is None:
            labels = ["Fire", "No_Fire"]

    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in enumerate(labels)}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = load_hf_model(
        model_name=model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        in_channels=target_channels,
    )

    weights_file = explicit_weights_file or find_weights_file(model_dir)
    state_dict = load_state_dict(weights_file)

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load model weights. Check model_name / target_channels / labels match training setup."
        ) from exc

    if missing_keys:
        print(f"[WARN] Missing keys while loading weights ({len(missing_keys)}).")
    if unexpected_keys:
        print(f"[WARN] Unexpected keys while loading weights ({len(unexpected_keys)}).")

    model.to(device)
    model.eval()

    output_path = os.path.abspath(args.output) if args.output else default_output_path(input_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pred_index, _, _ = generate_efficientnet_gradcam(
        model=model,
        processor=None,
        image_path=input_file,
        output_path=output_path,
    )

    pred_label = id2label.get(pred_index, str(pred_index))

    print("===========================================")
    print(f"Run dir: {run_dir}")
    print(f"Model dir: {model_dir}")
    print(f"Weights file: {weights_file}")
    print(f"Model name: {model_name}")
    print(f"Target channels: {target_channels}")
    print(f"Input file: {input_file}")
    print(f"Predicted class: {pred_label} (index={pred_index})")
    print(f"Grad-CAM saved to: {output_path}")
    print("===========================================")


if __name__ == "__main__":
    main()
