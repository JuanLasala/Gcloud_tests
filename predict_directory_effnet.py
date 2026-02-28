import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import safetensors

from data.augmentations import eval_augmentations_multiband
from data.multiband_tiff import load_multiband_tiff
from models.model_loader import load_hf_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify all TIFF files in a directory using a trained EfficientNet model "
            "from this project."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to a run directory (contains best_model/) OR directly to best_model/. "
            "Example: ./resultados_efficientnet/efficientnet_run_YYYY-mm-dd_HH-MM-SS"
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with TIFF files (.tif/.tiff). Search is recursive.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./predictions/predictions_directory.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="torchvision/efficientnet_v2_s",
        help="Base architecture used during training.",
    )
    parser.add_argument(
        "--target_channels",
        type=int,
        default=6,
        help="Number of input channels expected by the trained model.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="Fire,No_Fire",
        help="Comma-separated labels in class-index order used during training.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fire threshold. If set, prediction uses this threshold instead of argmax.",
    )
    parser.add_argument(
        "--threshold_json",
        type=str,
        default=None,
        help=(
            "Optional path to optimal_threshold.json. If omitted, script tries to auto-find it "
            "inside the run directory."
        ),
    )
    parser.add_argument(
        "--fire_label",
        type=str,
        default="Fire",
        help="Name of the positive class used for thresholding.",
    )
    return parser.parse_args()


def parse_labels(labels_arg: str) -> List[str]:
    labels = [label.strip() for label in labels_arg.split(",") if label.strip()]
    if len(labels) < 2:
        raise ValueError("At least two labels are required (e.g. 'Fire,No_Fire').")
    return labels


def resolve_model_paths(model_path: str) -> Tuple[str, str]:
    abs_input = os.path.abspath(model_path)
    best_model_dir = os.path.join(abs_input, "best_model")

    if os.path.isdir(best_model_dir):
        return abs_input, best_model_dir

    if os.path.isdir(abs_input):
        return os.path.dirname(abs_input), abs_input

    raise ValueError(f"Model path does not exist or is not a directory: {model_path}")


def find_weights_file(model_dir: str) -> str:
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.isfile(safetensors_path):
        return safetensors_path
    if os.path.isfile(pytorch_path):
        return pytorch_path

    raise FileNotFoundError(
        f"No model weights found in '{model_dir}'. Expected model.safetensors or pytorch_model.bin."
    )


def load_state_dict(weights_file: str) -> Dict[str, torch.Tensor]:
    if weights_file.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError as exc:
            raise ImportError(
                "Found model.safetensors but safetensors is not installed. Install with: pip install safetensors"
            ) from exc

        state_dict = load_safetensors(weights_file)
    else:
        state_dict = torch.load(weights_file, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise RuntimeError("Loaded weights are not a valid state_dict dictionary.")

    return state_dict


def resolve_threshold(
    run_dir: str,
    model_dir: str,
    threshold_arg: Optional[float],
    threshold_json_arg: Optional[str],
) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    if threshold_arg is not None:
        return float(threshold_arg), None, None

    candidate_files: List[str] = []
    if threshold_json_arg:
        candidate_files.append(os.path.abspath(threshold_json_arg))
    else:
        candidate_files.append(os.path.join(run_dir, "optimal_threshold.json"))
        candidate_files.append(os.path.join(model_dir, "optimal_threshold.json"))

    for candidate in candidate_files:
        if not os.path.isfile(candidate):
            continue

        with open(candidate, "r") as file_handle:
            payload = json.load(file_handle)

        threshold = payload.get("optimal_threshold")
        fire_index = payload.get("fire_index")
        if threshold is None:
            continue

        fire_index_value = int(fire_index) if fire_index is not None else None
        return float(threshold), fire_index_value, candidate

    return None, None, None


def collect_tiff_files(input_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            lower_name = filename.lower()
            if lower_name.endswith(".tif") or lower_name.endswith(".tiff"):
                files.append(os.path.join(root, filename))
    files.sort()
    return files


def main() -> None:
    args = parse_args()

    run_dir, model_dir = resolve_model_paths(args.model_path)
    labels = parse_labels(args.labels)

    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in enumerate(labels)}

    fire_index_from_label = label2id.get(args.fire_label)

    threshold, fire_index_from_json, threshold_source = resolve_threshold(
        run_dir=run_dir,
        model_dir=model_dir,
        threshold_arg=args.threshold,
        threshold_json_arg=args.threshold_json,
    )

    fire_index = fire_index_from_json if fire_index_from_json is not None else fire_index_from_label

    if threshold is not None and fire_index is None:
        raise ValueError(
            f"Threshold is set but fire label '{args.fire_label}' was not found in labels={labels}."
        )

    if threshold is not None and len(labels) != 2:
        raise ValueError("Threshold-based inference currently supports only binary classification.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = load_hf_model(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        in_channels=args.target_channels,
    )

    weights_file = find_weights_file(model_dir)
    state_dict = load_state_dict(weights_file)

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load model weights. Check --model_name, --target_channels and --labels "
            "match the original training setup."
        ) from exc

    if missing_keys:
        print(f"[WARN] Missing keys while loading weights ({len(missing_keys)}).")
    if unexpected_keys:
        print(f"[WARN] Unexpected keys while loading weights ({len(unexpected_keys)}).")

    model.to(device)
    model.eval()

    input_dir = os.path.abspath(args.input_dir)
    tiff_files = collect_tiff_files(input_dir)

    if not tiff_files:
        raise ValueError(f"No TIFF files found in: {input_dir}")

    rows: List[Dict[str, object]] = []
    failures = 0

    print("===========================================")
    print(f"Model dir: {model_dir}")
    print(f"Weights file: {weights_file}")
    print(f"Input dir: {input_dir}")
    print(f"Files found: {len(tiff_files)}")
    print(f"Device: {device}")
    if threshold is not None:
        threshold_msg = f"Using threshold={threshold:.6f}"
        if threshold_source:
            threshold_msg += f" (from {threshold_source})"
        print(threshold_msg)
    else:
        print("Using argmax inference (no threshold).")
    print("===========================================")

    with torch.no_grad():
        for index, file_path in enumerate(tiff_files, start=1):
            try:
                tensor = load_multiband_tiff(file_path)
                tensor = eval_augmentations_multiband(tensor)
                batch = tensor.unsqueeze(0).to(device)

                outputs = model(pixel_values=batch)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()

                argmax_index = int(np.argmax(probs))

                if threshold is not None:
                    fire_prob = float(probs[fire_index])
                    other_index = 1 - fire_index
                    pred_index = fire_index if fire_prob >= threshold else other_index
                else:
                    pred_index = argmax_index

                confidence = float(probs[pred_index])

                row: Dict[str, object] = {
                    "file_path": file_path,
                    "prediction": id2label[pred_index],
                    "prediction_index": pred_index,
                    "confidence": confidence,
                    "probabilities": json.dumps({
                        id2label[class_index]: float(prob) for class_index, prob in enumerate(probs)
                    }),
                    "threshold_used": "" if threshold is None else float(threshold),
                }

                if fire_index is not None:
                    row["prob_fire"] = float(probs[fire_index])

                rows.append(row)

                if index % 50 == 0 or index == len(tiff_files):
                    print(f"Processed {index}/{len(tiff_files)}")

            except Exception as exc:
                failures += 1
                print(f"[WARN] Skipping '{file_path}': {exc}")

    output_csv = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    base_fields = [
        "file_path",
        "prediction",
        "prediction_index",
        "confidence",
        "threshold_used",
        "prob_fire",
        "probabilities",
    ]
    fieldnames = [field for field in base_fields if any(field in row for row in rows)]

    with open(output_csv, "w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("===========================================")
    print(f"Done. Successful predictions: {len(rows)}")
    print(f"Failed files: {failures}")
    print(f"CSV saved to: {output_csv}")
    print("===========================================")


if __name__ == "__main__":
    main()
