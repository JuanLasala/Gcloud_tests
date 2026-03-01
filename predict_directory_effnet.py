import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data.augmentations import eval_augmentations_multiband
from data.multiband_tiff import load_multiband_tiff
from models.model_loader import load_hf_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on all TIFF images in a directory using a trained EfficientNet model."
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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory with TIFF files (.tif/.tiff). Search is recursive.",
    )
    input_group.add_argument(
        "--input_file",
        type=str,
        help="Path to a single TIFF file (.tif/.tiff).",
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold for positive class. If omitted, script tries optimal_threshold.json then argmax.",
    )
    parser.add_argument(
        "--threshold_json",
        type=str,
        default=None,
        help="Optional custom path to optimal_threshold.json.",
    )
    parser.add_argument(
        "--fire_label",
        type=str,
        default="Fire",
        help="Positive class name used by threshold logic.",
    )
    return parser.parse_args()


def parse_labels_csv(labels_csv: str) -> List[str]:
    labels = [label.strip() for label in labels_csv.split(",") if label.strip()]
    if len(labels) < 2:
        raise ValueError("At least two labels are required (e.g. Fire,No_Fire).")
    return labels


def resolve_model_paths(model_path: str) -> Tuple[str, str, Optional[str]]:
    abs_input = os.path.abspath(model_path)

    if os.path.isfile(abs_input):
        if abs_input.endswith(".safetensors") or abs_input.endswith(".bin"):
            model_dir = os.path.dirname(abs_input)
            run_dir = os.path.dirname(model_dir) if os.path.basename(model_dir) == "best_model" else model_dir
            return run_dir, model_dir, abs_input
        raise ValueError(
            f"If --model_path is a file, it must be model.safetensors or pytorch_model.bin. Got: {abs_input}"
        )

    if not os.path.isdir(abs_input):
        raise ValueError(f"Model path does not exist or is not a directory: {abs_input}")

    best_model_dir = os.path.join(abs_input, "best_model")
    if os.path.isdir(best_model_dir):
        return abs_input, best_model_dir, None

    if os.path.basename(abs_input) == "best_model":
        return os.path.dirname(abs_input), abs_input, None

    return abs_input, abs_input, None


def find_weights_file(model_dir: str) -> str:
    safe_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(safe_path):
        return safe_path
    if os.path.isfile(bin_path):
        return bin_path
    raise FileNotFoundError(
        f"No weights found in '{model_dir}'. Expected model.safetensors or pytorch_model.bin."
    )


def load_state_dict(weights_file: str) -> Dict[str, torch.Tensor]:
    if weights_file.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError as exc:
            raise ImportError(
                "Found model.safetensors but 'safetensors' is not installed. Install with: pip install safetensors"
            ) from exc
        state_dict = load_safetensors(weights_file)
    else:
        state_dict = torch.load(weights_file, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise RuntimeError("Loaded weights are not a valid state_dict dictionary.")

    return state_dict


def load_config(model_dir: str) -> Dict[str, object]:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r") as config_handle:
        return json.load(config_handle)


def labels_from_config(config_data: Dict[str, object]) -> Optional[List[str]]:
    id2label = config_data.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        return None

    pairs = []
    for raw_index, raw_label in id2label.items():
        try:
            idx = int(raw_index)
        except (ValueError, TypeError):
            continue
        pairs.append((idx, str(raw_label)))

    if not pairs:
        return None

    pairs.sort(key=lambda item: item[0])
    return [label for _, label in pairs]


def resolve_threshold(
    run_dir: str,
    model_dir: str,
    threshold_arg: Optional[float],
    threshold_json_arg: Optional[str],
) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    if threshold_arg is not None:
        return float(threshold_arg), None, None

    candidates: List[str] = []
    if threshold_json_arg:
        candidates.append(os.path.abspath(threshold_json_arg))
    else:
        candidates.append(os.path.join(run_dir, "optimal_threshold.json"))
        candidates.append(os.path.join(model_dir, "optimal_threshold.json"))

    for candidate in candidates:
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


def predict_file(
    model: torch.nn.Module,
    file_path: str,
    device: str,
    id2label: Dict[int, str],
    fire_index: Optional[int],
    threshold: Optional[float],
) -> Dict[str, object]:
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

    row: Dict[str, object] = {
        "file_path": file_path,
        "prediction": id2label[pred_index],
        "prediction_index": pred_index,
        "confidence": float(probs[pred_index]),
        "threshold_used": "" if threshold is None else float(threshold),
        "probabilities": json.dumps(
            {id2label[class_idx]: float(prob) for class_idx, prob in enumerate(probs)}
        ),
    }
    if fire_index is not None:
        row["prob_fire"] = float(probs[fire_index])

    return row


def main() -> None:
    args = parse_args()

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

    if args.input_file:
        input_file = os.path.abspath(args.input_file)
        if not os.path.isfile(input_file):
            raise ValueError(f"Input file does not exist: {input_file}")
        lower_name = input_file.lower()
        if not (lower_name.endswith(".tif") or lower_name.endswith(".tiff")):
            raise ValueError("--input_file must point to a .tif or .tiff file.")
        tiff_files = [input_file]
        input_target = input_file
    else:
        input_dir = os.path.abspath(args.input_dir)
        tiff_files = collect_tiff_files(input_dir)
        if not tiff_files:
            raise ValueError(f"No TIFF files found in: {input_dir}")
        input_target = input_dir

    print("===========================================")
    print(f"Model dir: {model_dir}")
    print(f"Weights file: {weights_file}")
    print(f"Model name: {model_name}")
    print(f"Target channels: {target_channels}")
    print(f"Labels: {labels}")
    print(f"Input target: {input_target}")
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

    rows: List[Dict[str, object]] = []
    failures = 0

    with torch.no_grad():
        for index, file_path in enumerate(tiff_files, start=1):
            try:
                row = predict_file(
                    model=model,
                    file_path=file_path,
                    device=device,
                    id2label=id2label,
                    fire_index=fire_index,
                    threshold=threshold,
                )
                rows.append(row)

                if index % 50 == 0 or index == len(tiff_files):
                    print(f"Processed {index}/{len(tiff_files)}")

            except Exception as exc:
                failures += 1
                print(f"[WARN] Skipping '{file_path}': {exc}")

    output_csv = os.path.abspath(args.output_csv)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
