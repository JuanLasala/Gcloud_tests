import numpy as np
from typing import Tuple
from scipy.special import softmax
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix


def find_optimal_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    fire_index: int,
    metric: str = "f1",
    beta: float = 2.0,
    min_recall: float = None,  # <-- optional constraint for deployment
) -> Tuple[float, dict]:

    # Stable softmax
    probs = softmax(logits, axis=1)
    fire_probs = probs[:, fire_index]

    binary_labels = (labels == fire_index).astype(int)

    precisions, recalls, thresholds = precision_recall_curve(
        binary_labels,
        fire_probs,
    )

    # PR curve returns len(thresholds) = len(precisions) - 1
    # So align arrays
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if metric == "f1":
        scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    elif metric == "f1_weighted":
        scores = (1 + beta**2) * (precisions * recalls) / (
            beta**2 * precisions + recalls + 1e-10
        )

    elif metric == "recall":
        scores = recalls

    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Optional: enforce minimum recall constraint (very useful for wildfire)
    if min_recall is not None:
        valid_idxs = np.where(recalls >= min_recall)[0]
        if len(valid_idxs) > 0:
            best_idx = valid_idxs[np.argmax(scores[valid_idxs])]
        else:
            best_idx = np.argmax(scores)
    else:
        best_idx = np.argmax(scores)

    optimal_threshold = thresholds[best_idx]

    preds_thresholded = (fire_probs >= optimal_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        binary_labels,
        preds_thresholded
    ).ravel()

    metrics_dict = {
        "threshold": float(optimal_threshold),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "score": float(scores[best_idx]),  # <-- generalized score
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return optimal_threshold, metrics_dict


def apply_threshold(
    logits: np.ndarray,
    fire_index: int,
    threshold: float
) -> np.ndarray:
    
    """
    Apply a custom threshold to logits and return predictions.
    
    Args:
        logits: Model output logits (N, 2).
        fire_index: Index of the Fire class.
        threshold: Probability threshold for Fire class.
    
    Returns:
        preds: Predicted labels (fire_index or other_index).
    """

    probs = softmax(logits, axis=1)  # <-- stable
    fire_probs = probs[:, fire_index]

    other_index = 1 - fire_index

    preds = np.where(
        fire_probs >= threshold,
        fire_index,
        other_index
    )

    return preds