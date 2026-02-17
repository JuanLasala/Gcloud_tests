import numpy as np
from typing import Tuple
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix


def find_optimal_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    fire_index: int,
    metric: str = "f1",
    beta: float = 2.0,
) -> Tuple[float, dict]:
    """
    Find optimal classification threshold for binary Fire/No_Fire classification.
    
    Args:
        logits: Model output logits (N, 2) for two classes.
        labels: Ground truth labels (N,).
        fire_index: Index of the Fire class (typically 0 or 1).
        metric: "f1" (standard F1), "f1_weighted" (beta-weighted), or "recall" (minimize FN).
        beta: If using F1-weighted, weight recall this many times over precision.
               beta=2 means recall is 2x as important (better for catching fires).
    
    Returns:
        optimal_threshold: Best threshold for Fire class probability.
        metrics_dict: Dict with threshold, precision, recall, f1 at that threshold.
    """
    # Extract Fire class probabilities
    fire_probs = np.exp(logits[:, fire_index]) / np.sum(np.exp(logits), axis=1)
    
    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(
        (labels == fire_index).astype(int),
        fire_probs,
    )
    
    # Evaluate F1 or weighted metric at each threshold
    if metric == "f1":
        # Standard F1: harmonic mean of precision and recall
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
    elif metric == "f1_weighted":
        # Beta-weighted F1: emphasize recall if beta > 1
        f1_beta = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_beta)
    elif metric == "recall":
        # Maximize recall (minimize false negatives)
        best_idx = np.argmax(recalls)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Compute confusion matrix at optimal threshold
    preds_thresholded = (fire_probs >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix((labels == fire_index).astype(int), preds_thresholded).ravel()
    
    metrics_dict = {
        "threshold": float(optimal_threshold),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1_scores[best_idx] if metric == "f1" else f1_beta[best_idx]),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    
    return optimal_threshold, metrics_dict


def apply_threshold(logits: np.ndarray, fire_index: int, threshold: float) -> np.ndarray:
    """
    Apply a custom threshold to logits and return predictions.
    
    Args:
        logits: Model output logits (N, 2).
        fire_index: Index of the Fire class.
        threshold: Probability threshold for Fire class.
    
    Returns:
        preds: Predicted labels (fire_index or other_index).
    """
    fire_probs = np.exp(logits[:, fire_index]) / np.sum(np.exp(logits), axis=1)
    other_index = 1 - fire_index
    preds = np.where(fire_probs >= threshold, fire_index, other_index)
    return preds
