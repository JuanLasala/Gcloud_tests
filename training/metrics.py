import numpy as np
import evaluate
from scipy.special import softmax
from sklearn.metrics import average_precision_score

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")
roc_auc = evaluate.load("roc_auc")

FIRE_INDEX = 1  # <-- set this explicitly once and reuse everywhere

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = np.asarray(labels)

    # Numerically stable softmax (avoid np.exp overflow)
    probs = softmax(logits, axis=1)
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    fire_probs = np.clip(probs[:, FIRE_INDEX], 0.0, 1.0)  # probability of Fire class
    fire_probs = np.nan_to_num(fire_probs, nan=0.5, posinf=1.0, neginf=0.0)

    # FIXED threshold during training (do NOT tune here)
    threshold = 0.5
    preds = (fire_probs >= threshold).astype(int)

    f1_fire_value = f1.compute(
        predictions=preds,
        references=labels,
        average="binary",
        pos_label=FIRE_INDEX,
    )["f1"]

    try:
        roc_auc_value = roc_auc.compute(
            prediction_scores=fire_probs,
            references=labels,
        )["roc_auc"]
    except ValueError:
        # Keep training/evaluation running for rare degenerate eval batches.
        roc_auc_value = 0.5

    try:
        pr_auc_value = average_precision_score(labels, fire_probs)
        if not np.isfinite(pr_auc_value):
            pr_auc_value = 0.0
    except ValueError:
        pr_auc_value = 0.0

    return {
        # Optional (can remove if not useful for your case)
        "accuracy": accuracy.compute(
            predictions=preds,
            references=labels
        )["accuracy"],

        # Focus on FIRE class specifically (NOT weighted)
        "precision_fire": precision.compute(
            predictions=preds,
            references=labels,
            average="binary",
            pos_label=FIRE_INDEX,  # <-- critical
        )["precision"],

        "recall_fire": recall.compute(
            predictions=preds,
            references=labels,
            average="binary",
            pos_label=FIRE_INDEX,
        )["recall"],

        "f1_fire": f1_fire_value,

        # Alias for Trainer metric_for_best_model="f1"
        "f1": f1_fire_value,

        # Threshold-independent (recommended for metric_for_best_model)
        "roc_auc": float(roc_auc_value),

        # PR-AUC (very informative for rare-event problems like wildfire)
        "pr_auc": float(pr_auc_value),
    }
