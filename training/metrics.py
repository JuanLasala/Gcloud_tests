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

    # Numerically stable softmax (avoid np.exp overflow)
    probs = softmax(logits, axis=1)
    fire_probs = probs[:, FIRE_INDEX]  # probability of Fire class

    # FIXED threshold during training (do NOT tune here)
    threshold = 0.5
    preds = (fire_probs >= threshold).astype(int)

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

        "f1_fire": f1.compute(
            predictions=preds,
            references=labels,
            average="binary",
            pos_label=FIRE_INDEX,
        )["f1"],

        # Threshold-independent (recommended for metric_for_best_model)
        "roc_auc": roc_auc.compute(
            prediction_scores=fire_probs,  # <-- use probabilities
            references=labels,
        )["roc_auc"],

        # PR-AUC (very informative for rare-event problems like wildfire)
        "pr_auc": average_precision_score(
            labels,
            fire_probs
        ),
    }
