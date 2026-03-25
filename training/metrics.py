import evaluate
import numpy as np
from sklearn.metrics import roc_auc_score

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

POS_LABEL = 0  # Set the positive class label (integer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred  # Unpack logits and labels
    preds = np.argmax(logits, axis=1)  # Convert logits to predictions by taking the argmax along axis 1
    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    metrics = {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="binary", pos_label=POS_LABEL)["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="binary", pos_label=POS_LABEL)["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary", pos_label=POS_LABEL)["f1"],
    }
    # ROC AUC (handle binary and multiclass)
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr")
        metrics["roc_auc"] = auc
    except Exception:
        metrics["roc_auc"] = float('nan')
    return metrics
