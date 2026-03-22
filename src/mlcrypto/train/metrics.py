from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.uint8)
    try:
        roc_auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        roc_auc = 0.5
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": roc_auc,
    }
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics.update(
        {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "true_positive_rate": float(tp / max(tp + fn, 1)),
            "true_negative_rate": float(tn / max(tn + fp, 1)),
        }
    )
    metrics["balanced_accuracy"] = (metrics["true_positive_rate"] + metrics["true_negative_rate"]) / 2.0
    return metrics
