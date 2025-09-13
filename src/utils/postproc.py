import numpy as np


def normalize_map(score_map: np.ndarray) -> np.ndarray:
    """Normalize anomaly map to [0, 1]."""
    min_val, max_val = score_map.min(), score_map.max()
    return (score_map - min_val) / (max_val - min_val + 1e-8)


def find_best_threshold(labels, scores):
    """
    Find best threshold using Youden's J statistic (maximizes TPR-FPR).

    Args:
        labels (list[int]): ground truth labels
        scores (list[float]): anomaly scores

    Returns:
        float: best threshold
    """
    labels = np.array(labels)
    scores = np.array(scores)

    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_thresh, best_j = thresholds[0], -1

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        j = tpr - fpr

        if j > best_j:
            best_j = j
            best_thresh = t

    return best_thresh