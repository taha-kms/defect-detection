import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage import measure


def compute_image_auroc(labels, scores) -> float:
    """
    Compute AUROC for image-level anomaly detection.

    Args:
        labels (list[int]): 0 = normal, 1 = anomaly
        scores (list[float]): anomaly scores per image

    Returns:
        float: AUROC score
    """
    return roc_auc_score(labels, scores)


def compute_pixel_auroc(masks, score_maps) -> float:
    """
    Compute AUROC for pixel-level segmentation.

    Args:
        masks (ndarray): ground truth masks, shape [N, H, W]
        score_maps (ndarray): anomaly score maps, shape [N, H, W]

    Returns:
        float: AUROC score
    """
    masks = masks.astype(np.uint8).ravel()
    score_maps = score_maps.ravel()
    return roc_auc_score(masks, score_maps)


def compute_auprc(masks, score_maps) -> float:
    """
    Compute AUPRC (Area Under Precision-Recall Curve).

    Args:
        masks (ndarray): ground truth masks [N, H, W]
        score_maps (ndarray): predicted anomaly maps [N, H, W]

    Returns:
        float: AUPRC score
    """
    masks = masks.astype(np.uint8).ravel()
    score_maps = score_maps.ravel()
    return average_precision_score(masks, score_maps)


def compute_pro(masks, score_maps, max_fpr: float = 0.3, num_thresholds: int = 50) -> float:
    """
    Compute PRO (Per-Region-Overlap) metric.
    Measures how well predicted anomaly regions overlap with ground-truth defect regions.

    Args:
        masks (ndarray): ground truth masks [N, H, W]
        score_maps (ndarray): predicted anomaly maps [N, H, W]
        max_fpr (float): maximum false positive rate for evaluation
        num_thresholds (int): number of thresholds to sample

    Returns:
        float: PRO score
    """
    masks = masks.astype(np.uint8)
    n = len(masks)

    # Flatten into list of regions
    all_fprs = []
    all_pros = []

    thresholds = np.linspace(score_maps.min(), score_maps.max(), num_thresholds)

    for t in thresholds:
        binary_preds = (score_maps >= t).astype(np.uint8)

        tp_pixels = (binary_preds * masks).sum()
        fp_pixels = (binary_preds * (1 - masks)).sum()
        total_pos = masks.sum()
        total_neg = (1 - masks).sum()

        tpr = tp_pixels / (total_pos + 1e-8)
        fpr = fp_pixels / (total_neg + 1e-8)

        # per-region overlap
        pros = []
        for mask, pred in zip(masks, binary_preds):
            labeled_regions = measure.label(mask, connectivity=2)
            for region_id in np.unique(labeled_regions):
                if region_id == 0:
                    continue
                region_mask = (labeled_regions == region_id)
                intersection = (pred * region_mask).sum()
                union = region_mask.sum()
                pros.append(intersection / (union + 1e-8))
        pro = np.mean(pros) if pros else 0.0

        all_fprs.append(fpr)
        all_pros.append(pro)

    # Average PRO under FPR constraint
    all_fprs = np.array(all_fprs)
    all_pros = np.array(all_pros)

    valid = all_fprs <= max_fpr
    if valid.sum() == 0:
        return 0.0

    return all_pros[valid].mean()