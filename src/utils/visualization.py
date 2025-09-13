
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from matplotlib import cm
from PIL import Image


def overlay_heatmap(image: np.ndarray, score_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay anomaly score heatmap on top of an image.

    Args:
        image (np.ndarray): RGB image, shape [H, W, 3], values in [0,1]
        score_map (np.ndarray): anomaly map, shape [H, W], values in [0,1]
        alpha (float): blending factor (0=no heatmap, 1=only heatmap)

    Returns:
        np.ndarray: blended image [H, W, 3]
    """
    cmap = cm.jet
    heatmap = cmap(score_map)[:, :, :3]  # ignore alpha
    overlay = (1 - alpha) * image + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def save_image_grid(images, titles, save_path: Path, cols: int = 3, figsize=(12, 6)):
    """
    Save a grid of images with titles.

    Args:
        images (list[np.ndarray]): list of HxWx3 or HxW arrays
        titles (list[str]): list of titles for each subplot
        save_path (Path): file path to save the plot
        cols (int): number of columns in grid
        figsize (tuple): figure size
    """
    n_images = len(images)
    rows = int(np.ceil(n_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(rows, cols)

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if img.ndim == 2:  # grayscale
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    # Hide any unused subplots
    for idx in range(n_images, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📸 Saved {save_path}")


def plot_roc_curve(labels, scores, save_path: Path):
    """
    Plot ROC curve with AUC.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Image-level)")
    plt.legend(loc="lower right")
    plt.grid(True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC curve: {save_path}")


def plot_pr_curve(labels, scores, save_path: Path):
    """
    Plot Precision-Recall curve with AUPRC.
    """
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Image-level)")
    plt.legend(loc="lower left")
    plt.grid(True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved PR curve: {save_path}")