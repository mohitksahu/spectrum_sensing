"""ROC and precision-recall plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_roc_curve(targets, probs, title: str, save_path: str):
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = Path(save_path)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(targets, probs, title: str, save_path: str):
    precision, recall, _ = precision_recall_curve(targets, probs)
    pr_auc = average_precision_score(targets, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"AP = {pr_auc:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = Path(save_path)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
