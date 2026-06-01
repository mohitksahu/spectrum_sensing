"""SNR analysis plotting utilities."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_snr_error_distribution(targets, predictions, title: str, save_path: str):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    errors = predictions - targets
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(errors, bins=40, color="#3498db", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Prediction Error (dB)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    out = Path(save_path)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_snr_accuracy(per_snr_metrics: Dict[str, Dict], title: str, save_path: str):
    labels = []
    values = []
    for key, metrics in per_snr_metrics.items():
        labels.append(key)
        values.append(metrics.get("pu_accuracy", float("nan")))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#27ae60")
    ax.set_title(title)
    ax.set_xlabel("SNR Bin")
    ax.set_ylabel("PU Accuracy")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = Path(save_path)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Dict[str, list], title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    out = Path(save_path)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
