"""Training and evaluation metrics for SpectraSense."""

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, mean_squared_error


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for classification logits."""
    with torch.no_grad():
        pred_labels = preds.argmax(dim=1)
        return float((pred_labels == targets).float().mean().item())


def compute_pu_metrics(pu_logits: torch.Tensor, pu_targets: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        pu_probs = torch.softmax(pu_logits, dim=1)
        pu_pred = pu_probs.argmax(dim=1)
        pu_true = pu_targets.detach().view(-1).cpu().numpy()
        pu_pred_np = pu_pred.detach().view(-1).cpu().numpy()
        pu_prob_np = pu_probs[:, 1].detach().view(-1).cpu().numpy()

    pu_acc = float((pu_pred.detach() == pu_targets.detach()).float().mean().item())
    pu_f1 = float(f1_score(pu_true, pu_pred_np, average="binary", zero_division=0))
    try:
        pu_auc = float(roc_auc_score(pu_true, pu_prob_np))
    except ValueError:
        pu_auc = float("nan")

    return {"pu_acc": pu_acc, "pu_f1": pu_f1, "pu_auc": pu_auc}


def compute_mod_metrics(mod_logits: torch.Tensor, mod_targets: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        mod_pred = mod_logits.argmax(dim=1)
        mod_true = mod_targets.detach().view(-1).cpu().numpy()
        mod_pred_np = mod_pred.detach().view(-1).cpu().numpy()

    mod_acc = float((mod_pred.detach() == mod_targets.detach()).float().mean().item())
    mod_f1 = float(f1_score(mod_true, mod_pred_np, average="macro", zero_division=0))
    return {"mod_acc": mod_acc, "mod_f1": mod_f1}


def compute_snr_metrics(snr_preds: torch.Tensor, snr_targets: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        snr_preds = snr_preds.view(-1)
        snr_targets = snr_targets.view(-1)
        snr_pred_np = snr_preds.detach().cpu().numpy()
        snr_true_np = snr_targets.detach().cpu().numpy()

    mse = float(mean_squared_error(snr_true_np, snr_pred_np))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(snr_true_np, snr_pred_np))
    return {"snr_mse": mse, "snr_rmse": rmse, "snr_mae": mae}


def aggregate_metrics(preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics = {}
    metrics.update(compute_pu_metrics(preds["pu"], targets["pu"]))
    metrics.update(compute_mod_metrics(preds["mod"], targets["mod"]))
    metrics.update(compute_snr_metrics(preds["snr"], targets["snr"]))
    return metrics
