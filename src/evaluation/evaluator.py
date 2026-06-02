"""Evaluation utilities for SpectraSense."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


class SpectraSenseEvaluator:
    """Run full model evaluation and persist structured metrics."""

    def __init__(self, model: torch.nn.Module, test_loader, device: torch.device, output_dir: str):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _collect(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()

        pu_probs = []
        pu_preds = []
        pu_targets = []
        mod_preds = []
        mod_targets = []
        snr_preds = []
        snr_targets = []
        snrs = []

        with torch.no_grad():
            for x, y_pu, y_mod, y_snr in self.test_loader:
                x = x.to(self.device)
                y_pu = y_pu.to(self.device)
                y_mod = y_mod.to(self.device)
                y_snr = y_snr.to(self.device)

                outputs = self.model(x)
                probs = torch.softmax(outputs["pu"], dim=1)

                pu_probs.append(probs[:, 1].detach().cpu().numpy())
                pu_preds.append(probs.argmax(dim=1).detach().cpu().numpy())
                pu_targets.append(y_pu.detach().cpu().numpy())
                mod_preds.append(outputs["mod"].argmax(dim=1).detach().cpu().numpy())
                mod_targets.append(y_mod.detach().cpu().numpy())
                snr_preds.append(outputs["snr"].view(-1).detach().cpu().numpy())
                snr_targets.append(y_snr.view(-1).detach().cpu().numpy())
                snrs.append(y_snr.view(-1).detach().cpu().numpy())

        return (
            np.concatenate(pu_probs),
            np.concatenate(pu_preds),
            np.concatenate(pu_targets),
            np.concatenate(mod_preds),
            np.concatenate(mod_targets),
            np.concatenate(snr_preds),
            np.concatenate(snr_targets),
            np.concatenate(snrs),
        )

    def evaluate(self) -> Dict[str, Any]:
        pu_probs, pu_preds, pu_targets, mod_preds, mod_targets, snr_preds, snr_targets, snrs = self._collect()

        pu_accuracy = float(accuracy_score(pu_targets, pu_preds))
        pu_precision = float(precision_score(pu_targets, pu_preds, zero_division=0))
        pu_recall = float(recall_score(pu_targets, pu_preds, zero_division=0))
        pu_f1 = float(f1_score(pu_targets, pu_preds, zero_division=0))
        pu_roc_auc = _safe_roc_auc(pu_targets, pu_probs)
        pu_pr_auc = _safe_pr_auc(pu_targets, pu_probs)

        mod_accuracy = float(accuracy_score(mod_targets, mod_preds))
        mod_per_class = {}
        for cls in np.unique(mod_targets):
            cls = int(cls)
            mask = mod_targets == cls
            mod_per_class[f"mod_acc_class_{cls}"] = float(accuracy_score(mod_targets[mask], mod_preds[mask])) if mask.any() else float("nan")

        snr_mae = float(mean_absolute_error(snr_targets, snr_preds))
        snr_rmse = float(np.sqrt(mean_squared_error(snr_targets, snr_preds)))
        snr_r2 = float(r2_score(snr_targets, snr_preds))

        pu_cm = confusion_matrix(pu_targets, pu_preds, labels=[0, 1]).tolist()
        mod_cm = confusion_matrix(mod_targets, mod_preds, labels=list(range(5))).tolist()

        per_snr: Dict[str, Dict[str, float]] = {}
        rounded_snr = np.round(snrs).astype(int)
        for snr_bin in sorted(np.unique(rounded_snr)):
            idx = rounded_snr == snr_bin
            if not np.any(idx):
                continue
            bin_name = f"{snr_bin}dB"
            bin_pu_targets = pu_targets[idx]
            bin_pu_preds = pu_preds[idx]
            tp = int(np.sum((bin_pu_targets == 1) & (bin_pu_preds == 1)))
            tn = int(np.sum((bin_pu_targets == 0) & (bin_pu_preds == 0)))
            fp = int(np.sum((bin_pu_targets == 0) & (bin_pu_preds == 1)))
            fn = int(np.sum((bin_pu_targets == 1) & (bin_pu_preds == 0)))
            tpr = _safe_rate(tp, tp + fn)
            tnr = _safe_rate(tn, tn + fp)
            if np.isnan(tpr) or np.isnan(tnr):
                balanced_acc = float("nan")
            else:
                balanced_acc = float((tpr + tnr) / 2.0)
            per_snr[bin_name] = {
                "n_samples": int(idx.sum()),
                "n_active": int(np.sum(bin_pu_targets == 1)),
                "n_inactive": int(np.sum(bin_pu_targets == 0)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "pu_tpr": tpr,
                "pu_tnr": tnr,
                "pu_balanced_accuracy": balanced_acc,
                "mod_accuracy": float(accuracy_score(mod_targets[idx], mod_preds[idx])),
                "snr_mae": float(mean_absolute_error(snr_targets[idx], snr_preds[idx])),
                "snr_rmse": float(np.sqrt(mean_squared_error(snr_targets[idx], snr_preds[idx]))),
            }

        low_idx = snrs < 8.0
        if np.any(low_idx):
            low_pu_targets = pu_targets[low_idx]
            low_pu_preds = pu_preds[low_idx]
            tp = int(np.sum((low_pu_targets == 1) & (low_pu_preds == 1)))
            tn = int(np.sum((low_pu_targets == 0) & (low_pu_preds == 0)))
            fp = int(np.sum((low_pu_targets == 0) & (low_pu_preds == 1)))
            fn = int(np.sum((low_pu_targets == 1) & (low_pu_preds == 0)))
            tpr = _safe_rate(tp, tp + fn)
            tnr = _safe_rate(tn, tn + fp)
            if np.isnan(tpr) or np.isnan(tnr):
                balanced_acc = float("nan")
            else:
                balanced_acc = float((tpr + tnr) / 2.0)
            per_snr["low_snr_<8dB"] = {
                "n_samples": int(low_idx.sum()),
                "n_active": int(np.sum(low_pu_targets == 1)),
                "n_inactive": int(np.sum(low_pu_targets == 0)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "pu_tpr": tpr,
                "pu_tnr": tnr,
                "pu_balanced_accuracy": balanced_acc,
                "mod_accuracy": float(accuracy_score(mod_targets[low_idx], mod_preds[low_idx])),
                "snr_mae": float(mean_absolute_error(snr_targets[low_idx], snr_preds[low_idx])),
                "snr_rmse": float(np.sqrt(mean_squared_error(snr_targets[low_idx], snr_preds[low_idx]))),
            }

        overall = {
            "pu_accuracy": pu_accuracy,
            "pu_precision": pu_precision,
            "pu_recall": pu_recall,
            "pu_f1": pu_f1,
            "pu_roc_auc": pu_roc_auc,
            "pu_pr_auc": pu_pr_auc,
            "mod_accuracy": mod_accuracy,
            "snr_mae": snr_mae,
            "snr_rmse": snr_rmse,
            "snr_r2": snr_r2,
            **mod_per_class,
        }

        param_breakdown = {}
        model_params = 0
        if hasattr(self.model, "get_parameter_breakdown"):
            param_breakdown = self.model.get_parameter_breakdown()
            model_params = int(param_breakdown.get("total", 0))
        else:
            model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            param_breakdown = {"total": model_params}

        results = {
            "overall": overall,
            "confusion_matrices": {"pu": pu_cm, "mod": mod_cm},
            "roc_data": {
                "pu_targets": pu_targets.tolist(),
                "pu_probs": pu_probs.tolist(),
            },
            "snr_data": {
                "targets": snr_targets.tolist(),
                "predictions": snr_preds.tolist(),
            },
            "per_snr": per_snr,
            "param_breakdown": param_breakdown,
            "model_params": model_params,
        }

        output_file = self.output_dir / "test_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results


class Evaluator(SpectraSenseEvaluator):
    """Backward-compatible alias."""

