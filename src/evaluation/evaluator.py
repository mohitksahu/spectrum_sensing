"""Evaluation utilities for SpectraSense."""

from typing import Dict
import torch
from ..training.metrics import aggregate_metrics


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    def evaluate_batch(self, batch) -> Dict[str, torch.Tensor]:
        x, pu, mod, snr = batch
        x = x.to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
        return outputs

    def compute_metrics(self, outputs, targets) -> Dict[str, float]:
        return aggregate_metrics(outputs, targets)
