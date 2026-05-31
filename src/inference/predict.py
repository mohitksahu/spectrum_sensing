"""Inference helpers for SpectraSense."""

from typing import Dict
import torch


def predict_batch(model: torch.nn.Module, x: torch.Tensor, device: str = "cpu") -> Dict[str, torch.Tensor]:
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        outputs = model(x)
    return outputs
