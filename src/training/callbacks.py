"""Training callbacks: EarlyStopping and LR scheduler helpers."""

from typing import Optional
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class EarlyStopping:
    """Stop training when monitored metric stops improving."""
    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
            return self.should_stop


def get_warmup_cosine_scheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
    """Return a LambdaLR scheduler with linear warmup and cosine decay."""
    def lr_lambda(step: int):
        step = step + 1
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay after warmup
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
