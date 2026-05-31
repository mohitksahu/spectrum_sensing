"""Model checkpointing utilities."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages model checkpoints with best model tracking."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = True,
        max_keep: int = 5,
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            monitor: Metric to monitor for best model.
            mode: 'min' or 'max' - whether lower or higher is better.
            save_best: Whether to save the best model.
            save_last: Whether to save the last model.
            max_keep: Maximum number of checkpoints to keep.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.max_keep = max_keep
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1
        self.saved_checkpoints = []
    
    def is_better(self, current: float) -> bool:
        """Check if current value is better than best.
        
        Args:
            current: Current metric value.
            
        Returns:
            True if current is better than best.
        """
        if self.mode == "min":
            return current < self.best_value
        return current > self.best_value
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer state.
            scheduler: LR scheduler state.
            epoch: Current epoch.
            metrics: Current metrics dictionary.
            config: Training configuration.
            extra: Any extra data to save.
            
        Returns:
            Path to saved checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
        }
        if extra:
            checkpoint.update(extra)
        
        # Save last
        if self.save_last:
            last_path = self.checkpoint_dir / "last.pt"
            torch.save(checkpoint, last_path)
        
        # Check if best
        monitor_value = metrics.get(self.monitor, None)
        if monitor_value is not None and self.is_better(monitor_value):
            self.best_value = monitor_value
            self.best_epoch = epoch
            checkpoint["best_value"] = self.best_value
            checkpoint["best_epoch"] = self.best_epoch
            
            if self.save_best:
                best_path = self.checkpoint_dir / "best.pt"
                torch.save(checkpoint, best_path)
        
        # Save periodic checkpoint
        epoch_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(checkpoint, epoch_path)
        self.saved_checkpoints.append(epoch_path)
        
        # Remove old checkpoints
        while len(self.saved_checkpoints) > self.max_keep:
            old_path = self.saved_checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
        
        return str(epoch_path)
    
    @staticmethod
    def load(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            model: Model to load state into.
            optimizer: Optional optimizer to load state into.
            scheduler: Optional scheduler to load state into.
            device: Device to load tensors onto.
            
        Returns:
            Checkpoint dictionary with metadata.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint
