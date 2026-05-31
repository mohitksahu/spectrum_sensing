"""Logging utilities for experiment tracking."""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch


class ExperimentLogger:
    """Unified experiment logger supporting console, file, TensorBoard, and W&B."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "spectrasense",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment.
            log_dir: Directory for log files.
            use_tensorboard: Enable TensorBoard logging.
            use_wandb: Enable Weights & Biases logging.
            wandb_project: W&B project name.
            config: Experiment configuration dict.
        """
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{experiment_name}_{timestamp}"
        self.log_dir = Path(log_dir) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logger
        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / "training.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                self.logger.info(f"TensorBoard logging enabled at {self.log_dir / 'tensorboard'}")
            except ImportError:
                self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")
        
        # Weights & Biases
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=self.run_name,
                    config=config,
                    dir=str(self.log_dir),
                )
                self.logger.info("Weights & Biases logging enabled")
            except ImportError:
                self.logger.warning("wandb not available. Install with: pip install wandb")
    
    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)
    
    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)
    
    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard and W&B.
        
        Args:
            tag: Metric name.
            value: Metric value.
            step: Global step.
        """
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        if self.wandb_run:
            import wandb
            wandb.log({tag: value}, step=step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars under a main tag.
        
        Args:
            main_tag: Group name for the scalars.
            tag_scalar_dict: Dictionary of tag-value pairs.
            step: Global step.
        """
        if self.tb_writer:
            self.tb_writer.add_scalars(main_tag, tag_scalar_dict, step)
        if self.wandb_run:
            import wandb
            wandb.log({f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}, step=step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram to TensorBoard.
        
        Args:
            tag: Histogram name.
            values: Tensor of values.
            step: Global step.
        """
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)
    
    def log_figure(self, tag: str, figure, step: int) -> None:
        """Log a matplotlib figure.
        
        Args:
            tag: Figure name.
            figure: Matplotlib figure object.
            step: Global step.
        """
        if self.tb_writer:
            self.tb_writer.add_figure(tag, figure, step)
        if self.wandb_run:
            import wandb
            wandb.log({tag: wandb.Image(figure)}, step=step)
    
    def close(self) -> None:
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()
