#!/usr/bin/env python3
"""Main training script for SpectraSense.

Executes the full training pipeline:
- Phase 1: Masked Spectrum Modeling (self-supervised)
- Phase 2: Supervised Multi-Task Fine-Tuning

Usage:
    python train.py --config configs/train.yaml
    python train.py --config configs/train.yaml --skip-phase1
    python train.py --config configs/train.yaml --resume checkpoints/last.pt
"""

import argparse
import time
from pathlib import Path

import torch
import yaml

from src.models.spectrasense import SpectraSense, build_spectrasense
from src.datasets.spectrum_dataset import create_dataloaders
from src.training.trainer import SpectraSenseTrainer
from src.utils.seed import set_seed
from src.utils.logger import ExperimentLogger


def load_processed_data(processed_dir: str):
    """Load preprocessed train/val/test datasets.
    
    Args:
        processed_dir: Path to processed data directory.
        
    Returns:
        Tuple of (train_data, val_data, test_data, class_weights) dictionaries.
    """
    processed_path = Path(processed_dir)
    
    train_data = torch.load(processed_path / "train.pt", weights_only=False)
    val_data = torch.load(processed_path / "val.pt", weights_only=False)
    test_data = torch.load(processed_path / "test.pt", weights_only=False)
    class_weights = torch.load(processed_path / "class_weights.pt", weights_only=False)
    
    # Convert tensors to numpy for DataLoader
    for data in [train_data, val_data, test_data]:
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].numpy()
    
    return train_data, val_data, test_data, class_weights


def get_device(config_device: str = "auto") -> torch.device:
    """Determine training device.
    
    Args:
        config_device: Device from config ('auto', 'cuda', 'cpu', 'mps').
        
    Returns:
        torch.device instance.
    """
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config_device)


def main():
    parser = argparse.ArgumentParser(description="Train SpectraSense model")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Training configuration file")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml",
                        help="Model configuration file")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Processed data directory")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Phase 1 pre-training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config, "r") as f:
        train_config = yaml.safe_load(f)["training"]
    
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)
    
    # Set seed
    seed = args.seed or train_config["seed"]
    set_seed(seed, deterministic=train_config["deterministic"])
    
    # Override phase 1 if skip flag
    if args.skip_phase1:
        train_config["phase1"]["enabled"] = False
    
    # Setup device
    device = get_device(train_config.get("device", "auto"))
    
    # Initialize logger
    logger = ExperimentLogger(
        experiment_name=train_config["experiment_name"],
        log_dir="logs",
        use_tensorboard=train_config["logging"]["tensorboard"],
        use_wandb=train_config["logging"]["wandb"],
        wandb_project=train_config["logging"].get("wandb_project", "spectrasense"),
        config={**train_config, **model_config},
    )
    
    logger.info(f"SpectraSense Training")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    
    # Load data
    logger.info(f"\nLoading processed data from {args.data_dir}...")
    train_data, val_data, test_data, class_weights = load_processed_data(args.data_dir)
    
    logger.info(f"  Train: {len(train_data['psds']):,} samples")
    logger.info(f"  Val: {len(val_data['psds']):,} samples")
    logger.info(f"  Test: {len(test_data['psds']):,} samples")
    
    # Create DataLoaders
    dl_config = train_config["dataloader"]
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=train_config["phase2"]["batch_size"],
        num_workers=dl_config["num_workers"],
        pin_memory=dl_config["pin_memory"],
        seed=seed,
        use_snr_balancing=True,
    )
    
    # Build model
    logger.info("\nBuilding SpectraSense model...")
    model = build_spectrasense(model_config)
    
    param_breakdown = model.get_parameter_breakdown()
    logger.info(f"  Total parameters: {param_breakdown['total']:,}")
    logger.info(f"  CNN Front-End: {param_breakdown['cnn_frontend']:,}")
    logger.info(f"  Tokenizer: {param_breakdown['tokenizer']:,}")
    logger.info(f"  Transformer: {param_breakdown['transformer']:,}")
    logger.info(f"  Task Heads: {param_breakdown['task_heads']:,}")
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"  Resumed from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Initialize trainer
    trainer = SpectraSenseTrainer(
        model=model,
        config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        pu_class_weights=class_weights["pu_weights"],
        mod_class_weights=class_weights["mod_weights"],
    )
    
    # Train
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training Pipeline")
    logger.info("=" * 60)
    
    results = trainer.train()
    
    # Save training history
    torch.save(
        {
            "history": results.get("phase2", {}).get("history", {}),
            "best_metrics": trainer.best_metrics,
            "config": {**train_config, **model_config},
            "param_breakdown": param_breakdown,
        },
        Path(train_config["checkpoint"]["dir"]) / "training_results.pt",
    )
    
    logger.info("\nTraining complete!")
    logger.close()


if __name__ == "__main__":
    main()