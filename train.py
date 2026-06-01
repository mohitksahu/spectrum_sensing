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
import torch.nn as nn
import yaml

from src.models.spectrasense import SpectraSense, build_spectrasense
from src.models.spectrasense import PSDDenoisingHead, compute_denoising_target
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


def _extract_psd_batch(batch):
    """Extract PSD tensor with shape (B, 1, 192) from DataLoader batch."""
    if isinstance(batch, dict):
        x = batch["psd"]
    else:
        x = batch[0]
    if x.dim() == 2:
        x = x.unsqueeze(1)
    return x


def run_phase1a_denoising_pretrain(model, train_loader, device, logger, epochs: int = 15):
    """Phase 1A: self-supervised denoising pre-training."""
    logger.info("Starting Phase 1A: Raw PSD Denoising Pre-Training...")

    model_d_model = int(getattr(model, "d_model", 96))
    denoising_head = PSDDenoisingHead(d_model=model_d_model, output_bins=192).to(device)

    phase1a_params = (
        list(model.cnn.parameters())
        + list(model.tokenizer.parameters())
        + list(model.transformer.parameters())
        + list(denoising_head.parameters())
    )

    optimizer_1a = torch.optim.Adam(phase1a_params, lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    criterion_denoise = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        denoising_head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            psd_batch = _extract_psd_batch(batch).to(device)

            optimizer_1a.zero_grad(set_to_none=True)

            _, _, _, e_cls = model(psd_batch, return_features=True)
            psd_recon = denoising_head(e_cls)

            psd_raw_2d = psd_batch.squeeze(1)
            psd_target = compute_denoising_target(psd_raw_2d, pool_kernel=11)

            loss = criterion_denoise(psd_recon, psd_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phase1a_params, max_norm=1.0)
            optimizer_1a.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        logger.info(f"Phase 1A Epoch [{epoch + 1}/{epochs}] | Denoise MSE: {avg_loss:.4f}")

    phase1a_dir = Path("checkpoints") / "phase1a"
    phase1a_dir.mkdir(parents=True, exist_ok=True)
    phase1a_ckpt = phase1a_dir / "slm_phase1a_best.pt"
    torch.save({"model_state": model.state_dict(), "epoch": epochs, "phase": "1a"}, phase1a_ckpt)

    del denoising_head, optimizer_1a
    logger.info("Phase 1A complete. Denoising head discarded. Proceeding to Phase 1B...")

    # Phase 1B warm start load hook.
    checkpoint_1a = torch.load(phase1a_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_1a["model_state"])
    logger.info("Loaded Phase 1A checkpoint for Phase 1B warm start.")


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
    model = model.to(device)
    
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
        model = model.to(device)
        logger.info(f"  Resumed from epoch {checkpoint.get('epoch', 'unknown')}")

    # Phase 1A pre-training runs before existing supervised training.
    if train_config.get("phase1", {}).get("enabled", False) and not args.resume:
        phase1a_epochs = int(train_config.get("phase1", {}).get("epochs_phase1a", 15))
        run_phase1a_denoising_pretrain(model, train_loader, device, logger, epochs=phase1a_epochs)
    
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