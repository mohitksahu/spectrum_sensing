#!/usr/bin/env python3
"""Data preparation script for SpectraSense.

Loads raw .pth files, performs preprocessing, splits data, fits scaler,
and saves processed datasets.

Usage:
    python prepare_data.py --config configs/dataset.yaml
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.datasets.preprocessing import (
    load_all_data,
    bin_snrs,
    remove_non_finite,
    stratified_split,
    fit_scaler,
    apply_scaler,
    compute_class_weights,
    compute_dataset_statistics,
)
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Prepare SpectraSense dataset")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml",
                        help="Dataset configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["dataset"]
    
    set_seed(args.seed)
    
    print("=" * 60)
    print("SpectraSense Data Preparation Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Load raw data
    print("\n[1/6] Loading raw data...")
    raw_dir = config["raw_dir"]
    data = load_all_data(raw_dir, config)
    
    psds = data["psds"]
    pu_labels = data["pu_labels"]
    mod_labels = data["mod_labels"]
    snrs = data["snrs"]
    
    print(f"\n  Total samples loaded: {len(psds):,}")
    print(f"  PSD shape: {psds.shape}")
    print(f"  SNR range: [{snrs.min():.1f}, {snrs.max():.1f}] dB")
    
    # Step 2: Bin SNR values and filter
    print("\n[2/6] Binning SNR values and filtering out-of-range samples...")
    # These bins are hardcoded as per the user's specification.
    # In a real project, they might be moved to a config file.
    target_bins = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    psds, pu_labels, mod_labels, snrs = bin_snrs(psds, pu_labels, mod_labels, snrs, target_bins=target_bins)
    
    # Step 3: Remove non-finite samples
    print("\n[3/6] Removing non-finite samples...")
    psds, pu_labels, mod_labels, snrs = remove_non_finite(psds, pu_labels, mod_labels, snrs)
    print(f"  Samples after cleaning: {len(psds):,}")
    
    # Step 4: Train/Val/Test split (BEFORE normalization)
    print("\n[4/6] Performing stratified train/val/test split...")
    split_config = config["split"]
    splits = stratified_split(
        psds, pu_labels, mod_labels, snrs,
        train_ratio=split_config["train"],
        val_ratio=split_config["val"],
        test_ratio=split_config["test"],
        seed=split_config["random_seed"],
        stratify_by=split_config.get("stratify_by", "modulation_pu"),
    )
    
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data['psds']):,} samples")
    
    # Step 5: Fit scaler on training data only
    print("\n[5/6] Fitting StandardScaler on training data...")
    clip_min = config["preprocessing"]["clip_min"]
    clip_max = config["preprocessing"]["clip_max"]
    
    scaler, train_psds_norm = fit_scaler(splits["train"]["psds"], clip_min, clip_max)
    splits["train"]["psds"] = train_psds_norm
    
    # Apply to val and test
    splits["val"]["psds"] = apply_scaler(scaler, splits["val"]["psds"], clip_min, clip_max)
    splits["test"]["psds"] = apply_scaler(scaler, splits["test"]["psds"], clip_min, clip_max)
    
    print(f"  Scaler mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  Scaler std range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
    print(f"  Train PSD after norm: [{train_psds_norm.min():.2f}, {train_psds_norm.max():.2f}]")
    
    # Step 6: Compute class weights
    print("\n[6/6] Computing class weights...")
    pu_weights = compute_class_weights(splits["train"]["pu_labels"], num_classes=2)
    mod_weights = compute_class_weights(splits["train"]["mod_labels"], num_classes=5)
    
    print(f"  PU weights: {pu_weights.numpy()}")
    print(f"  Mod weights: {mod_weights.numpy()}")
    
    # Step 6: Save processed data
    print("\n[6/6] Saving processed data...")
    processed_dir = Path(config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits as .pt files
    for split_name, split_data in splits.items():
        save_dict = {
            "psds": torch.from_numpy(split_data["psds"]).float(),
            "pu_labels": torch.from_numpy(split_data["pu_labels"]).long(),
            "mod_labels": torch.from_numpy(split_data["mod_labels"]).long(),
            "snrs": torch.from_numpy(split_data["snrs"]).float(),
        }
        torch.save(save_dict, processed_dir / f"{split_name}.pt")
        print(f"  Saved {split_name}.pt ({len(split_data['psds']):,} samples)")
    
    # Save scaler
    scaler_path = processed_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler.pkl")
    
    # Save class weights
    weights_dict = {
        "pu_weights": pu_weights,
        "mod_weights": mod_weights,
    }
    torch.save(weights_dict, processed_dir / "class_weights.pt")
    print(f"  Saved class_weights.pt")
    
    # Save metadata
    metadata_dir = Path(config["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    stats = compute_dataset_statistics(splits)
    stats["preprocessing"] = {
        "scaler": "StandardScaler",
        "clip_range": [clip_min, clip_max],
        "seed": args.seed,
    }
    
    with open(metadata_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))
    print(f"  Saved dataset_stats.json")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Data preparation complete in {elapsed:.1f}s")
    print(f"Processed files saved to: {processed_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()