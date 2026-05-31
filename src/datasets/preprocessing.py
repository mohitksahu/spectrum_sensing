"""Data preprocessing pipeline for SpectraSense.

Handles loading raw .pth files, normalization, clipping, filtering,
train/val/test splitting, and class weight computation.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Modulation class mapping
MODULATION_MAP = {
    "bpsk": 0,
    "qpsk": 1,
    "8psk": 2,
    "16qam": 3,
    "dqpsk": 4,
}

# Reverse mapping
MODULATION_NAMES = {v: k.upper() for k, v in MODULATION_MAP.items()}


def _to_psd_192(psd_vector: Any) -> np.ndarray:
    """Convert raw PSD sample to a flat float32 vector of length 192."""
    if isinstance(psd_vector, torch.Tensor):
        psd_vector = psd_vector.numpy()

    arr = np.asarray(psd_vector, dtype=np.float32)
    arr = np.squeeze(arr)
    arr = arr.reshape(-1)

    if arr.shape[0] == 192:
        return arr
    if arr.shape[0] > 192:
        return arr[:192]

    padded = np.zeros(192, dtype=np.float32)
    padded[:arr.shape[0]] = arr
    return padded


def load_binned_pth(filepath: str, modulation_label: int) -> List[Tuple[np.ndarray, int, int, float]]:
    """Load a binned .pth file and extract all samples.
    
    Each .pth file has structure:
        {'pairs_by_bin': {snr_value: [(psd_vector, pu_label, snr), ...]}}
    
    Args:
        filepath: Path to .pth file.
        modulation_label: Integer modulation class label.
        
    Returns:
        List of (psd_vector, pu_label, mod_label, snr) tuples.
    """
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    samples = []
    
    pairs_by_bin = data.get("pairs_by_bin", data)
    
    for snr_bin, pairs in pairs_by_bin.items():
        snr_value = float(snr_bin)
        for item in pairs:
            if len(item) >= 3:
                psd_vector = item[0]
                pu_label = int(item[1])
                snr = float(item[2])
            elif len(item) == 2:
                psd_vector = item[0]
                pu_label = int(item[1])
                snr = snr_value
            else:
                continue
            
            psd_vector = _to_psd_192(psd_vector)
            samples.append((psd_vector, pu_label, modulation_label, snr))
    
    return samples


def load_log_pth(filepath: str, modulation_label: int) -> List[Tuple[np.ndarray, int, int, float]]:
    """Load a log-format .pth file (from new dataset directory).
    
    Log format files may have slightly different internal structure.
    
    Args:
        filepath: Path to .pth file.
        modulation_label: Integer modulation class label.
        
    Returns:
        List of (psd_vector, pu_label, mod_label, snr) tuples.
    """
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    samples = []
    
    # Handle different possible structures
    if isinstance(data, dict):
        if "pairs_by_bin" in data:
            return load_binned_pth(filepath, modulation_label)
        
        # Try log format: {snr: [samples...]}
        for key, value in data.items():
            try:
                snr_value = float(key)
            except (ValueError, TypeError):
                continue
            
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        psd_vector = item[0]
                        pu_label = int(item[1]) if len(item) > 1 else 1
                        snr = float(item[2]) if len(item) > 2 else snr_value
                    elif isinstance(item, torch.Tensor):
                        psd_vector = item
                        pu_label = 1
                        snr = snr_value
                    else:
                        continue
                    
                    psd_vector = _to_psd_192(psd_vector)
                    samples.append((psd_vector, pu_label, modulation_label, snr))
    
    return samples


def load_all_data(raw_dir: str, config: Dict) -> Dict[str, np.ndarray]:
    """Load all raw data from primary and secondary sources.
    
    Args:
        raw_dir: Path to raw data directory.
        config: Dataset configuration dictionary.
        
    Returns:
        Dictionary with 'psds', 'pu_labels', 'mod_labels', 'snrs' arrays.
    """
    raw_path = Path(raw_dir)
    all_samples = []
    
    # Load primary source (Secondary_User directory)
    primary_dir = raw_path / config.get("primary_source", "Secondary_User")
    
    primary_files = config.get("primary_files", {})
    
    # Standard binned files
    file_mod_map = {
        "psd_binned_by_snr_bpsk.pth": 0,    # BPSK
        "psd_binned_by_snr_qpsk.pth": 1,    # QPSK
        "psd_binned_by_snr_16qam.pth": 3,   # 16QAM
    }
    
    for filename, mod_label in file_mod_map.items():
        filepath = primary_dir / filename
        if filepath.exists():
            print(f"  Loading {filepath.name} (mod={mod_label})...")
            samples = load_binned_pth(str(filepath), mod_label)
            all_samples.extend(samples)
            print(f"    → {len(samples)} samples loaded")
    
    # Log format files
    log_file_map = {
        "psd_log_8psk.pth": 2,     # 8PSK
        "psd_log_16qam.pth": 3,    # 16QAM (additional)
    }
    
    for filename, mod_label in log_file_map.items():
        filepath = primary_dir / filename
        if filepath.exists():
            print(f"  Loading {filepath.name} (mod={mod_label}, log format)...")
            samples = load_log_pth(str(filepath), mod_label)
            all_samples.extend(samples)
            print(f"    → {len(samples)} samples loaded")
    
    # Load secondary source (New Dataset directory)
    secondary_dir = raw_path / config.get("secondary_source", "New_Dataset")
    if secondary_dir.exists():
        print(f"\n  Loading secondary source: {secondary_dir}")
        for pth_file in sorted(secondary_dir.glob("*.pth")):
            # Determine modulation from filename
            fname_lower = pth_file.stem.lower()
            mod_label = None
            for mod_name, mod_idx in MODULATION_MAP.items():
                if mod_name in fname_lower:
                    mod_label = mod_idx
                    break
            
            if mod_label is None:
                # Try to infer DQPSK
                if "dqpsk" in fname_lower or "dpsk" in fname_lower:
                    mod_label = 4
                else:
                    print(f"    Skipping {pth_file.name} (unknown modulation)")
                    continue
            
            print(f"  Loading {pth_file.name} (mod={mod_label})...")
            try:
                samples = load_log_pth(str(pth_file), mod_label)
                all_samples.extend(samples)
                print(f"    → {len(samples)} samples loaded")
            except Exception as e:
                print(f"    Error loading {pth_file.name}: {e}")
    
    # Convert to numpy arrays
    if not all_samples:
        raise ValueError("No samples loaded! Check data directory paths.")
    
    psds = np.stack([s[0] for s in all_samples], axis=0).astype(np.float32)
    if psds.ndim > 2:
        psds = psds.reshape(psds.shape[0], -1)
    if psds.shape[1] != 192:
        raise ValueError(f"Expected PSD vectors of length 192, got shape {psds.shape}")
    pu_labels = np.array([s[1] for s in all_samples], dtype=np.int64)
    mod_labels = np.array([s[2] for s in all_samples], dtype=np.int64)
    snrs = np.array([s[3] for s in all_samples], dtype=np.float32)
    
    return {
        "psds": psds,
        "pu_labels": pu_labels,
        "mod_labels": mod_labels,
        "snrs": snrs,
    }


def remove_non_finite(
    psds: np.ndarray,
    pu_labels: np.ndarray,
    mod_labels: np.ndarray,
    snrs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove samples with non-finite PSD values.
    
    Args:
        psds: PSD array of shape (N, 192).
        pu_labels: PU label array.
        mod_labels: Modulation label array.
        snrs: SNR array.
        
    Returns:
        Filtered arrays with non-finite samples removed.
    """
    finite_mask = np.isfinite(psds).reshape(psds.shape[0], -1).all(axis=1)
    n_removed = (~finite_mask).sum()
    if n_removed > 0:
        print(f"  Removed {n_removed} non-finite samples")
    
    return (
        psds[finite_mask],
        pu_labels[finite_mask],
        mod_labels[finite_mask],
        snrs[finite_mask],
    )


def stratified_split(
    psds: np.ndarray,
    pu_labels: np.ndarray,
    mod_labels: np.ndarray,
    snrs: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Perform stratified train/val/test split based on SNR bins.
    
    Args:
        psds: PSD array of shape (N, 192).
        pu_labels: PU labels.
        mod_labels: Modulation labels.
        snrs: SNR values.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        test_ratio: Test set ratio.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits.
    """
    # Create stratification key combining SNR bin and modulation
    snr_bins = np.round(snrs / 2) * 2  # Round to nearest even for binning
    stratify_key = snr_bins * 10 + mod_labels  # Composite key
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    
    try:
        train_idx, valtest_idx = train_test_split(
            np.arange(len(psds)),
            test_size=val_test_ratio,
            random_state=seed,
            stratify=stratify_key,
        )
    except ValueError:
        # Fall back to non-stratified if some classes have too few samples
        print("  Warning: Falling back to non-stratified split (insufficient samples in some strata)")
        train_idx, valtest_idx = train_test_split(
            np.arange(len(psds)),
            test_size=val_test_ratio,
            random_state=seed,
        )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    
    try:
        val_idx, test_idx = train_test_split(
            valtest_idx,
            test_size=relative_test_ratio,
            random_state=seed,
            stratify=stratify_key[valtest_idx],
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            valtest_idx,
            test_size=relative_test_ratio,
            random_state=seed,
        )
    
    return {
        "train": {
            "psds": psds[train_idx],
            "pu_labels": pu_labels[train_idx],
            "mod_labels": mod_labels[train_idx],
            "snrs": snrs[train_idx],
        },
        "val": {
            "psds": psds[val_idx],
            "pu_labels": pu_labels[val_idx],
            "mod_labels": mod_labels[val_idx],
            "snrs": snrs[val_idx],
        },
        "test": {
            "psds": psds[test_idx],
            "pu_labels": pu_labels[test_idx],
            "mod_labels": mod_labels[test_idx],
            "snrs": snrs[test_idx],
        },
    }


def fit_scaler(
    train_psds: np.ndarray,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
) -> Tuple[StandardScaler, np.ndarray]:
    """Fit StandardScaler on training data only (per-bin normalization).
    
    Args:
        train_psds: Training PSD array of shape (N_train, 192).
        clip_min: Minimum clip value after normalization.
        clip_max: Maximum clip value after normalization.
        
    Returns:
        Tuple of (fitted scaler, normalized training PSDs).
    """
    scaler = StandardScaler()
    scaler.fit(train_psds)
    
    normalized = scaler.transform(train_psds).astype(np.float32)
    normalized = np.clip(normalized, clip_min, clip_max)
    
    return scaler, normalized


def apply_scaler(
    scaler: StandardScaler,
    psds: np.ndarray,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
) -> np.ndarray:
    """Apply fitted scaler to data with clipping.
    
    Args:
        scaler: Fitted StandardScaler.
        psds: PSD array to transform.
        clip_min: Minimum clip value.
        clip_max: Maximum clip value.
        
    Returns:
        Normalized and clipped PSD array.
    """
    normalized = scaler.transform(psds).astype(np.float32)
    normalized = np.clip(normalized, clip_min, clip_max)
    return normalized


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced classification.
    
    Formula: w_c = N_total / (2 * N_c)
    
    Args:
        labels: Array of class labels.
        num_classes: Total number of classes.
        
    Returns:
        Tensor of class weights.
    """
    n_total = len(labels)
    weights = []
    for c in range(num_classes):
        n_c = (labels == c).sum()
        if n_c > 0:
            weights.append(n_total / (2.0 * n_c))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_dataset_statistics(
    splits: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics.
    
    Args:
        splits: Dictionary with 'train', 'val', 'test' data splits.
        
    Returns:
        Statistics dictionary.
    """
    stats = {}
    
    for split_name, split_data in splits.items():
        psds = split_data["psds"]
        pu_labels = split_data["pu_labels"]
        mod_labels = split_data["mod_labels"]
        snrs = split_data["snrs"]
        
        stats[split_name] = {
            "total_samples": len(psds),
            "pu_distribution": {
                "idle": int((pu_labels == 0).sum()),
                "active": int((pu_labels == 1).sum()),
            },
            "mod_distribution": {
                MODULATION_NAMES.get(i, f"class_{i}"): int((mod_labels == i).sum())
                for i in range(5)
            },
            "snr_distribution": {
                f"{int(snr)}dB": int((np.abs(snrs - snr) < 0.5).sum())
                for snr in sorted(np.unique(np.round(snrs)))
            },
            "psd_stats": {
                "mean": float(psds.mean()),
                "std": float(psds.std()),
                "min": float(psds.min()),
                "max": float(psds.max()),
            },
        }
    
    stats["total"] = sum(s["total_samples"] for s in stats.values())
    
    return stats
