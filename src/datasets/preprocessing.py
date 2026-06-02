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


def _infer_modulation_from_path(path: Path) -> Optional[int]:
    """Infer modulation label from the immediate parent directory name."""
    # Step 2: Fix modulation labeling to use immediate parent folder name.
    # The parent directory name must be one of: bpsk, qpsk, 8psk, 16qam, dqpsk.
    parent_name = path.parent.name.lower()
    mod_id = MODULATION_MAP.get(parent_name)
    
    if mod_id is None and path.suffix == ".pth":
        print(f"  [WARNING] Skipping {path.name}: parent folder '{parent_name}' is not a valid modulation")
        
    return mod_id


def _rank_raw_file(path: Path) -> Tuple[int, str]:
    """Sort candidate raw files so we pick one canonical source per folder."""
    name = path.name.lower()
    # Prefer dataset.pth, then binned variants
    if name == "dataset.pth":
        rank = 0
    elif name == "dataset_binned.pth":
        rank = 1
    elif name.startswith("psd_binned"):
        rank = 2
    elif name.startswith("psd_log"):
        rank = 3
    else:
        rank = 4
    return rank, name


def _load_raw_sample_dict(data: Dict[str, Any], modulation_label: int) -> List[Tuple[np.ndarray, int, int, float]]:
    """Convert a raw sample dictionary into standardized sample tuples."""
    samples: List[Tuple[np.ndarray, int, int, float]] = []

    if "pairs_by_bin" in data:
        pairs_by_bin = data.get("pairs_by_bin", {})
        for snr_bin, pairs in pairs_by_bin.items():
            snr_value = float(snr_bin)
            for item in pairs:
                if len(item) < 2:
                    continue

                psd_vector = item[0]

                # New binned dumps are typically shaped like:
                #   (psd_vector, iq_samples, pu_label)
                # while older variants may use (psd_vector, pu_label, snr).
                pu_label = None
                snr = snr_value

                if len(item) >= 3 and np.isscalar(item[2]):
                    pu_label = int(item[2])
                elif np.isscalar(item[1]):
                    pu_label = int(item[1])

                if pu_label is None:
                    continue

                samples.append((_to_psd_192(psd_vector), pu_label, modulation_label, snr))
        return samples

    psds = data.get("psds")
    pu_flags = data.get("pu_flags")
    snrs = data.get("snrs")

    if psds is None or pu_flags is None or snrs is None:
        return samples

    for psd_vector, pu_label, snr in zip(psds, pu_flags, snrs):
        samples.append((_to_psd_192(psd_vector), int(pu_label), modulation_label, float(snr)))

    return samples


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


def load_raw_pth(filepath: str, modulation_label: int) -> List[Tuple[np.ndarray, int, int, float]]:
    """Load a raw .pth file and extract all samples."""
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        return []
    return _load_raw_sample_dict(data, modulation_label)


def load_all_data(raw_dir: str, config: Dict) -> Dict[str, np.ndarray]:
    """Load all raw data from primary and secondary sources.
    
    Args:
        raw_dir: Path to raw data directory.
        config: Dataset configuration dictionary.
        
    Returns:
        Dictionary with 'psds', 'pu_labels', 'mod_labels', 'snrs' arrays.
    """
    raw_path = Path(raw_dir)
    all_samples: List[Tuple[np.ndarray, int, int, float]] = []

    grouped_files: Dict[Tuple[Path, int], Path] = {}
    skipped_files: List[Path] = []

    for pth_file in sorted(raw_path.rglob("*.pth")):
        if not pth_file.is_file() or pth_file.stat().st_size == 0:
            continue
            
        # Step 1: Fix the loader to skip Symbol1 completely
        if "Symbol1" in pth_file.parts or "Symbol1" in pth_file.name:
            continue

        mod_label = _infer_modulation_from_path(pth_file)
        if mod_label is None:
            skipped_files.append(pth_file)
            continue

        group_key = (pth_file.parent, mod_label)
        chosen = grouped_files.get(group_key)
        if chosen is None or _rank_raw_file(pth_file) < _rank_raw_file(chosen):
            grouped_files[group_key] = pth_file

    if skipped_files:
        print(f"  Skipping {len(skipped_files)} raw .pth files (invalid modulation folders)")

    for (mod_dir, mod_label), pth_file in sorted(grouped_files.items(), key=lambda item: str(item[0][0])):
        print(f"  Loading {pth_file.relative_to(raw_path)} (mod={mod_label})...")
        try:
            samples = load_raw_pth(str(pth_file), mod_label)
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


def bin_snrs(
    psds: np.ndarray,
    pu_labels: np.ndarray,
    mod_labels: np.ndarray,
    snrs: np.ndarray,
    target_bins: List[int] = [4, 6, 8, 10, 12, 14, 16, 18, 20],
    tolerance: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map raw SNR values to the nearest specified bin and filter out-of-range samples.
    
    Args:
        psds: PSD array of shape (N, 192).
        pu_labels: PU labels.
        mod_labels: Modulation labels.
        snrs: Raw continuous SNR values.
        target_bins: List of specified SNR bins.
        tolerance: Maximum allowed distance to the nearest bin.
        
    Returns:
        Filtered and binned arrays.
    """
    bins = np.array(target_bins)
    binned_snrs = []
    keep_indices = []
    
    for i, snr in enumerate(snrs):
        # Find nearest bin
        distances = np.abs(bins - snr)
        nearest_idx = np.argmin(distances)
        min_dist = distances[nearest_idx]
        
        if min_dist <= tolerance:
            binned_snrs.append(bins[nearest_idx])
            keep_indices.append(i)
            
    keep_indices = np.array(keep_indices)
    
    if len(keep_indices) == 0:
        print("  Warning: No samples left after SNR binning!")
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    print(f"  Binned SNR: kept {len(keep_indices)}/{len(snrs)} samples")
    
    return (
        psds[keep_indices],
        pu_labels[keep_indices],
        mod_labels[keep_indices],
        np.array(binned_snrs, dtype=np.float32),
    )


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
    stratify_by: str = "modulation_pu",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Perform stratified train/val/test split.
    
    Args:
        psds: PSD array of shape (N, 192).
        pu_labels: PU labels.
        mod_labels: Modulation labels.
        snrs: SNR values.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        test_ratio: Test set ratio.
        seed: Random seed for reproducibility.
        stratify_by: Split key strategy. Supported values are ``modulation_pu``
            (default) and ``snr``.
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits.
    """
    if stratify_by == "snr":
        # Coarse SNR bucket so the split remains stable on wider SNR ranges.
        snr_edges = np.quantile(snrs, [0.2, 0.4, 0.6, 0.8])
        snr_bins = np.digitize(snrs, snr_edges, right=True)
        stratify_key = snr_bins * 100 + mod_labels * 2 + pu_labels
    else:
        # The previous SNR-only key was too fine-grained for the new raw data
        # and could fall back to random splitting. Modulation + PU keeps every
        # class represented while still balancing the binary occupancy target.
        stratify_key = mod_labels * 2 + pu_labels
    
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
    
    Formula: w_c = N_total / (num_classes * N_c)
    
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
            weights.append(n_total / (float(num_classes) * n_c))
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
