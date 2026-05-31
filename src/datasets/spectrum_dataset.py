"""PyTorch Dataset and DataLoader factories for SpectraSense."""

from typing import Optional, Callable, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def _ensure_psd_2d(psds: np.ndarray) -> np.ndarray:
    """Normalize PSD arrays to shape (N, 192)."""
    psds = np.asarray(psds, dtype=np.float32)
    if psds.ndim == 3 and psds.shape[-1] == 1:
        psds = psds[..., 0]
    elif psds.ndim > 2:
        psds = psds.reshape(psds.shape[0], -1)
    if psds.ndim != 2 or psds.shape[1] != 192:
        raise ValueError(f"Expected PSD shape (N, 192), got {psds.shape}")
    return psds


class SpectrumDataset(Dataset):
    """Dataset for PSD spectra.
    
    Expects arrays:
        psds: (N, 192) float32
        pu_labels: (N,) int64
        mod_labels: (N,) int64
        snrs: (N,) float32
    """

    def __init__(
        self,
        psds: np.ndarray,
        pu_labels: np.ndarray,
        mod_labels: np.ndarray,
        snrs: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        self.psds = _ensure_psd_2d(psds)
        self.pu_labels = pu_labels.astype(np.int64)
        self.mod_labels = mod_labels.astype(np.int64)
        self.snrs = snrs.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.psds)

    def __getitem__(self, idx: int):
        x = self.psds[idx]
        y_pu = self.pu_labels[idx]
        y_mod = self.mod_labels[idx]
        y_snr = self.snrs[idx]

        if self.transform is not None:
            x = self.transform(x)

        # Return tensors
        return (
            torch.from_numpy(x).float(),
            torch.tensor(y_pu, dtype=torch.long),
            torch.tensor(y_mod, dtype=torch.long),
            torch.tensor(y_snr, dtype=torch.float32),
        )


def create_dataloaders(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    use_snr_balancing: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from processed split dictionaries."""
    train_ds = SpectrumDataset(
        train_data["psds"],
        train_data["pu_labels"],
        train_data["mod_labels"],
        train_data["snrs"],
    )
    val_ds = SpectrumDataset(
        val_data["psds"],
        val_data["pu_labels"],
        val_data["mod_labels"],
        val_data["snrs"],
    )
    test_ds = SpectrumDataset(
        test_data["psds"],
        test_data["pu_labels"],
        test_data["mod_labels"],
        test_data["snrs"],
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader_kwargs = {
        "dataset": train_ds,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "generator": g,
    }

    if use_snr_balancing:
        snr_bins = np.round(train_ds.snrs).astype(np.int64)
        unique_bins, counts = np.unique(snr_bins, return_counts=True)
        inv_freq = {b: 1.0 / c for b, c in zip(unique_bins, counts)}
        sample_weights = np.array([inv_freq[b] for b in snr_bins], dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
            generator=g,
        )
        train_loader = DataLoader(sampler=sampler, shuffle=False, **train_loader_kwargs)
    else:
        train_loader = DataLoader(shuffle=True, **train_loader_kwargs)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
