"""PyTorch Dataset and DataLoader factories for SpectraSense."""

import math
from typing import Optional, Callable, Dict, Tuple, Iterator, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


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


class SNRStratifiedBatchSampler(Sampler[List[int]]):
    """Yield batches that mix low/high SNR samples within every batch."""

    def __init__(self, snrs: np.ndarray, batch_size: int, generator: torch.Generator):
        self.batch_size = int(batch_size)
        self.generator = generator
        self.snr_bins = np.round(np.asarray(snrs, dtype=np.float32)).astype(np.int64)
        self.unique_bins = sorted(int(b) for b in np.unique(self.snr_bins))
        self.indices_by_bin = {
            snr_bin: np.where(self.snr_bins == snr_bin)[0].tolist()
            for snr_bin in self.unique_bins
        }
        self.num_batches = math.ceil(len(self.snr_bins) / max(1, self.batch_size))
        self.per_bin_quota = max(1, self.batch_size // max(1, len(self.unique_bins)))
        self.remainder = self.batch_size - (self.per_bin_quota * len(self.unique_bins))

    def __len__(self) -> int:
        return self.num_batches

    def _reshuffle_bin(self, indices: List[int]) -> List[int]:
        if not indices:
            return []
        order = torch.randperm(len(indices), generator=self.generator).tolist()
        return [indices[i] for i in order]

    def _draw(self, pools: Dict[int, List[int]], positions: Dict[int, int], snr_bin: int) -> int:
        pool = pools[snr_bin]
        if not pool:
            raise RuntimeError(f"No samples available for SNR bin {snr_bin}")
        position = positions[snr_bin]
        if position >= len(pool):
            pool[:] = self._reshuffle_bin(pool)
            positions[snr_bin] = 0
            position = 0
        index = pool[position]
        positions[snr_bin] = position + 1
        return index

    def __iter__(self) -> Iterator[List[int]]:
        pools = {snr_bin: self._reshuffle_bin(indices) for snr_bin, indices in self.indices_by_bin.items()}
        positions = {snr_bin: 0 for snr_bin in self.unique_bins}

        for _ in range(self.num_batches):
            batch_indices: List[int] = []

            # Phase 2 Decision 2: replace the old inverse-frequency sampler (previously one sampled index at a time)
            # with per-batch SNR stratification so each batch always contains low- and high-SNR examples.
            for snr_bin in self.unique_bins:
                for _ in range(self.per_bin_quota):
                    batch_indices.append(self._draw(pools, positions, snr_bin))

            if self.remainder > 0:
                for i in range(self.remainder):
                    snr_bin = self.unique_bins[i % len(self.unique_bins)]
                    batch_indices.append(self._draw(pools, positions, snr_bin))

            yield batch_indices


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

    if use_snr_balancing:
        batch_sampler = SNRStratifiedBatchSampler(train_ds.snrs, batch_size=batch_size, generator=g)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            generator=g,
        )

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
