"""UCI HAR (motion sensing).

Each sample is 9 inertial channels x 128 timesteps, classified into 6 activities
(walking, walking_upstairs, walking_downstairs, sitting, standing, laying).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

UCIHAR_SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _load_signals(split_dir: Path, split: str) -> np.ndarray:
    """Load and stack the 9 inertial signals -> (N, 9, 128)."""
    inertial = split_dir / "Inertial Signals"
    arrays = []
    for sig in UCIHAR_SIGNALS:
        f = inertial / f"{sig}_{split}.txt"
        arrays.append(np.loadtxt(f, dtype=np.float32))
    return np.stack(arrays, axis=1)  # (N, 9, 128)


def _load_labels(split_dir: Path, split: str) -> np.ndarray:
    f = split_dir / f"y_{split}.txt"
    y = np.loadtxt(f, dtype=np.int64) - 1  # original labels are 1..6
    return y


def get_ucihar_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    normalize: bool = True,
) -> tuple[DataLoader, DataLoader]:
    base = Path(root) / "uci_har" / "UCI HAR Dataset"
    if not base.exists():
        raise FileNotFoundError(
            f"UCI HAR not found at {base}. Run scripts/download_datasets.py --only ucihar first."
        )
    X_train = _load_signals(base / "train", "train")
    y_train = _load_labels(base / "train", "train")
    X_test = _load_signals(base / "test", "test")
    y_test = _load_labels(base / "test", "test")

    if normalize:
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader
