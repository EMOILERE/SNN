"""UT-HAR (WiFi CSI human activity recognition).

The "UT-HAR" pre-processed split used in WiFi-sensing literature contains 7
activities (lie down, fall, walk, run, sit down, stand up, pick up) over
90 CSI subcarriers x 250 frames. The download script attempts to fetch the
pre-processed npz/csv bundle from a public mirror; otherwise we fall back to
expecting the user-provided files at `<root>/ut_har/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _find_array(d: Path, names: list[str]) -> Optional[Path]:
    for n in names:
        for p in d.rglob(n):
            return p
    return None


def _load_processed(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Try a few common UT-HAR layouts and return (X_train, y_train, X_test, y_test).

    Priority order — REAL data wins over the synthetic placeholder:
      1. xyanchen layout at <root>/ut_har/{data,label}/*.csv   (preferred)
      2. xyanchen with a UT_HAR wrapper folder
      3. Plain .npy files anywhere under <root>/ut_har/
      4. The raw ermongroup csv release
      5. Synthetic UT_HAR.npz fallback (only if nothing real is found)
    """
    base = root / "ut_har"

    # Layout 1: xyanchen preprocessed at the standard path.
    #   <root>/ut_har/{data,label}/{X_train,X_val,X_test,y_train,y_val,y_test}.csv
    # NOTE: these "*.csv" files are actually .npy binaries (numpy header), so we
    # use np.load instead of np.loadtxt.
    if (base / "data").exists() and (base / "label").exists():
        return _load_xyanchen_layout(base / "data", base / "label")

    # Layout 2: same layout but inside a UT_HAR wrapper folder.
    if (base / "UT_HAR" / "data").exists() and (base / "UT_HAR" / "label").exists():
        return _load_xyanchen_layout(base / "UT_HAR" / "data", base / "UT_HAR" / "label")

    # Layout 3: separate .npy files anywhere under <root>/ut_har/.
    Xtr = _find_array(base, ["X_train.npy"])
    Xte = _find_array(base, ["X_test.npy"])
    ytr = _find_array(base, ["y_train.npy"])
    yte = _find_array(base, ["y_test.npy"])
    if all(p is not None for p in (Xtr, Xte, ytr, yte)):
        return (
            _ensure_csi_shape(np.load(Xtr).astype(np.float32)),
            np.load(ytr).astype(np.int64),
            _ensure_csi_shape(np.load(Xte).astype(np.float32)),
            np.load(yte).astype(np.int64),
        )

    # Layout 4: raw ermongroup csvs.
    raw_dir = base / "Wifi_Activity_Recognition-master"
    if raw_dir.is_dir() and any(raw_dir.rglob("*.csv")):
        return _parse_wifi_activity_recognition(raw_dir)

    # Layout 5 (LAST RESORT): the synthetic npz placeholder.
    npz = _find_array(base, ["UT_HAR.npz", "ut_har.npz"])
    if npz is not None:
        print(f"  [warning] using synthetic UT-HAR placeholder at {npz}; "
              "results are NOT meaningful. Please replace with real data.")
        data = np.load(npz)
        return (
            data["X_train"].astype(np.float32),
            data["y_train"].astype(np.int64),
            data["X_test"].astype(np.float32),
            data["y_test"].astype(np.int64),
        )

    raise FileNotFoundError(
        f"No recognised UT-HAR layout under {base}. Place the dataset there manually "
        "(e.g. data/ut_har/{data,label}/X_*.csv from xyanchen/WiFi-CSI-Sensing-Benchmark)."
    )


def _ensure_csi_shape(arr: np.ndarray) -> np.ndarray:
    """Normalise CSI tensors to (N, C=90, L=250).

    The xyanchen preprocessed bundle stores samples as (N, 250, 90) — time
    along axis 1, subcarriers along axis 2. Our 1D LeNet expects channels
    first, so we transpose to (N, 90, 250)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D CSI tensor, got shape {arr.shape}")
    # Heuristic: 90 subcarriers, 250 frames. If the layout is (N, 250, 90),
    # swap the last two axes.
    if arr.shape[1] == 250 and arr.shape[2] == 90:
        arr = arr.transpose(0, 2, 1)
    elif arr.shape[1] == 90 and arr.shape[2] == 250:
        pass  # already correct
    else:
        raise ValueError(
            f"Unexpected UT-HAR sample shape {arr.shape}; expected (N, 90, 250) or (N, 250, 90)."
        )
    return arr


def _load_xyanchen_layout(
    data_dir: Path, label_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the xyanchen WiFi-CSI-Sensing-Benchmark preprocessed UT-HAR.

    Despite the `.csv` extension, the files in this release are actually
    NumPy `.npy` binaries (the magic-number prefix `\\x93NUMPY` makes it
    obvious). We therefore use `np.load`, not `np.loadtxt`.

    The X tensors are stored as (N, 250, 90); we transpose to (N, 90, 250)
    so they match SpikingLeNet1D's (B, C=90, L=250) input convention.
    """

    def _x(name: str) -> np.ndarray:
        arr = np.load(data_dir / f"X_{name}.csv", allow_pickle=False)
        return _ensure_csi_shape(arr.astype(np.float32))

    def _y(name: str) -> np.ndarray:
        arr = np.load(label_dir / f"y_{name}.csv", allow_pickle=False)
        return arr.astype(np.int64).reshape(-1)

    X_train = _x("train")
    y_train = _y("train")

    # If a val split is present, merge it into train (we only need train/test).
    val_x = data_dir / "X_val.csv"
    val_y = label_dir / "y_val.csv"
    if val_x.exists() and val_y.exists():
        X_train = np.concatenate([X_train, _x("val")], axis=0)
        y_train = np.concatenate([y_train, _y("val")], axis=0)

    X_test = _x("test")
    y_test = _y("test")
    return X_train, y_train, X_test, y_test


def _parse_wifi_activity_recognition(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best-effort parser for the raw ermongroup release. Falls back to a small
    held-out split if no canonical split is provided."""
    # The original code provides preprocessed CSVs per activity. We do a naive
    # 80/20 split and 250-frame, 90-subcarrier resampling.
    import pandas as pd

    activities = ["lie_down", "fall", "walk", "run", "sit_down", "stand_up", "pick_up"]
    label_map = {a: i for i, a in enumerate(activities)}
    samples, labels = [], []
    win = 250
    for csv in root.rglob("*.csv"):
        name = csv.stem.lower()
        activity = next((a for a in activities if a in name), None)
        if activity is None:
            continue
        try:
            arr = pd.read_csv(csv, header=None).values.astype(np.float32)
        except Exception:
            continue
        # arr shape: (frames, subcarriers); we keep first 90 subcarriers.
        if arr.shape[1] < 90:
            continue
        arr = arr[:, :90]
        # Slide non-overlapping windows of 250 frames.
        n = arr.shape[0] // win
        for k in range(n):
            samples.append(arr[k * win : (k + 1) * win].T)  # (90, 250)
            labels.append(label_map[activity])
    if not samples:
        raise RuntimeError("No CSI samples parsed from Wifi_Activity_Recognition-master.")
    X = np.stack(samples)
    y = np.array(labels, dtype=np.int64)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    return X[perm[:cut]], y[perm[:cut]], X[perm[cut:]], y[perm[cut:]]


def get_uthar_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 2,
    normalize: bool = True,
) -> tuple[DataLoader, DataLoader]:
    X_train, y_train, X_test, y_test = _load_processed(Path(root))
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
