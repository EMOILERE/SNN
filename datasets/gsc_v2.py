"""Google Speech Commands V2 (acoustic sensing).

Implements the standard 12-class split (10 keywords + "unknown" + "silence")
following Warden 2018. Each clip is 1 s at 16 kHz; we extract 64-channel
log-mel features over 100 frames giving (B, 64, 100) inputs for the 1D LeNet.

To keep things light we compute features on-the-fly with torchaudio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

CORE_WORDS = (
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
)
UNKNOWN_LABEL = "_unknown_"
SILENCE_LABEL = "_silence_"
CLASS_NAMES: tuple[str, ...] = (SILENCE_LABEL, UNKNOWN_LABEL) + CORE_WORDS
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _read_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


class GSCV2Dataset(Dataset):
    """Google Speech Commands V2 with on-the-fly log-mel features."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
        max_seconds: float = 1.0,
        unknown_ratio: float = 0.1,
        silence_ratio: float = 0.1,
        rng_seed: int = 0,
    ) -> None:
        assert split in ("train", "valid", "test")
        self.root = Path(root) / "speech_commands_v2"
        if not self.root.exists():
            raise FileNotFoundError(
                f"GSC v2 not found at {self.root}. Run scripts/download_datasets.py --only gsc."
            )
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)

        val_set = _read_list(self.root / "validation_list.txt")
        test_set = _read_list(self.root / "testing_list.txt")

        files: list[tuple[str, int]] = []
        for wav in self.root.rglob("*.wav"):
            rel = wav.relative_to(self.root).as_posix()
            label_word = wav.parent.name
            # background_noise_/* is used only for silence samples
            if label_word.startswith("_background_noise_"):
                continue
            if rel in val_set:
                target_split = "valid"
            elif rel in test_set:
                target_split = "test"
            else:
                target_split = "train"
            if target_split != split:
                continue
            if label_word in CORE_WORDS:
                label = LABEL_TO_IDX[label_word]
            else:
                label = LABEL_TO_IDX[UNKNOWN_LABEL]
            files.append((rel, label))

        # Re-balance: in the raw split, "unknown" massively outnumbers core words.
        rng = np.random.default_rng(rng_seed)
        core = [f for f in files if f[1] != LABEL_TO_IDX[UNKNOWN_LABEL]]
        unknown = [f for f in files if f[1] == LABEL_TO_IDX[UNKNOWN_LABEL]]
        n_keep = max(1, int(len(core) * unknown_ratio / max(len(CORE_WORDS), 1)))
        n_keep = max(n_keep, len(core) // max(len(CORE_WORDS), 1))
        unknown_idx = rng.choice(len(unknown), size=min(n_keep, len(unknown)), replace=False)
        files = core + [unknown[i] for i in unknown_idx]

        # Generate silence clips from background noise.
        bg_dir = self.root / "_background_noise_"
        self.bg_wavs = sorted(bg_dir.glob("*.wav")) if bg_dir.exists() else []
        n_silence = max(1, int(len(files) * silence_ratio))
        self.silence_indices = list(range(len(files), len(files) + n_silence))
        self.entries = files  # list of (rel_path, label)
        self.n_silence = n_silence

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
        self._rng = rng

    def __len__(self) -> int:
        return len(self.entries) + self.n_silence

    def _load_wav(self, rel: str) -> torch.Tensor:
        wav, sr = torchaudio.load(str(self.root / rel))
        wav = wav.mean(dim=0, keepdim=False)  # mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.numel() < self.max_samples:
            pad = self.max_samples - wav.numel()
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            wav = wav[: self.max_samples]
        return wav

    def _sample_silence(self) -> torch.Tensor:
        if not self.bg_wavs:
            return torch.zeros(self.max_samples)
        bg = self.bg_wavs[self._rng.integers(0, len(self.bg_wavs))]
        wav, sr = torchaudio.load(str(bg))
        wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.numel() < self.max_samples:
            return torch.nn.functional.pad(wav, (0, self.max_samples - wav.numel()))
        start = int(self._rng.integers(0, wav.numel() - self.max_samples))
        scale = float(self._rng.uniform(0.0, 0.5))
        return wav[start : start + self.max_samples] * scale

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if idx >= len(self.entries):
            wav = self._sample_silence()
            label = LABEL_TO_IDX[SILENCE_LABEL]
        else:
            rel, label = self.entries[idx]
            wav = self._load_wav(rel)
        mel = self.db(self.mel(wav))  # (n_mels, time)
        # Normalize per-clip to unit variance.
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel, label


def get_gsc_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = GSCV2Dataset(root, split="train")
    test_ds = GSCV2Dataset(root, split="test")
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, test_loader
