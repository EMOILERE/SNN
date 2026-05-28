"""CIFAR-10 loaders (vision sensing)."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader]:
    root_path = Path(root) / "cifar10"
    train_tfm = (
        transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        if augment
        else transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
        )
    )
    test_tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    train = datasets.CIFAR10(str(root_path), train=True, transform=train_tfm, download=True)
    test = datasets.CIFAR10(str(root_path), train=False, transform=test_tfm, download=True)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, drop_last=True, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
