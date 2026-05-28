"""Dataset wrappers exposing a common `get_loaders(root, batch_size, ...)` API."""

from .cifar10 import get_cifar10_loaders
from .gsc_v2 import get_gsc_loaders
from .uci_har import get_ucihar_loaders
from .ut_har import get_uthar_loaders

__all__ = [
    "get_cifar10_loaders",
    "get_gsc_loaders",
    "get_ucihar_loaders",
    "get_uthar_loaders",
]
