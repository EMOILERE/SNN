"""SpikingLeNet (3 conv + 2 fc) following the reference in task-requirements.md.

The neuron is passed in as a *prototype* and deep-copied at each layer, so the
same skeleton works for spikingjelly's MultiStepLIFNode, our DLIF, and any other
multi-step neuron sharing the (T, B, C, H, W) / (T, B, D) interface.
"""

from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional


def multi_time_forward(x: torch.Tensor, modules) -> torch.Tensor:
    """Apply non-temporal module(s) to a (T, B, ...) tensor by flattening T into B."""
    T, B = x.shape[0], x.shape[1]
    rest = x.shape[2:]
    y = x.reshape(T * B, *rest)
    if isinstance(modules, (list, tuple, nn.Sequential)):
        for m in modules:
            y = m(y)
    else:
        y = modules(y)
    return y.reshape(T, B, *y.shape[1:])


class SpikingLeNet(nn.Module):
    """3-conv + 2-fc Spiking LeNet.

    `config` keys:
        num_classes:    int
        time_step:      int (T)
        input_channels: C
        input_height:   H
        input_width:    W
        hidden_dim:     int
        neuron:         prototype neuron module (will be deepcopied per layer)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_classes = config["num_classes"]
        self.T = config["time_step"]

        C, H, W = config["input_channels"], config["input_height"], config["input_width"]
        lif = config["neuron"]

        self.conv1 = nn.Conv2d(C, 32, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = deepcopy(lif)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        H, W = (H - 4) // 2, (W - 4) // 2

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = deepcopy(lif)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        H, W = (H - 4) // 2, (W - 4) // 2

        self.conv3 = nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(96)
        self.lif3 = deepcopy(lif)
        H, W = H - 4, W - 4

        if H <= 0 or W <= 0:
            raise ValueError(
                f"Input {config['input_height']}x{config['input_width']} too small for "
                "the 3-conv5 LeNet; got spatial size {H}x{W} after layer 3."
            )

        self.ln1 = nn.Linear(96 * H * W, config["hidden_dim"])
        self.lif4 = deepcopy(lif)
        self.head = nn.Linear(config["hidden_dim"], self.num_classes)

        self._flat_dim = 96 * H * W

    def _expand_T(self, x: torch.Tensor) -> torch.Tensor:
        """Make sure input has a leading time dim of size T."""
        if x.dim() == 4:  # (B, C, H, W) static image -> repeat along T
            x = x.unsqueeze(0).expand(self.T, *x.shape).contiguous()
        elif x.dim() == 5:
            # Either (T, B, C, H, W) already or (B, T, C, H, W).
            if x.shape[0] != self.T and x.shape[1] == self.T:
                x = x.permute(1, 0, 2, 3, 4).contiguous()
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        x = self._expand_T(x)

        x = multi_time_forward(x, [self.conv1, self.bn1])
        x = self.lif1(x)
        x = multi_time_forward(x, self.pool1)

        x = multi_time_forward(x, [self.conv2, self.bn2])
        x = self.lif2(x)
        x = multi_time_forward(x, self.pool2)

        x = multi_time_forward(x, [self.conv3, self.bn3])
        x = self.lif3(x)

        x = x.flatten(2)  # (T, B, C, H, W) -> (T, B, CHW)

        x = multi_time_forward(x, self.ln1)
        x = self.lif4(x)

        x = self.head(x.mean(0))  # (T, B, D) -> (B, num_classes) via mean over time
        return x


class SpikingLeNet1D(nn.Module):
    """1D variant for motion / acoustic feature sequences.

    Input: (B, C, L)  or (T, B, C, L). Treated as 1D conv. The lif layers reuse
    the same DLIF/LIF prototype since both shapes are bilinear on the channel dim.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_classes = config["num_classes"]
        self.T = config["time_step"]

        C, L = config["input_channels"], config["input_length"]
        lif = config["neuron"]

        self.conv1 = nn.Conv1d(C, 32, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(32)
        self.lif1 = deepcopy(lif)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        L = (L - 4) // 2

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(64)
        self.lif2 = deepcopy(lif)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        L = (L - 4) // 2

        self.conv3 = nn.Conv1d(64, 96, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(96)
        self.lif3 = deepcopy(lif)
        L = L - 4

        if L <= 0:
            raise ValueError(f"Input length too small for 1D LeNet (got L={L}).")

        self.ln1 = nn.Linear(96 * L, config["hidden_dim"])
        self.lif4 = deepcopy(lif)
        self.head = nn.Linear(config["hidden_dim"], self.num_classes)

    def _expand_T(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (B, C, L)
            x = x.unsqueeze(0).expand(self.T, *x.shape).contiguous()
        elif x.dim() == 4:
            if x.shape[0] != self.T and x.shape[1] == self.T:
                x = x.permute(1, 0, 2, 3).contiguous()
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        x = self._expand_T(x)

        x = multi_time_forward(x, [self.conv1, self.bn1])
        x = self.lif1(x)
        x = multi_time_forward(x, self.pool1)

        x = multi_time_forward(x, [self.conv2, self.bn2])
        x = self.lif2(x)
        x = multi_time_forward(x, self.pool2)

        x = multi_time_forward(x, [self.conv3, self.bn3])
        x = self.lif3(x)

        x = x.flatten(2)  # (T, B, C, L) -> (T, B, CL)
        x = multi_time_forward(x, self.ln1)
        x = self.lif4(x)
        x = self.head(x.mean(0))
        return x


__all__ = ["SpikingLeNet", "SpikingLeNet1D", "multi_time_forward"]
