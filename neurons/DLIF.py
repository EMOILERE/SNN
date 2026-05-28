"""DLIF: Dendritic Leaky Integrate-and-Fire neuron.

A plug-and-play replacement for spikingjelly.activation_based.neuron.MultiStepLIFNode.
Reference: Ma et al., "Beyond Linear Processing: Dendritic Bilinear Integration
in Spiking Neural Networks", ICLR 2026.

DLIF augments the LIF input current with a bilinear interaction term:

    I[t] = w^T s[t]            (linear, identical to LIF)
         + s[t]^T K s[t]       (bilinear, new in DLIF)

For plug-and-play replacement of a MultiStepLIFNode applied AFTER a conv/linear
layer, the layer input `x` already represents the post-synaptic current `w^T s`.
We therefore treat `x` (per spatial location for conv inputs, or as-is for linear
inputs) as both the linear current and the proxy for pre-synaptic activity, and
add a learned bilinear term computed over the channel/feature dimension.

The bilinear coefficient tensor K has shape (C, C, C), where K[c, i, j] is the
coefficient of x_i * x_j contributing to channel c's extra current. Following the
paper, K is constrained to be symmetric in its last two dims with zero diagonal,
and 90% of entries are masked out by a fixed random binary mask (only the
unmasked entries are trainable).

The neuron is shape-agnostic at construction: K is allocated lazily on the first
forward call using the input's channel/feature dimension. This lets
`deepcopy(dlif)` work as a drop-in for `deepcopy(lif)` in the LeNet skeleton.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from spikingjelly.activation_based import base, surrogate


class DLIF(base.MemoryModule):
    """Multi-step Dendritic LIF neuron.

    Args:
        tau: membrane time constant (>= 1). Same meaning as MultiStepLIFNode.
        v_threshold: firing threshold.
        v_reset: reset potential after a spike (hard reset).
        sparsity: fraction of K entries forced to zero (paper uses 0.9).
        surrogate_function: differentiable spike surrogate. Defaults to ATan(2.0).
        detach_reset: detach the reset gate from the autograd graph.
        K_init_scale: std of the Gaussian used to initialise K's free entries.
        bilinear_scale: optional scalar applied to the bilinear current. Useful
            to keep the extra current at a similar magnitude as the linear part
            during early training, especially when C is large.

    Input:  (T, B, C, H, W) or (T, B, D).
    Output: spike tensor with the same shape as the input.
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        sparsity: float = 0.9,
        surrogate_function: Optional[nn.Module] = None,
        detach_reset: bool = True,
        K_init_scale: float = 0.01,
        bilinear_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        assert tau >= 1.0, "tau must be >= 1"
        assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)"
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.v_reset = float(v_reset)
        self.sparsity = float(sparsity)
        self.detach_reset = bool(detach_reset)
        self.K_init_scale = float(K_init_scale)
        self.bilinear_scale = bilinear_scale
        self.surrogate_function = surrogate_function or surrogate.ATan(alpha=2.0)

        self._initialized = False
        self.num_channels: Optional[int] = None
        self.K: Optional[nn.Parameter] = None
        # K_mask is a buffer (not trained); the effective bilinear weights are K * K_mask.
        self.register_buffer("K_mask", torch.empty(0), persistent=True)
        # Per-sequence membrane potential. Registered as a "memory" so spikingjelly's
        # functional.reset_net() will call reset() on us and zero it back out.
        self.register_memory("v", None)

    def _lazy_init(self, num_channels: int, device: torch.device, dtype: torch.dtype) -> None:
        """Allocate K and its sparse mask the first time we see input."""
        self.num_channels = int(num_channels)
        c = self.num_channels

        # Symmetric K with zero diagonal in the last two dims.
        U = torch.randn(c, c, c, device=device, dtype=dtype) * self.K_init_scale
        # Zero the diagonal entries K[c, i, i].
        diag_idx = torch.arange(c, device=device)
        U[:, diag_idx, diag_idx] = 0.0
        # Symmetrize: K[c, i, j] = K[c, j, i].
        K_init = 0.5 * (U + U.transpose(-1, -2))
        self.K = nn.Parameter(K_init)

        # Sparse, symmetric, zero-diagonal binary mask. Generated once, frozen.
        rand = torch.rand(c, c, c, device=device)
        mask = (rand > self.sparsity).to(dtype)
        mask = torch.maximum(mask, mask.transpose(-1, -2))  # symmetrize via OR
        mask[:, diag_idx, diag_idx] = 0.0
        self.K_mask = mask  # registered buffer; replaces the empty placeholder

        # Default bilinear scaling: keep extra current at O(linear current) magnitude.
        # Each output channel sees roughly c*(c-1)*(1-sparsity) nonzero K entries;
        # if x_i, x_j are O(1), the bilinear sum is O(c^2 * (1-sparsity) * K_init_scale).
        if self.bilinear_scale is None:
            n_active = max(int(c * (c - 1) * (1.0 - self.sparsity)), 1)
            self.bilinear_scale = 1.0 / math.sqrt(n_active)

        self._initialized = True

    def extra_repr(self) -> str:
        return (
            f"tau={self.tau}, v_threshold={self.v_threshold}, "
            f"v_reset={self.v_reset}, sparsity={self.sparsity}, "
            f"num_channels={self.num_channels}"
        )

    def _bilinear(self, xt: torch.Tensor) -> torch.Tensor:
        """Compute the per-channel bilinear current at one timestep.

        For input xt of shape (B, C, H, W):
            out[b, c, h, w] = sum_{i,j} K_eff[c, i, j] * xt[b, i, h, w] * xt[b, j, h, w]
        For input xt of shape (B, C, L):
            out[b, c, l]    = sum_{i,j} K_eff[c, i, j] * xt[b, i, l] * xt[b, j, l]
        For input xt of shape (B, D):
            out[b, c]       = sum_{i,j} K_eff[c, i, j] * xt[b, i] * xt[b, j]
        """
        K_eff = self.K * self.K_mask
        if xt.dim() == 4:
            return torch.einsum("cij,bihw,bjhw->bchw", K_eff, xt, xt)
        if xt.dim() == 3:
            return torch.einsum("cij,bil,bjl->bcl", K_eff, xt, xt)
        if xt.dim() == 2:
            return torch.einsum("cij,bi,bj->bc", K_eff, xt, xt)
        raise ValueError(f"DLIF unsupported per-step input dim {xt.dim()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (3, 4, 5):
            raise ValueError(f"DLIF expects 3D, 4D or 5D input (T,B,...); got shape {tuple(x.shape)}")
        T = x.shape[0]
        # Channel/feature dimension is index 2 for both (T,B,C,H,W) and (T,B,D).
        C = x.shape[2]

        if not self._initialized:
            self._lazy_init(C, x.device, x.dtype)
        elif self.num_channels != C:
            raise RuntimeError(
                f"DLIF was initialised for {self.num_channels} channels but received {C}."
            )

        # Reset state at the start of each sequence; spikingjelly's reset_net() sets self.v=None.
        v = self.v if self.v is not None else torch.zeros_like(x[0])

        spikes = []
        decay = 1.0 - 1.0 / self.tau  # leak factor
        for t in range(T):
            xt = x[t]
            current = xt + self.bilinear_scale * self._bilinear(xt)
            # Pre-spike membrane potential.
            u = decay * v + current / self.tau
            spike = self.surrogate_function(u - self.v_threshold)
            # Hard reset: V := V_reset on spike, else keep u.
            if self.detach_reset:
                v = u * (1.0 - spike.detach()) + self.v_reset * spike.detach()
            else:
                v = u * (1.0 - spike) + self.v_reset * spike
            spikes.append(spike)

        self.v = v
        return torch.stack(spikes, dim=0)


__all__ = ["DLIF"]
