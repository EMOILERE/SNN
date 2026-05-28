"""DGN: Dynamic Gated Neuron — LSTM-style spiking unit with dynamic conductance.

A plug-and-play replacement for spikingjelly.activation_based.neuron.LIFNode.
Reference: Bai, Wang, Yu, "A Brain-Inspired Gating Mechanism Unlocks Robust
Computation in Spiking Neural Networks", ICLR 2026 (closed source).

DGN replaces LIF's fixed leak factor (1 − 1/τ) with a *dynamic forget gate*
whose value is computed from the recent input itself, mirroring how
biological neurons' membrane conductance is modulated by Ca²⁺-dependent
plasticity (Abbott & LeMasson 1993; Gütig 2014).

Discrete dynamics (paper Eq. 5–8):

    D^t   = e^{−Δt/τ_s} · D^{t−1} + z^t                  # synaptic current
    ρ^t   = φ(1 − g_l·Δt − Δt · Σ_i C_i D^t_i)            # dynamic forget gate
    V^t   = ρ^t · V^{t−1} + Δt · Σ_i W_i D^t_i − ϑ z^{t−1}  # soft reset
    z^t   = H(V^t − ϑ)                                    # fire

where C_i are *learnable per-input conductance coefficients* — this is what
makes ρ adapt to input activity.

**Plug-and-play simplification** (analogous to DLIF):

The original DGN sits before the synaptic weight W, so it sees raw input
spikes z_i. To make it a drop-in for `LIFNode` after `Conv+BN`, we treat the
post-conv current `x[t]` as both the input activity for the synaptic
filter D and as the proxy for `Σ W_i z_i`. Concretely:

    D^t       = decay · D^{t−1} + x[t]
    ρ^t[c]    = sigmoid(b_g − α · C[c] · D^t[c])          # channel-wise gate
    V^t       = ρ^t · V^{t−1} + D^t − ϑ · z^{t−1}
    z^t       = surrogate(V^t − ϑ)

where `b_g` (≈ `1 − g_l·Δt`) is a learnable bias and `C[c]` is per-channel
(lazy-initialised on the first forward call, like DLIF's K).

This preserves DGN's defining property: ρ depends on the current input, so
the neuron has an LSTM-like forget gate without any extra MLP. Like LIF and
DLIF, we use a smooth ATan surrogate for the spike step.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from spikingjelly.activation_based import base, surrogate


class DGN(base.MemoryModule):
    """Multi-step Dynamic Gated Neuron.

    Args:
        v_threshold: Firing threshold ϑ.
        v_reset:     Reset potential after a spike (soft reset uses `−ϑ z`, so
                     this is informational only — kept for parity with LIFNode).
        tau_s:       Synaptic time constant (default 2.0 — same as LIF τ).
        g_l_init:    Initial leak conductance, mapped to the sigmoid bias
                     `b_g = 1 − g_l·dt`. Larger → ρ closer to 1 (more memory).
        C_init:      Initial value of the per-channel conductance coefficient.
        dt:          Integration step length (default 1.0).
        detach_reset: Whether to detach the soft-reset term from autograd.
        surrogate_function: Differentiable spike surrogate (default ATan(2)).
        rho_clip:     Hard clip on `sigmoid` argument to prevent saturation.

    I/O shape: (T, B, C, ...) — identical to `LIFNode(step_mode='m')`.

    The per-channel `C` parameter is allocated lazily on the first forward
    call (based on the input's channel dim, index 2). This lets
    `deepcopy(dgn)` produce four layer-specific DGN instances from a single
    prototype, exactly mirroring DLIF's behaviour.
    """

    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        tau_s: float = 2.0,
        g_l_init: float = 0.5,
        C_init: float = 0.5,
        dt: float = 1.0,
        detach_reset: bool = True,
        surrogate_function: Optional[nn.Module] = None,
        rho_clip: float = 10.0,
    ) -> None:
        super().__init__()
        assert tau_s > 0
        self.v_threshold = float(v_threshold)
        self.v_reset = float(v_reset)
        self.tau_s = float(tau_s)
        self.dt = float(dt)
        self.detach_reset = bool(detach_reset)
        self.rho_clip = float(rho_clip)
        self.surrogate_function = surrogate_function or surrogate.ATan(alpha=2.0)

        # Pre-compute synaptic decay factor exp(-dt/tau_s). Kept as a buffer so
        # it moves with .to(device).
        decay = math.exp(-self.dt / self.tau_s)
        self.register_buffer("synaptic_decay", torch.tensor(decay))

        # Learnable forget-gate bias b_g = 1 − g_l·dt. We parameterize b_g
        # directly so any sign is allowed (more flexible than constraining g_l > 0).
        self.gate_bias = nn.Parameter(torch.tensor(1.0 - g_l_init * self.dt))

        # Per-channel conductance coefficient C[c] — allocated lazily.
        self.C_init = float(C_init)
        self._initialized = False
        self.num_channels: Optional[int] = None
        self.C: Optional[nn.Parameter] = None

        # Memories
        self.register_memory("D", None)        # synaptic current
        self.register_memory("V", None)        # membrane potential
        self.register_memory("z_prev", None)   # previous spike (for soft reset)

    def _lazy_init(self, num_channels: int, device, dtype) -> None:
        self.num_channels = int(num_channels)
        C = torch.full((num_channels,), self.C_init, device=device, dtype=dtype)
        self.C = nn.Parameter(C)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (3, 4, 5):
            raise ValueError(f"DGN expects 3D/4D/5D (T, B, ...); got shape {tuple(x.shape)}")
        T, B = x.shape[0], x.shape[1]
        # Channel dim is index 2 for (T, B, C, ...) — same as DLIF.
        C_dim = x.shape[2]

        if not self._initialized:
            self._lazy_init(C_dim, x.device, x.dtype)
        elif self.num_channels != C_dim:
            raise RuntimeError(
                f"DGN was lazy-initialised for {self.num_channels} channels but "
                f"received {C_dim}."
            )

        # Broadcast C across all non-channel dims: shape (1, C, 1, 1, ...).
        C_view = self.C.view(1, C_dim, *([1] * (x.dim() - 3)))
        decay = self.synaptic_decay

        # Initialize state if needed.
        D = self.D if self.D is not None else torch.zeros_like(x[0])
        V = self.V if self.V is not None else torch.zeros_like(x[0])
        z_prev = self.z_prev if self.z_prev is not None else torch.zeros_like(x[0])

        spikes = []
        for t in range(T):
            # ── 1. Synaptic current update ──
            #    D^t = exp(−dt/τ_s) · D^{t−1} + x[t]
            D = decay * D + x[t]

            # ── 2. Dynamic forget gate (LSTM-style) ──
            #    ρ^t = sigmoid(b_g − dt · C[c] · D^t)
            #    Larger input → smaller ρ → faster forgetting (DGN core property).
            rho_arg = (self.gate_bias - self.dt * C_view * D).clamp(
                min=-self.rho_clip, max=self.rho_clip
            )
            rho = torch.sigmoid(rho_arg)

            # ── 3. Membrane update with soft reset (paper Eq. 7) ──
            #    V^t = ρ · V^{t−1} + dt · D^t − ϑ · z^{t−1}
            if self.detach_reset:
                reset_term = self.v_threshold * z_prev.detach()
            else:
                reset_term = self.v_threshold * z_prev
            V = rho * V + self.dt * D - reset_term

            # ── 4. Fire via surrogate gradient ──
            spike = self.surrogate_function(V - self.v_threshold)
            spikes.append(spike)
            z_prev = spike

        # Persist state for spikingjelly's reset_net mechanism.
        self.D = D
        self.V = V
        self.z_prev = z_prev

        return torch.stack(spikes, dim=0)

    def extra_repr(self) -> str:
        return (
            f"v_threshold={self.v_threshold}, tau_s={self.tau_s}, "
            f"dt={self.dt}, num_channels={self.num_channels}"
        )


__all__ = ["DGN"]
