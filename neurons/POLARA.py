"""POLARA: Polarization-aware spiking neuron with stable gradient propagation.

A plug-and-play replacement for spikingjelly.activation_based.neuron.LIFNode.
Reference: Lai & Cao, "Stabilizing Spiking Neurons Through Biologically
Inspired Polarization", AAAI 2026 (closed source).

The neuron decomposes membrane behavior into the three biological polarization
phases through five forward components executed each timestep:

    (1) Membrane decay        — u_d   = (1 − leak) · Σ_i d(i) · u_{t−i}
    (2) Input integration     — u_s   = u_d + x_t
    (3) Stochastic amplification — u_a = u_s · (1 + B·e^{βδ})  if δ ≤ 0
    (4) Spike activation      — s_t   = σ(u_a · O(Δ)) · 1[Δ ≥ 0]
    (5) Refractory inhibition — u_t   = u_a · (1 + Σ h(i))
        Adaptive threshold    — ϑ_{t+1} = ϑ_t · (1 + Σ Θ(i))

with O(Δ) = min(Δ(1 + Δ·e^{−γΔ}), o_max).

The bounded derivative of O is what gives POLARA its no-surrogate-gradient
stability guarantee: O'(Δ) = 1 + Δ e^{−γΔ}(2 − γΔ) is bounded by a constant
M that does not depend on T, so ∏_t |∂s_t/∂u_{t−1}| stays in [A^T, B^T] with
A·B ≈ 1 (Corollary 1 of the paper).

Like DLIF, POLARA can be dropped into the same SpikingLeNet skeleton: it
inherits SpikingJelly's MemoryModule so `functional.reset_net()` clears state,
and it supports 3D / 4D / 5D inputs (T, B, ...) — i.e. the same interface as
`LIFNode(step_mode='m')`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from spikingjelly.activation_based import base


class POLARA(base.MemoryModule):
    """Multi-step polarization-aware spiking neuron.

    Args:
        window:           Size of the temporal kernel window I (paper uses 6).
        v_threshold_init: Initial firing threshold ϑ_0.
        amp_B, amp_beta:  Stochastic amplification a(δ) = B·e^{β δ}.
        gamma, o_max:     Spike activation O(Δ) = min(Δ(1+Δ e^{−γΔ}), o_max).
        leak_init:        Initial scalar leak (before sigmoid). 0 → ρ-residual 0.5.
        refrac_amp:       Total amplitude of the refractory inhibition kernel
                          h(i). Negative → multiplicative suppression after
                          firing (1 + Σh < 1). Positive → amplification.
        n_refrac_gauss:   Number of Gaussians composing h(i).
        theta_E, theta_eta: Adaptive threshold kernel Θ(i) = E·e^{−η i}.
        learnable_kernels: If True, kernel parameters are nn.Parameter; else
                           registered as buffers.
        use_grad_mask:    Enable the CIFAR-style rectangular gradient mask
                          (POLARA §4 *Gradient Stabilization*). Off by default
                          since the paper says O alone is enough on shallow nets.
        grad_mask_alpha:  α of the rectangular gradient mask.
        v_threshold_clip: Hard cap on the adaptive threshold to prevent runaway.

    I/O shape: (T, B, C, ...) — identical to `LIFNode(step_mode='m')`.
    """

    def __init__(
        self,
        window: int = 6,
        v_threshold_init: float = 1.0,
        amp_B: float = 0.3,
        amp_beta: float = 2.0,
        gamma: float = 1.0,
        o_max: float = 5.0,
        leak_init: float = 0.0,
        refrac_amp: float = -0.3,
        n_refrac_gauss: int = 3,
        theta_E: float = 0.02,
        theta_eta: float = 1.0,
        learnable_kernels: bool = True,
        use_grad_mask: bool = False,
        grad_mask_alpha: float = 0.3,
        v_threshold_clip: float = 5.0,
    ) -> None:
        super().__init__()
        assert window >= 1
        self.window = int(window)
        self.v_threshold_init = float(v_threshold_init)
        self.o_max = float(o_max)
        self.v_threshold_clip = float(v_threshold_clip)
        self.use_grad_mask = bool(use_grad_mask)
        self.grad_mask_alpha = float(grad_mask_alpha)

        def _maybe_param(t: torch.Tensor) -> torch.Tensor:
            return nn.Parameter(t) if learnable_kernels else t

        # ── Decay kernel d(i): window-length softmax distribution × scalar leak ──
        self.d_logits = _maybe_param(torch.zeros(self.window))
        self.leak_logit = _maybe_param(torch.tensor(float(leak_init)))

        # ── Refractory kernel h(i) = Σ_j D_j · exp(−(i−μ_j)² / 2σ_j²) ──────────
        amps = torch.full((n_refrac_gauss,), float(refrac_amp) / n_refrac_gauss)
        means = torch.linspace(1.0, float(self.window), n_refrac_gauss)
        log_sigmas = torch.zeros(n_refrac_gauss)
        if learnable_kernels:
            self.h_amps = nn.Parameter(amps)
            self.h_means = nn.Parameter(means)
            self.h_log_sigmas = nn.Parameter(log_sigmas)
        else:
            self.register_buffer("h_amps", amps)
            self.register_buffer("h_means", means)
            self.register_buffer("h_log_sigmas", log_sigmas)

        # ── Threshold adaptation kernel Θ(i) = E · exp(−η i) ──────────────────
        self.theta_E = _maybe_param(torch.tensor(float(theta_E)))
        self.theta_eta = _maybe_param(torch.tensor(float(theta_eta)))

        # ── Stochastic amplification scalars B, β ─────────────────────────────
        self.amp_B = _maybe_param(torch.tensor(float(amp_B)))
        self.amp_beta = _maybe_param(torch.tensor(float(amp_beta)))

        # ── Spike activation scalar γ ─────────────────────────────────────────
        self.gamma = _maybe_param(torch.tensor(float(gamma)))

        # ── State variables (registered as SpikingJelly memories) ─────────────
        # u_history: (window, B, C, ...) — past membrane potentials
        # v_threshold: (B, C, ...)       — per-element adaptive threshold
        self.register_memory("u_history", None)
        self.register_memory("v_threshold", None)

    # ── Kernel computations ────────────────────────────────────────────────────
    def _decay_weights(self) -> torch.Tensor:
        """(window,) — d(i) softmax-normalized so Σ d(i) = 1."""
        return torch.softmax(self.d_logits, dim=0)

    def _leak(self) -> torch.Tensor:
        """Scalar leak ∈ (0, 1) — the LIF-equivalent 1/τ."""
        return torch.sigmoid(self.leak_logit)

    def _refrac_kernel(self) -> torch.Tensor:
        """(window,) — h(i) = Σ_j amp_j · exp(−(i−μ_j)² / 2σ_j²)."""
        i = torch.arange(1, self.window + 1, device=self.h_amps.device,
                         dtype=self.h_amps.dtype)
        sigmas = torch.exp(self.h_log_sigmas).clamp(min=1e-3)
        # broadcast: (n_gauss, window)
        gauss = self.h_amps.unsqueeze(1) * torch.exp(
            -((i.unsqueeze(0) - self.h_means.unsqueeze(1)) ** 2)
            / (2.0 * sigmas.unsqueeze(1) ** 2)
        )
        return gauss.sum(dim=0)  # (window,)

    def _theta_kernel(self) -> torch.Tensor:
        """(window,) — Θ(i) = E · exp(−|η| · i)."""
        i = torch.arange(1, self.window + 1, device=self.theta_E.device,
                         dtype=self.theta_E.dtype)
        return self.theta_E * torch.exp(-self.theta_eta.abs() * i)

    # ── State initialization ───────────────────────────────────────────────────
    def _init_state(self, shape: tuple, device, dtype) -> None:
        self.u_history = torch.zeros(self.window, *shape, device=device, dtype=dtype)
        self.v_threshold = torch.full(shape, self.v_threshold_init,
                                      device=device, dtype=dtype)

    # ── Forward pass ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(f"POLARA expects (T, B, ...); got shape {tuple(x.shape)}")
        T = x.shape[0]
        rest = x.shape[1:]

        if (self.u_history is None) or (self.u_history.shape[1:] != rest):
            self._init_state(rest, x.device, x.dtype)

        # Precompute shape-free kernels once per sequence.
        d_weights = self._decay_weights()                       # (window,)
        leak = self._leak()                                     # scalar
        h_sum = self._refrac_kernel().sum()                     # scalar
        theta_sum = self._theta_kernel().sum()                  # scalar

        # Broadcast d_weights to (window, 1, 1, …) so it can multiply u_history.
        broadcast_shape = (self.window,) + (1,) * (len(rest))
        d_view = d_weights.view(broadcast_shape)

        spikes = []
        u_history = self.u_history
        v_th = self.v_threshold

        for t in range(T):
            # ── 1. Membrane decay over window I ──
            # u_d = (1 − leak) · Σ_i d(i) · u_{t−i}
            u_recent = (d_view * u_history).sum(dim=0)
            u_d = (1.0 - leak) * u_recent

            # ── 2. Input integration ──
            u_s = u_d + x[t]

            # ── 3. Stochastic amplification (only when below threshold) ──
            delta = u_s - v_th
            below = (delta <= 0).to(x.dtype)
            # exp(β·δ) blows up for very negative δ if β is negative; clamp for safety.
            amp = self.amp_B * torch.exp(torch.clamp(self.amp_beta * delta, max=10.0))
            u_a = u_s * (1.0 + amp * below)

            # ── 4. Spike activation via bounded O(Δ) ──
            Delta = u_a - v_th
            pos = (Delta >= 0).to(x.dtype)
            # Clamp Delta to a sensible range for numerical stability of e^{−γΔ}.
            Delta_clip = Delta.clamp(min=0.0, max=20.0)
            o_val = Delta_clip * (1.0 + Delta_clip * torch.exp(-self.gamma * Delta_clip))
            o_val = torch.clamp(o_val, max=self.o_max) * pos

            if self.use_grad_mask and self.training:
                spike = _grad_masked_spike(u_a, Delta, o_val, self.grad_mask_alpha)
            else:
                spike = torch.sigmoid(u_a * o_val) * pos

            spikes.append(spike)

            # ── 5. Refractory inhibition ──
            u_t = u_a * (1.0 + h_sum)

            # ── 6. Adaptive threshold update (capped to prevent runaway) ──
            v_th = (v_th * (1.0 + theta_sum)).clamp(
                min=self.v_threshold_init * 0.1, max=self.v_threshold_clip
            )

            # ── 7. Update history ring ──
            u_history = torch.cat([u_t.unsqueeze(0), u_history[:-1]], dim=0)

        self.u_history = u_history
        self.v_threshold = v_th
        return torch.stack(spikes, dim=0)

    def extra_repr(self) -> str:
        return (
            f"window={self.window}, v_th_init={self.v_threshold_init}, "
            f"gamma={float(self.gamma):.2f}, o_max={self.o_max}, "
            f"grad_mask={self.use_grad_mask}"
        )


def _grad_masked_spike(
    u_a: torch.Tensor, Delta: torch.Tensor, o_val: torch.Tensor, alpha: float
) -> torch.Tensor:
    """CIFAR-style rectangular gradient mask: forward = σ(u_a·O(Δ)) · 1[Δ≥0],
    backward = α · 1[|Δ| < 1]. Uses straight-through to splice forward/backward."""
    pos = (Delta >= 0).to(u_a.dtype)
    spike_fwd = torch.sigmoid(u_a * o_val) * pos
    # Surrogate: linear in u_a inside the |Δ|<1 band, zero outside.
    band = (Delta.abs() < 1.0).to(u_a.dtype)
    surrogate = alpha * band * u_a
    return (spike_fwd - surrogate).detach() + surrogate


__all__ = ["POLARA"]
