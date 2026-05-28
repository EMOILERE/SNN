"""Evaluate a trained checkpoint and report rich metrics.

Loads `outputs/<task>_<neuron>/best.pt` (or any other --ckpt), runs the test
loop once, and prints:
  - overall test accuracy / loss
  - per-class accuracy + confusion matrix
  - average spike firing rate per layer (useful for energy estimation)
  - theoretical energy cost per inference (45 nm CMOS, paper §A.6.1)

Run:
    python scripts/evaluate.py --task cifar10 --neuron dlif
    python scripts/evaluate.py --task gsc    --ckpt outputs/gsc_dgn/epoch_30.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.neuron import LIFNode

from datasets import (
    get_cifar10_loaders,
    get_gsc_loaders,
    get_ucihar_loaders,
    get_uthar_loaders,
)
from models import SpikingLeNet, SpikingLeNet1D
from neurons import DGN, DLIF, POLARA

# Same TASKS dict as train.py — duplicated here so this script is standalone.
TASKS = {
    "cifar10": dict(loader=get_cifar10_loaders, model_cls=SpikingLeNet,
                    model_cfg=dict(num_classes=10, input_channels=3,
                                   input_height=32, input_width=32),
                    default_batch=128, default_T=4, default_hidden=256),
    "gsc": dict(loader=get_gsc_loaders, model_cls=SpikingLeNet1D,
                model_cfg=dict(num_classes=12, input_channels=64, input_length=101),
                default_batch=64, default_T=4, default_hidden=256),
    "ucihar": dict(loader=get_ucihar_loaders, model_cls=SpikingLeNet1D,
                   model_cfg=dict(num_classes=6, input_channels=9, input_length=128),
                   default_batch=128, default_T=4, default_hidden=128),
    "uthar": dict(loader=get_uthar_loaders, model_cls=SpikingLeNet1D,
                  model_cfg=dict(num_classes=7, input_channels=90, input_length=250),
                  default_batch=64, default_T=4, default_hidden=128),
}


def build_neuron(name, tau, sparsity):
    if name == "lif":
        return LIFNode(tau=tau, detach_reset=True, step_mode="m", backend="torch")
    if name == "dlif":
        return DLIF(tau=tau, sparsity=sparsity)
    if name == "polara":
        return POLARA()
    if name == "dgn":
        return DGN(tau_s=tau)
    raise ValueError(f"unknown neuron {name!r}")


def hook_firing_rates(model: nn.Module) -> tuple[dict, list]:
    """Register forward hooks on every neuron-like layer to record firing rates."""
    stats: dict[str, list[float]] = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                # out shape: (T, B, C, ...) — average firing rate across all dims
                rate = out.float().clamp(0, 1).mean().item()
                stats.setdefault(name, []).append(rate)
        return hook

    for name, m in model.named_modules():
        if isinstance(m, (LIFNode, DLIF, POLARA, DGN)):
            handles.append(m.register_forward_hook(make_hook(name)))
    return stats, handles


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Run test loop; return overall + per-class accuracy + confusion matrix."""
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        functional.reset_net(model)
        total_loss += loss.item() * y.size(0)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for t, p in zip(y.cpu(), pred.cpu()):
            confusion[t, p] += 1
    per_class = (confusion.diag().float()
                 / confusion.sum(dim=1).clamp(min=1).float()).tolist()
    return total_loss / total, correct / total, per_class, confusion


def estimate_energy(model, firing_rates, T, n_params):
    """Theoretical energy (45 nm CMOS, paper §A.6.1).

    E_SNN ≈ T · r · (E_AC · N_AC + E_MAC · N_MAC)

    For a rough estimate we treat all parameters as one MAC each per timestep,
    weighted by the average firing rate of incoming spikes. This is the same
    approximation the DLIF paper uses for reporting.
    """
    E_MAC = 4.6e-12  # J
    E_AC = 0.9e-12
    mean_rate = sum(rs[0] for rs in firing_rates.values()) / max(len(firing_rates), 1)
    # Approximation: each parameter is an AC per timestep if pre-spike fires.
    energy = T * mean_rate * n_params * E_AC
    return energy, mean_rate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=list(TASKS.keys()), required=True)
    p.add_argument("--neuron", choices=["lif", "dlif", "polara", "dgn"], default="dlif")
    p.add_argument("--ckpt", type=str, default=None,
                   help="path to checkpoint; defaults to outputs/<task>_<neuron>/best.pt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--T", type=int)
    p.add_argument("--hidden-dim", type=int)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-confusion", action="store_true",
                   help="skip printing the confusion matrix")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TASKS[args.task]
    batch_size = args.batch_size or cfg["default_batch"]
    T = args.T or cfg["default_T"]
    hidden = args.hidden_dim or cfg["default_hidden"]
    num_classes = cfg["model_cfg"]["num_classes"]

    ckpt_path = args.ckpt or f"outputs/{args.task}_{args.neuron}/best.pt"
    print(f"loading {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    _, test_loader = cfg["loader"](args.data_root, batch_size=batch_size,
                                   num_workers=args.num_workers)
    neuron_proto = build_neuron(args.neuron, tau=args.tau, sparsity=args.sparsity)
    model_cfg = dict(cfg["model_cfg"])
    model_cfg.update(time_step=T, hidden_dim=hidden, neuron=neuron_proto)
    model = cfg["model_cls"](model_cfg).to(device)
    # Trigger lazy-init by one dummy forward.
    sample_x, _ = next(iter(test_loader))
    with torch.no_grad():
        model(sample_x.to(device))
    functional.reset_net(model)
    model.load_state_dict(state["model"], strict=False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rates, handles = hook_firing_rates(model)
    loss, acc, per_class, confusion = evaluate(model, test_loader, device, num_classes)
    for h in handles:
        h.remove()

    print(f"\n== Evaluation: task={args.task}, neuron={args.neuron} ==")
    print(f"  overall test_loss = {loss:.4f}")
    print(f"  overall test_acc  = {acc:.4f}   ({100*acc:.2f}%)")
    print(f"  trainable params  = {n_params:,}")

    print("\n== Per-class accuracy ==")
    for c, a in enumerate(per_class):
        print(f"  class {c:2d}: {a:.4f}")

    if not args.no_confusion and num_classes <= 16:
        print("\n== Confusion matrix (rows = true, cols = pred) ==")
        print(confusion.numpy())

    print("\n== Spike firing rates per layer ==")
    for name, rs in rates.items():
        print(f"  {name:30s}: {rs[0]:.4f}")

    energy, mean_rate = estimate_energy(model, rates, T, n_params)
    print(f"\n== Energy (rough estimate, 45 nm CMOS) ==")
    print(f"  mean firing rate        = {mean_rate:.4f}")
    # Auto-pick unit
    if energy >= 1e-3:
        print(f"  per-inference energy    = {energy*1e3:.3f} mJ")
    elif energy >= 1e-6:
        print(f"  per-inference energy    = {energy*1e6:.3f} uJ")
    else:
        print(f"  per-inference energy    = {energy*1e9:.3f} nJ")


if __name__ == "__main__":
    main()
