"""Train a SpikingLeNet on one of the four sensing tasks with LIF/DLIF/POLARA/DGN.

Auto-resume: by default, if `outputs/<task>_<neuron>/last.pt` exists, training
picks up from the next epoch. The optimizer state, history, and best_acc are
all restored so the run is bit-identical to an uninterrupted training (modulo
the data shuffle order, which depends on the seed).

Examples:
    python scripts/train.py --task cifar10 --neuron dlif --epochs 30
    python scripts/train.py --task ucihar  --neuron dgn  --epochs 50
    python scripts/train.py --task cifar10 --neuron dlif --epochs 30 --restart
    python scripts/train.py --task cifar10 --neuron dlif --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the project root importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
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


# ---------------------------------------------------------------------------
TASKS = {
    "cifar10": dict(
        loader=get_cifar10_loaders,
        model_cls=SpikingLeNet,
        model_cfg=dict(num_classes=10, input_channels=3, input_height=32, input_width=32),
        default_batch=128,
        default_T=4,
        default_hidden=256,
    ),
    "gsc": dict(
        loader=get_gsc_loaders,
        model_cls=SpikingLeNet1D,
        model_cfg=dict(num_classes=12, input_channels=64, input_length=101),
        default_batch=64,
        default_T=4,
        default_hidden=256,
    ),
    "ucihar": dict(
        loader=get_ucihar_loaders,
        model_cls=SpikingLeNet1D,
        model_cfg=dict(num_classes=6, input_channels=9, input_length=128),
        default_batch=128,
        default_T=4,
        default_hidden=128,
    ),
    "uthar": dict(
        loader=get_uthar_loaders,
        model_cls=SpikingLeNet1D,
        model_cfg=dict(num_classes=7, input_channels=90, input_length=250),
        default_batch=64,
        default_T=4,
        default_hidden=128,
    ),
}

# Args that must match between a checkpoint and a resuming run; mismatches
# are a sign the user is mixing experiments and should be warned about.
_ARGS_MUST_MATCH = ("task", "neuron", "T", "hidden_dim", "sparsity", "tau", "seed")


def build_neuron(name: str, tau: float, sparsity: float) -> nn.Module:
    if name == "lif":
        # In spikingjelly >=0.0.0.0.14, LIFNode(step_mode='m') is the multi-step LIF.
        return LIFNode(tau=tau, detach_reset=True, step_mode="m", backend="torch")
    if name == "dlif":
        return DLIF(tau=tau, sparsity=sparsity)
    if name == "polara":
        return POLARA()
    if name == "dgn":
        return DGN(tau_s=tau)
    raise ValueError(f"unknown neuron {name!r}")


def run_epoch(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    optimizer=None,
    criterion=None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0
    correct = 0
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
        if is_train:
            loss.backward()
            optimizer.step()
        functional.reset_net(model)
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _atomic_save(obj, path: Path) -> None:
    """torch.save that survives ungraceful kills mid-write.

    Writes to <path>.tmp, then atomically renames over <path>. On POSIX this
    is a single inode swap, so readers always see either the old file or the
    fully-written new file — never a half-written one.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _verify_checkpoint_args(prev_args: dict, args, label: str) -> None:
    """Print warnings for any structural argument that mismatches the saved run."""
    mismatches = []
    for k in _ARGS_MUST_MATCH:
        prev_v = prev_args.get(k)
        curr_v = getattr(args, k, None)
        # T / hidden_dim default to None in argparse; resolve via TASKS table.
        if curr_v is None and k in ("T", "hidden_dim"):
            cfg = TASKS[args.task]
            curr_v = cfg["default_T"] if k == "T" else cfg["default_hidden"]
        if prev_v is not None and curr_v is not None and prev_v != curr_v:
            mismatches.append(f"{k}: ckpt={prev_v!r} vs current={curr_v!r}")
    if mismatches:
        print(f"  [warning] {label} has different args than current run:")
        for m in mismatches:
            print(f"    - {m}")
        print("  proceeding anyway, but you probably want --restart or a fresh --output-dir.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS.keys()), required=True)
    parser.add_argument("--neuron", choices=["lif", "dlif", "polara", "dgn"], default="dlif")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--T", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="2-epoch smoke run")
    parser.add_argument("--resume", type=str, default=None,
                        help="explicit checkpoint path; takes precedence over auto-resume")
    parser.add_argument("--restart", action="store_true",
                        help="ignore any existing last.pt and start from epoch 0")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; run the test loop once on the best ckpt")
    parser.add_argument("--save-every", type=int, default=0,
                        help="also save outputs/<task>_<neuron>/epoch_<N>.pt every N epochs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = TASKS[args.task]
    batch_size = args.batch_size or cfg["default_batch"]
    T = args.T or cfg["default_T"]
    hidden = args.hidden_dim or cfg["default_hidden"]
    epochs = 2 if args.quick else args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | task: {args.task} | neuron: {args.neuron}")
    print(f"epochs={epochs}, batch={batch_size}, T={T}, hidden={hidden}, lr={args.lr}")

    train_loader, test_loader = cfg["loader"](
        args.data_root, batch_size=batch_size, num_workers=args.num_workers
    )

    neuron_proto = build_neuron(args.neuron, tau=args.tau, sparsity=args.sparsity)
    model_cfg = dict(cfg["model_cfg"])
    model_cfg.update(time_step=T, hidden_dim=hidden, neuron=neuron_proto)
    model = cfg["model_cls"](model_cfg).to(device)

    # Trigger lazy-init in DLIF / DGN (or any other shape-agnostic module) by
    # running a dummy forward BEFORE creating the optimizer — otherwise the
    # optimizer wouldn't see lazy-initialised parameters at all.
    model.eval()
    sample_x, _ = next(iter(test_loader))
    with torch.no_grad():
        model(sample_x.to(device))
    functional.reset_net(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out = Path(args.output_dir) / f"{args.task}_{args.neuron}"
    out.mkdir(parents=True, exist_ok=True)
    last_path = out / "last.pt"
    best_path = out / "best.pt"
    history_path = out / "history.json"

    # ── Resolve which checkpoint to resume from ───────────────────────────────
    # Priority:
    #   1. --resume <path>            (explicit override)
    #   2. last.pt in output dir      (auto-resume, default)
    #   3. best.pt in output dir      (only for --eval-only fallback)
    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume {resume_path} does not exist")
    elif args.eval_only and last_path.exists():
        resume_path = last_path
    elif args.eval_only and best_path.exists():
        resume_path = best_path
    elif not args.restart and last_path.exists():
        resume_path = last_path

    start_epoch = 0
    history: list[dict] = []
    best_acc = 0.0
    if resume_path is not None:
        print(f"[resume] loading {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"  state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        # Optimizer state is only saved by recent train runs (in last.pt).
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("  optimizer state restored")
            except (ValueError, KeyError) as e:
                print(f"  optimizer state could not be restored ({e}); using fresh optimizer")
        # Restore epoch counter / history / best_acc if present.
        history = list(ckpt.get("history", []))
        best_acc = float(ckpt.get("best_acc", 0.0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        # Verify args against the saved run (warn-only).
        _verify_checkpoint_args(ckpt.get("args", {}), args, label=resume_path.name)
        print(f"  resumed at epoch {start_epoch}  best_acc={best_acc:.4f}  history len={len(history)}")

    # ── Eval-only short-circuit ───────────────────────────────────────────────
    if args.eval_only:
        te_loss, te_acc = run_epoch(model, test_loader, device=device,
                                    optimizer=None, criterion=criterion)
        print(f"[eval-only] test_loss={te_loss:.4f}  test_acc={te_acc:.4f}")
        return

    # ── Skip if already trained ───────────────────────────────────────────────
    if start_epoch >= epochs:
        print(
            f"[done] checkpoint already trained {start_epoch} epoch(s) "
            f"(≥ requested {epochs}). Use --restart to start over, or pass "
            f"--epochs > {start_epoch} to continue."
        )
        print(f"best test acc so far: {best_acc:.4f}  (artefacts under {out})")
        return

    if start_epoch > 0:
        print(f"continuing training from epoch {start_epoch} to {epochs - 1}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device=device,
                                    optimizer=optimizer, criterion=criterion)
        te_loss, te_acc = run_epoch(model, test_loader, device=device,
                                    optimizer=None, criterion=criterion)
        elapsed = time.time() - t0
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            test_loss=te_loss, test_acc=te_acc, time_s=elapsed))
        print(
            f"epoch {epoch:3d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"| test_loss={te_loss:.4f} test_acc={te_acc:.4f}  ({elapsed:.1f}s)"
        )

        # Save best (model + args; small, no optimizer needed).
        if te_acc > best_acc:
            best_acc = te_acc
            _atomic_save(
                {"model": model.state_dict(), "args": vars(args),
                 "epoch": epoch, "best_acc": best_acc},
                best_path,
            )

        # Save last (full resume state, every epoch).
        _atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_acc": best_acc,
                "args": vars(args),
            },
            last_path,
        )

        # Save optional per-epoch snapshot.
        if args.save_every and (epoch + 1) % args.save_every == 0:
            _atomic_save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                out / f"epoch_{epoch + 1}.pt",
            )

        # Persist history.json after every epoch (not just at the end). This
        # makes mid-run analysis possible and survives crashes.
        _atomic_write_text(
            json.dumps(
                {"history": history, "best_acc": best_acc,
                 "n_params": n_params, "args": vars(args)},
                indent=2,
            ),
            history_path,
        )

    print(f"best test acc: {best_acc:.4f}  (artefacts under {out})")


if __name__ == "__main__":
    main()
