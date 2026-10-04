"""
Supervised baseline training for the 2D FNO (theta prediction).

Loss = MSE(predicted_theta, target_theta)

Does not include physics loss. Does not modify HDF5 / dataset_generator files.

Usage
-----
    python -m training.train_fno --config configs/fno_baseline.yaml

Verification only (no training):
    python -m training.train_fno --verify-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_data import N_INPUT_CHANNELS, make_dataloader  # noqa: E402
from models.fno import FNO2d, build_fno_from_config  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "fno_baseline.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Return True if this step improved the best metric."""
        improved = metric < (self.best - self.min_delta)
        if improved:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return improved


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    t = cfg["train"]
    name = str(t.get("optimizer", "adamw")).lower()
    lr = float(t["lr"])
    wd = float(t.get("weight_decay", 0.0))
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    t = cfg["train"]
    name = str(t.get("scheduler", "cosine")).lower()
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(t.get("scheduler_T_max", t.get("epochs", 100))),
            eta_min=float(t.get("scheduler_eta_min", 1e-6)),
        )
    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, int(t.get("early_stopping_patience", 15)) // 3),
            min_lr=float(t.get("scheduler_eta_min", 1e-6)),
        )
    if name in ("none", "null", "off"):
        return None
    raise ValueError(f"Unknown scheduler: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, criterion) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(X)
        loss = criterion(pred, y)
        bs = X.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


def train_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float],
    log_every: int,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for step, (X, y) in enumerate(loader, start=1):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = X.shape[0]
        total_loss += loss.item() * bs
        n += bs
        if log_every and step % log_every == 0:
            print(
                f"  epoch {epoch:03d} step {step:04d}  "
                f"batch_mse={loss.item():.6e}  lr={optimizer.param_groups[0]['lr']:.3e}"
            )
    return total_loss / max(n, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_val: float,
    history: list,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val,
        "history": history,
        "config": cfg,
        "n_parameters": count_parameters(model),
    }
    torch.save(payload, path)


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    print(f"Device: {device}")

    model = build_fno_from_config(cfg).to(device)
    n_params = count_parameters(model)
    print(f"Model: FNO2d  params={n_params:,}")

    data_cfg = cfg["data"]
    train_loader = make_dataloader(
        "train",
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        normalize=bool(data_cfg.get("normalize", True)),
    )
    val_loader = make_dataloader(
        "validation",
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        normalize=bool(data_cfg.get("normalize", True)),
    )

    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    tcfg = cfg["train"]
    early = EarlyStopping(
        patience=int(tcfg.get("early_stopping_patience", 15)),
        min_delta=float(tcfg.get("early_stopping_min_delta", 0.0)),
    )
    paths = cfg["paths"]
    ckpt_dir = PROJECT_ROOT / paths["checkpoint_dir"]
    best_path = PROJECT_ROOT / paths["best_checkpoint"]
    last_path = PROJECT_ROOT / paths["last_checkpoint"]
    history_path = PROJECT_ROOT / paths["history_path"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_val = float("inf")
    epochs = int(tcfg["epochs"])
    grad_clip = tcfg.get("grad_clip", None)
    log_every = int(tcfg.get("log_every", 50))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, device, criterion, optimizer,
            grad_clip=float(grad_clip) if grad_clip is not None else None,
            log_every=log_every,
            epoch=epoch,
        )
        val_loss = evaluate(model, val_loader, device, criterion)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "seconds": time.time() - t0,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"train_mse={train_loss:.6e}  val_mse={val_loss:.6e}  "
            f"lr={lr:.3e}  ({record['seconds']:.1f}s)"
        )

        improved = early.step(val_loss)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val, history, cfg)
        if improved:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val, history, cfg)
            print(f"  -> new best checkpoint: {best_path}  (val_mse={best_val:.6e})")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({"history": history, "best_val_loss": best_val, "n_parameters": n_params}, f, indent=2)

        if early.should_stop:
            print(f"Early stopping at epoch {epoch} (patience={early.patience})")
            break

    print(f"Training finished. Best val MSE={best_val:.6e}")
    print(f"Best checkpoint: {best_path}")
    return {"best_val_loss": best_val, "history": history, "n_parameters": n_params}


def verify_forward(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Import / construct / one real DataLoader batch forward pass.
    Does not train.
    """
    cfg = cfg or load_config(DEFAULT_CONFIG)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()

    print("=" * 60)
    print("FNO baseline verification (no training)")
    print("=" * 60)
    print(f"Device: {device}")

    # 1) import / construct
    model = build_fno_from_config(cfg).to(device)
    n_params = count_parameters(model)
    print(f"[1] Model constructed: {model.__class__.__name__}")
    print(f"    width={model.width}  modes=({model.modes1},{model.modes2})  "
          f"layers={model.n_layers}  use_grid={model.use_grid}")
    print(f"[2] Trainable parameters: {n_params:,}")

    # 2) one real batch
    loader = make_dataloader(
        "train",
        batch_size=min(8, int(cfg["data"]["batch_size"])),
        shuffle=False,
        num_workers=0,
        normalize=bool(cfg["data"].get("normalize", True)),
    )
    X, y = next(iter(loader))
    X = X.to(device)
    y = y.to(device)
    print(f"[3] Real batch from ml_data DataLoader:")
    print(f"    X shape: {tuple(X.shape)}")
    print(f"    y shape: {tuple(y.shape)}")

    assert X.ndim == 4 and X.shape[1] == N_INPUT_CHANNELS, X.shape
    assert y.shape[1] == 1, y.shape
    assert torch.isfinite(X).all(), "NaN or Inf found in input batch X"
    assert torch.isfinite(y).all(), "NaN or Inf found in target batch y"
    print("    Input X finite check: PASS (no NaN/Inf)")
    print("    Target y finite check: PASS (no NaN/Inf)")

    model.eval()
    with torch.no_grad():
        pred = model(X)

    print(f"[4] Forward pass:")
    print(f"    pred shape: {tuple(pred.shape)}")
    assert pred.shape == y.shape, (pred.shape, y.shape)
    assert pred.shape == (X.shape[0], 1, 32, 32), pred.shape
    assert torch.isfinite(pred).all(), "NaN or Inf found in prediction pred"
    print("    Prediction pred finite check: PASS (no NaN/Inf)")

    mse = torch.nn.functional.mse_loss(pred, y).item()
    print(f"    initial MSE (untrained): {mse:.6e}")
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f"    CUDA memory allocated: {mem_allocated:.2f} MB")
    print("=" * 60)
    print("VERIFY: PASS")
    print("=" * 60)
    return {
        "n_parameters": n_params,
        "input_shape": tuple(X.shape),
        "target_shape": tuple(y.shape),
        "output_shape": tuple(pred.shape),
        "device": str(device),
        "initial_mse": mse,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train or verify baseline FNO2d")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to YAML config",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Run import/construction/forward checks only (no training)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.verify_only:
        verify_forward(cfg)
        return 0
    train(cfg)
    return 0


if __name__ == "__main__":
    # Default when invoked as a script: train. Use --verify-only for checks.
    sys.exit(main())
