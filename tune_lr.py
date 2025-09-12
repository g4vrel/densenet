import argparse
import copy
import math
import random

import logging
import sys
from pathlib import Path

from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from densenet import densenet121
from utils import get_loaders


def make_logger(log_path: str = "lr_log.txt", level=logging.INFO):
    logger = logging.getLogger("lr_tuner")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = make_logger("lr_log.txt")


def sample_log_uniform(low: float, high: float) -> float:
    """Sample log-uniform in [low, high]."""
    return math.exp(random.uniform(math.log(low), math.log(high)))


def set_optimizer_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for g in optim.param_groups:
        g["lr"] = lr


@torch.no_grad()
def eval_model(
    model: torch.nn.Module, loader: DataLoader, loss_fn, device: str
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)
    if total_samples == 0:
        return float("inf"), 0.0
    return total_loss / total_samples, total_correct / total_samples


def short_train_step(
    model: torch.nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optim: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loss_fn,
    device: str,
    clip_norm: float,
    amp: bool,
) -> float:
    model.train()
    x, y = batch
    x, y = x.to(device), y.to(device)
    with torch.amp.autocast(device_type=device, enabled=bool(amp)):
        logits = model(x)
        loss = loss_fn(logits, y)

    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)
    return float(loss.item())


def run_trial(
    init_state: Dict[str, torch.Tensor],
    cfg: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    lr: float,
    trial_steps: int,
    device: str,
) -> Dict[str, Any]:
    """Restore model from init_state, run short training, eval on val, return metrics."""
    model = densenet121().to(device)
    model.load_state_dict(init_state)

    optim = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=cfg.optim.get("momentum", 0.9),
        nesterov=cfg.optim.get("nesterov", False),
        weight_decay=cfg.optim.get("weight_decay", 0.0),
    )
    scaler = torch.GradScaler(enabled=bool(cfg.amp))

    device_type = device
    step = 0
    best_seen_loss = float("inf")
    running_loss = 0.0

    train_iter = iter(train_loader)

    while step < trial_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss_item = short_train_step(
            model,
            batch,
            optim,
            scaler,
            loss_fn,
            device_type,
            cfg.trainer.grad_clip_norm,
            cfg.amp,
        )
        running_loss += loss_item
        step += 1

        # Divergence check
        if math.isnan(loss_item) or loss_item > 1e4:
            return {
                "status": "diverged",
                "lr": lr,
                "val_loss": float("inf"),
                "val_acc": 0.0,
            }

        if running_loss / step < best_seen_loss:
            best_seen_loss = running_loss / step

    val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)
    return {"status": "ok", "lr": lr, "val_loss": val_loss, "val_acc": val_acc}


def tune_loop(
    cfg,
    num_trials: int,
    trial_steps: int,
    low: float,
    high: float,
    seed: int,
    device: str,
):
    train_loader, val_loader, _ = get_loaders(cfg, root=cfg.data.root)

    # Fair starting point
    init_model = densenet121().to(device)
    init_state = copy.deepcopy(init_model.state_dict())

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.trainer.label_smoothing)

    results = []
    for i in range(num_trials):
        lr = sample_log_uniform(low, high)
        logger.info(f"[trial {i + 1}/{num_trials}] testing lr={lr:.3e}")
        rec = run_trial(
            init_state, cfg, train_loader, val_loader, loss_fn, lr, trial_steps, device
        )
        results.append(rec)
        if rec["status"] == "diverged":
            logger.info(f"  -> Diverged (lr = {lr:.3e})")
        else:
            logger.info(
                f"  -> val_loss={rec['val_loss']:.4f} val_acc={rec['val_acc']:.2%}"
            )

    ok = [r for r in results if r["status"] == "ok"]
    ok_sorted = sorted(ok, key=lambda r: r["val_loss"])
    logger.info("\nTop candidates:")
    for r in ok_sorted[:10]:
        logger.info(
            f"lr={r['lr']:.3e} | val_loss={r['val_loss']:.4f} | val_acc={r['val_acc']:.2%}"
        )

    return ok_sorted


def parse_args():
    p = argparse.ArgumentParser(description="Simple log-uniform LR tuner.")
    p.add_argument(
        "--cfg-path",
        type=str,
        required=True,
    )
    p.add_argument("--num-trials", type=int, default=30)
    p.add_argument(
        "--trial-steps", type=int, default=500, help="max mini-batches per trial"
    )
    p.add_argument("--low", type=float, default=1e-6)
    p.add_argument("--high", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        from omegaconf import OmegaConf
    except Exception as e:
        raise

    cfg = OmegaConf.load(args.cfg_path)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info(f"Device: {cfg.device} | Bs: {cfg.data.batch_size}")

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    best = tune_loop(
        cfg,
        args.num_trials,
        args.trial_steps,
        args.low,
        args.high,
        args.seed,
        args.device,
    )
