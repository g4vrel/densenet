import math
import os
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module
from torch.utils.data import DataLoader

from densenet import densenet121
from utils import get_loaders, make_checkpoint, print_steps_info


def _data_root_and_download(cfg):
    root = Path(get_original_cwd()) / cfg.data.root
    root.mkdir(parents=True, exist_ok=True)
    download = bool(cfg.data.download) and not any(root.iterdir())
    cfg.data.root = str(root)
    cfg.data.download = download
    return cfg


def set_flags(cfg: DictConfig):
    """Set performance flags and seed."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def make_optim_utils(cfg: DictConfig, model: Module):
    optim = None
    if cfg.optim.name == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optim.lr,
            momentum=cfg.optim.momentum,
            nesterov=cfg.optim.nesterov,
            weight_decay=cfg.optim.weight_decay,
        )
    elif cfg.optim.name == "adamw":
        optim = torch.optim.AdamW(
            model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
    else:
        raise ValueError()
    scaler = torch.GradScaler(enabled=bool(cfg.amp))
    return optim, scaler


@torch.no_grad()
def eval(
    cfg: DictConfig, model: Module, loader: DataLoader, loss_fn
) -> Tuple[float, float]:
    model.eval()

    device = cfg.device
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"Eval | Loss: {avg_loss:.5f} | Acc: {accuracy:.2%}")

    return avg_loss, accuracy


def get_lr(config: DictConfig, step: int, batches_per_epoch: int) -> float:
    """Linear warmup from min_lr -> max_lr, then cosine decay back to min_lr."""
    warmup = int(config.optim.warmup_steps)
    total = batches_per_epoch * int(config.trainer.epochs)
    max_lr = float(config.optim.lr)
    min_lr = float(config.optim.min_lr)

    s = max(0, min(step, total))

    if s < warmup:
        return min_lr + (max_lr - min_lr) * (s / max(warmup, 1))

    t = (s - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def train_step(
    cfg: DictConfig,
    model: Module,
    optim: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loss_fn: torch.nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device_type: str,
    clip_grad_fn,
    step: int,
    batches_per_epoch: int,
) -> Tuple[float, float, int, int]:
    """Runs a single optimization step on one batch."""
    model.train()

    x, y = batch
    x, y = x.to(cfg.device), y.to(cfg.device)

    with torch.amp.autocast(device_type=device_type, enabled=bool(cfg.amp)):
        logits = model(x)
        loss = loss_fn(logits, y)

    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    grad = float(clip_grad_fn(model))

    lr_t = get_lr(cfg, step, batches_per_epoch)
    for g in optim.param_groups:
        g["lr"] = lr_t

    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = int((preds == y).sum().item())
        bs = int(y.size(0))

    return float(loss.item()), grad, correct, bs


def train(cfg: DictConfig) -> Module:
    device_type = cfg.device
    lbls = cfg.trainer.label_smoothing

    model = densenet121(
        separable_convs=cfg.model.separable_convs,
        num_min_groups=cfg.model.num_min_groups,
        dilation=cfg.model.dilation,
        dropout=cfg.model.dropout,
        dropout_type=cfg.model.dropout_type,
        norm_type=cfg.model.norm_type,
    ).to(cfg.device)
    if cfg.compile:
        model = torch.compile(model)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=lbls)
    cfg = _data_root_and_download(cfg)
    train_loader, val_loader, test_loader = get_loaders(cfg, root=cfg.data.root)
    optim, scaler = make_optim_utils(cfg, model)

    batches_per_epoch = len(train_loader)

    print_steps_info(train_loader)

    clip_grad_fn = lambda m: torch.nn.utils.clip_grad_norm_(
        m.parameters(), cfg.trainer.grad_clip_norm
    )

    step = 0
    epk = 0
    for epoch in range(cfg.trainer.epochs):
        epk += 1
        running_loss, correct, seen = 0.0, 0, 0

        for batch in train_loader:
            loss_item, grad, corr, bs = train_step(
                cfg,
                model,
                optim,
                scaler,
                loss_fn,
                batch,
                device_type,
                clip_grad_fn,
                step,
                batches_per_epoch,
            )
            step += 1
            running_loss += loss_item * bs
            correct += corr
            seen += bs

            if (step % cfg.trainer.log_interval) == 0:
                lr = optim.param_groups[0]["lr"]
                default_log(step, epoch, loss_item, grad, lr)

        train_loss = running_loss / seen
        train_acc = correct / seen

        val_loss, val_acc = eval(cfg, model, val_loader, loss_fn)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "train/epoch_loss": float(train_loss),
                    "train/epoch_acc": float(train_acc),
                    "val/loss": float(val_loss),
                    "val/acc": float(val_acc),
                    "epoch": epoch,
                    "step": step,
                },
                step=step,
            )

    test_loss, test_acc = eval(cfg, model, test_loader, loss_fn)
    print(f"Test_loss={test_loss:.4f} test_acc={test_acc:.2%}")

    if wandb.run is not None:
        wandb.log(
            {
                "test/loss": float(test_loss),
                "test/acc": float(test_acc),
                "step": step,
            },
            step=step,
        )
        wandb.summary["test/loss"] = float(test_loss)
        wandb.summary["test/acc"] = float(test_acc)

    make_checkpoint("final_model.pt", step, epk, model)

    return model


def default_log(
    step: int, epoch: int, true_loss: torch.Tensor, grad: torch.Tensor, lr: float = 0.0
) -> None:
    print(
        f"Step: {step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad:.5f} | Lr: {lr:.3e}"
    )
    if wandb.run is not None:
        wandb.log(
            {
                "train/step_loss": float(true_loss),
                "train/grad_norm": float(grad),
                "lr": float(lr),
                "epoch": epoch,
                "step": step,
            },
            step=step,
        )


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    assert cfg.device == "cuda"

    project = os.environ.get("WANDB_PROJECT", "runs")
    name = os.environ.get("WANDB_NAME", None)
    mode = "online"

    wandb.init(
        project=project,
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=mode,
    )

    set_flags(cfg)
    try:
        _ = train(cfg)
    finally:
        # Ensure the run closes cleanly even on exceptions
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
