from typing import Tuple

from omegaconf import OmegaConf, DictConfig
import hydra

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from densenet import densenet121
from utils import get_loaders


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
    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        nesterov=cfg.optim.nesterov,
        weight_decay=cfg.optim.weight_decay,
    )
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


def train_step(
    cfg: DictConfig,
    model: Module,
    optim: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loss_fn: torch.nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device_type: str,
    clip_grad_fn,
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

    model = densenet121().to(cfg.device)
    if cfg.compile:
        model = torch.compile(model)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=lbls)
    train_loader, val_loader, test_loader = get_loaders(cfg)
    optim, scaler = make_optim_utils(cfg, model)

    clip_grad_fn = lambda m: torch.nn.utils.clip_grad_norm_(
        m.parameters(), cfg.train.max_grad_norm
    )

    step = 0
    for epoch in range(cfg.trainer.epochs):
        running_loss, correct, seen = 0.0, 0, 0

        for batch in train_loader:
            loss_item, grad, corr, bs = train_step(
                cfg, model, optim, scaler, loss_fn, batch, device_type, clip_grad_fn
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

    test_loss, test_acc = eval(cfg, model, test_loader, loss_fn)
    print(f"Test_loss={test_loss:.4f} test_acc={test_acc:.2%}")
    return model


def default_log(
    step: int, epoch: int, true_loss: torch.Tensor, grad: torch.Tensor, lr: float = 0.0
) -> None:
    print(
        f"Step: {step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad:.5f} | Lr: {lr:.3e}"
    )


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    assert cfg.device == "cuda"

    set_flags(cfg)
    model = train(cfg)


if __name__ == "__main__":
    main()
