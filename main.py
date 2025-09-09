from omegaconf import OmegaConf, DictConfig
import hydra

import numpy as np
import torch
from torch.nn import Module

from densenet import densenet121
from utils import get_loaders


def set_flags(cfg: DictConfig):
    """Set performance flags and seed."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def make_optimizer(model: Module):
    optim = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=cfg.train.momentum,
        nesterov=cfg.train.nesterov,
        weight_decay=cfg.train.weight_decay,
    )

def train(cfg: DictConfig) -> Module:
    device = cfg.device
    lbls = cfg.trainer.label_smoothing

    model = densenet121().to(device)
    if cfg.compile:
        model = torch.compile(model)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=lbls)
    train_loader, val_loader, test_loader = get_loaders(cfg)


@torch.no_grad()
def eval() -> None: ...


def default_log(
    step: int, epoch: int, true_loss: torch.Tensor, grad: torch.Tensor, lr: float = 0.0
) -> None:
    print(
        f"Step: {step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad:.5f} | Lr: {lr:.3e}"
    )


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_flags(cfg)
