import math

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.v2 as v2


def get_loaders(config, root="data/"):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    size = config.data.img_size
    bs = config.data.batch_size
    j = config.data.num_workers
    val_split = config.data.val_split
    seed = config.seed

    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                size, scale=(0.7, 1.0), ratio=(3 / 4, 4 / 3), antialias=True
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, antialias=True),
            v2.CenterCrop(size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    base = datasets.OxfordIIITPet(
        root=root, split="trainval", target_types="category", download=True
    )
    g = torch.Generator().manual_seed(seed)
    n_val = int(val_split * len(base))
    perm = torch.randperm(len(base), generator=g)
    val_idx, trn_idx = perm[:n_val], perm[n_val:]

    train_ds = datasets.OxfordIIITPet(
        root=root, split="trainval", target_types="category", transform=train_transform
    )
    val_ds = datasets.OxfordIIITPet(
        root=root, split="trainval", target_types="category", transform=test_transform
    )
    test_ds = datasets.OxfordIIITPet(
        root=root,
        split="test",
        target_types="category",
        transform=test_transform,
        download=True,
    )

    train_loader = DataLoader(
        Subset(train_ds, trn_idx),
        batch_size=bs,
        shuffle=True,
        num_workers=j,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=bs,
        shuffle=False,
        num_workers=j,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=j,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def make_checkpoint(path, step, epoch, model, optim=None, scaler=None, ema_model=None):
    checkpoint = {
        "epoch": int(epoch),
        "step": int(step),
        "model_state_dict": model.state_dict(),
    }

    if optim is not None:
        checkpoint["optim_state_dict"] = optim.state_dict()

    if ema_model is not None:
        checkpoint["ema_model_state_dict"] = ema_model.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optim=None, scaler=None, ema_model=None):
    checkpoint = torch.load(path, weights_only=True)
    step = int(checkpoint["step"])
    epoch = int(checkpoint["epoch"])

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if optim is not None:
        optim.load_state_dict(checkpoint["optim_state_dict"])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        ema_model.eval()

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    model.eval()

    return step, epoch, model, optim, scaler, ema_model


def print_steps_info(loader: DataLoader):
    batches_per_epoch = len(loader)
    samples_per_epoch = len(loader.dataset)
    steps_per_epoch = math.ceil(batches_per_epoch)
    print(f"samples/epoch={samples_per_epoch} | batches/epoch={batches_per_epoch}")
