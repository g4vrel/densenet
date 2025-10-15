import math

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


def make_norm(cfg):
    spec = cfg.data.norm
    assert spec in ["z_imagenet", "z", "div255", "minus_one"]
    if spec == "z_imagenet":
        mu = (0.485, 0.456, 0.406)
        sigma = (0.229, 0.224, 0.225)
    if spec == "z":
        mu = (0.484, 0.456, 0.397)
        sigma = (0.261, 0.255, 0.262)
    if spec == "div255":
        mu = (0.0, 0.0, 0.0)
        sigma = (1.0, 1.0, 1.0)
    if spec == "minus_one":
        mu = (0.5, 0.5, 0.5)
        sigma = (0.5, 0.5, 0.5)

    comp = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mu, sigma)]
    return comp


def get_loaders(config, root="data/"):
    size = config.data.img_size
    bs = config.data.batch_size
    j = config.data.num_workers
    val_split = config.data.val_split
    seed = config.seed

    train_transform = None
    if config.data.aug:
        train_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop(
                    size, scale=(0.7, 1.0), ratio=(3 / 4, 4 / 3), antialias=True
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2),
                v2.ToDtype(torch.float32, scale=True),
                *make_norm(config),
                v2.RandomErasing(p=0.25, scale=(0.02, 0.33)),
            ]
        )
    else:
        train_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(256, antialias=True),
                v2.CenterCrop(size),
                *make_norm(config),
            ]
        )

    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, antialias=True),
            v2.CenterCrop(size),
            v2.ToDtype(torch.float32, scale=True),
            *make_norm(config),
        ]
    )

    base = datasets.OxfordIIITPet(
        root=root, split="trainval", target_types="category", download=False
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
        download=False,
    )

    train_loader = DataLoader(
        Subset(train_ds, trn_idx),
        batch_size=bs,
        shuffle=True,
        num_workers=j,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(j > 0),
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=bs,
        shuffle=False,
        num_workers=j,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(j > 0),
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=j,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(j > 0),
        prefetch_factor=4,
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


def compute_mean_std(dataset, indices, img_size=224, batch_size=64, num_workers=4):
    stat_tf = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, antialias=False),
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),  # [0,1]
        ]
    )
    ds = Subset(
        datasets.OxfordIIITPet(
            dataset.root, split="trainval", target_types="category", transform=stat_tf
        ),
        indices,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    n_pixels = 0
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)

    with torch.no_grad():
        for x, _ in dl:
            b, c, h, w = x.shape
            npix = b * h * w
            n_pixels += npix
            x_ = x.permute(1, 0, 2, 3).reshape(c, -1)  # [3, B*H*W]
            channel_sum += x_.sum(dim=1)
            channel_sum_sq += (x_**2).sum(dim=1)

    mean = channel_sum / n_pixels
    var = (channel_sum_sq / n_pixels) - mean**2
    std = torch.sqrt(var.clamp(min=1e-12))
    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    root = "data/"
    size = 224
    bs = 128
    j = 16
    base = datasets.OxfordIIITPet(
        root=root, split="trainval", target_types="category", download=False
    )
    seed = 159753
    val_split = 0.1
    g = torch.Generator().manual_seed(seed)
    n_val = int(val_split * len(base))
    perm = torch.randperm(len(base), generator=g)
    val_idx, trn_idx = perm[:n_val], perm[n_val:]

    mean, std = compute_mean_std(
        base, trn_idx.tolist(), img_size=size, batch_size=bs, num_workers=j
    )
    print("Pets mean:", mean, "std:", std)
