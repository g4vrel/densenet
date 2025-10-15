import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(num_channels: int, norm_type: str = "batch") -> nn.Module:
    norm_type = (norm_type or "batch").lower()
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "layernorm":
        return nn.GroupNorm(1, num_channels, affine=True)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


def make_dropout(p: float, dropout_type: str = "standard") -> nn.Module:
    p = float(p or 0.0)
    if p <= 0.0:
        return nn.Identity()
    dt = (dropout_type or "standard").lower()
    if dt in "standard":
        return nn.Dropout(p)
    elif dt in "spatial":
        return nn.Dropout2d(p)
    else:
        raise ValueError(f"Unknown dropout_type: {dropout_type}")


def make_conv(
    inter_channels: int,
    growth_rate,
    separable: bool = False,
    num_min_groups: int = 1,
    dilation: int = 1,
) -> nn.Module:
    if separable:
        return nn.Sequential(
            nn.Conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=dilation,
                groups=inter_channels,
                bias=False,
                dilation=dilation,
            ),
            nn.Conv2d(
                inter_channels,
                growth_rate,
                kernel_size=1,
                bias=False,
                groups=num_min_groups,
            ),
        )
    else:
        return nn.Conv2d(
            inter_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )


def make_stem(init_features: int, separable: bool = False) -> nn.Module:
    if separable:
        return nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3, groups=3, bias=False),
            nn.Conv2d(3, init_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(3, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        dropout: float = 0.0,
        separable: bool = False,
        num_min_groups: int = 1,
        dilation: int = 1,
        norm_type: str = "batch",
        dropout_type: str = "standard",
    ):
        super().__init__()
        self.dropout = dropout
        inter_channels = 4 * growth_rate
        self.norm1 = make_norm(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.norm2 = make_norm(inter_channels, norm_type)
        self.conv2 = make_conv(
            inter_channels, growth_rate, separable, num_min_groups, dilation
        )
        self.drop = make_dropout(dropout, dropout_type)
        self.se = SE(growth_rate, r=16)

    def forward(self, prev_feats: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(prev_feats, dim=1)
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        out = self.se(out)
        out = self.drop(out)
        return out


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        dropout: float = 0.0,
        dropout_type: str = "standard",
        norm_type: str = "batch",
        separable: bool = False,
        num_min_groups: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(
                DenseLayer(
                    channels,
                    growth_rate,
                    dropout=dropout,
                    dropout_type=dropout_type,
                    norm_type=norm_type,
                    separable=separable,
                    num_min_groups=num_min_groups,
                    dilation=dilation,
                )
            )
            channels += growth_rate
        self.block = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for layer in self.block:
            feat = layer(feats)
            feats.append(feat)
        out = torch.cat(feats, 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels: int, compression: float = 0.5):
        super().__init__()
        out_channels = int(math.floor(in_channels * compression))
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_layers: List[int] = [6, 12, 24, 16],
        compression: float = 0.5,
        num_classes: int = 37,
        init_features: int = 64,
        dropout: float = 0.0,
        separable_convs: bool = False,
        num_min_groups: int = 1,
        dilation: int = 1,
        dropout_type: str = "standard",
        norm_type: str = "batch",
    ):
        super().__init__()

        self.stem = make_stem(init_features, separable_convs)

        channels = init_features
        blocks = []

        for i, num_layers in enumerate(block_layers):
            db = DenseBlock(
                num_layers,
                channels,
                growth_rate,
                dropout=dropout,
                dropout_type=dropout_type,
                norm_type=norm_type,
                separable=separable_convs,
                num_min_groups=num_min_groups,
                dilation=dilation,
            )
            blocks.append(db)
            channels = db.out_channels

            if i != len(block_layers) - 1:
                tr = Transition(channels, compression=compression)
                blocks.append(tr)
                channels = tr.out_channels

        self.features = nn.Sequential(*blocks)
        self.norm_final = make_norm(channels, norm_type)
        self.classifier = nn.Linear(channels, num_classes)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = F.relu(self.norm_final(x))
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        x = self.classifier(x)
        return x


def densenet121(
    num_classes: int = 37,
    separable_convs: bool = False,
    num_min_groups: int = 1,
    dilation: int = 1,
    dropout: float = 0.0,
    dropout_type: str = "standard",
    norm_type: str = "batch",
) -> DenseNet:
    return DenseNet(
        growth_rate=32,
        block_layers=[6, 12, 24, 16],
        compression=0.5,
        num_classes=num_classes,
        separable_convs=separable_convs,
        num_min_groups=num_min_groups,
        dilation=dilation,
        dropout=dropout,
        dropout_type=dropout_type,
        norm_type=norm_type,
    )


if __name__ == "__main__":
    model = densenet121(num_classes=37)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print("Output shape:", logits.shape)  # Expect [2, 37]
    print("Param count (M):", sum(p.numel() for p in model.parameters()) / 1e6)
