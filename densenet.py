import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(3, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        dropout: float = 0.0,
        separable: bool = False,
        num_min_groups: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.dropout = dropout
        inter_channels = 4 * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = make_conv(
            inter_channels, growth_rate, separable, num_min_groups, dilation
        )

    def forward(self, prev_feats: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(prev_feats, dim=1)
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        if self.dropout and self.training:
            out = F.dropout(out, p=self.dropout)
        return out


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        dropout: float = 0.0,
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
                    dropout,
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
        return torch.cat(feats, 1)


class Transition(nn.Module):
    def __init__(self, in_channels: int, compression: float = 0.5):
        super().__init__()
        out_channels = int(math.floor(in_channels * compression))
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.norm(x), inplace=True))
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
                dropout,
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
        self.norm_final = nn.BatchNorm2d(channels)
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
        x = F.relu(self.norm_final(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        x = self.classifier(x)
        return x


def densenet121(
    num_classes: int = 37,
    separable_convs: bool = False,
    num_min_groups: int = 1,
    dilation: int = 1,
) -> DenseNet:
    return DenseNet(
        growth_rate=32,
        block_layers=[6, 12, 24, 16],
        compression=0.5,
        num_classes=num_classes,
        separable_convs=separable_convs,
        num_min_groups=num_min_groups,
        dilation=dilation,
    )


if __name__ == "__main__":
    model = densenet121(num_classes=37)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print("Output shape:", logits.shape)  # Expect [2, 37]
    print("Param count (M):", sum(p.numel() for p in model.parameters()) / 1e6)
