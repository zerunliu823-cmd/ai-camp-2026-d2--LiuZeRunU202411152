"""Models for real concrete crack image classification."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """A small two-class network for 3 by 64 by 64 image tensors."""

    def __init__(self) -> None:
        super().__init__()
        # 输入张量: 3 通道 x 64 x 64
        # Conv2d(3, 8, k=3, pad=1)  -> 8 x 64 x 64
        # ReLU + MaxPool2d(2)       -> 8 x 32 x 32
        # Conv2d(8, 16, k=3, pad=1) -> 16 x 32 x 32
        # ReLU + MaxPool2d(2)       -> 16 x 16 x 16
        # Flatten                   -> 16 * 16 * 16 = 4096
        # Linear(4096, 2)           -> 2 类分数 (no_crack, crack)
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 把图像批次送入网络，得到每个样本的 2 类分数
        return self.network(images)


class CNN(nn.Module):
    """Configurable CNN for hyperparameter search.

    Built from ``filters`` conv blocks, each ``Conv2d -> [BatchNorm] -> ReLU
    -> MaxPool2d(2)``, followed by a final linear classifier for two classes.
    """

    def __init__(
        self,
        filters: tuple[int, ...] = (16, 32),
        kernel_size: int = 3,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_channels = 3
        blocks: list[nn.Module] = []
        for out_channels in filters:
            blocks.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            if batch_norm:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d(2))
            in_channels = out_channels
        blocks.append(nn.Flatten())
        if dropout:
            blocks.append(nn.Dropout(dropout))
        spatial = 64 // (2 ** len(filters))
        blocks.append(nn.Linear(filters[-1] * spatial * spatial, 2))
        self.network = nn.Sequential(*blocks)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class MLP(nn.Module):
    """Fully connected network that flattens raw pixels.

    Used as an architecture comparison baseline: it ignores the 2D spatial
    structure of the image and treats each pixel as an independent feature.
    """

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_features = 3 * 64 * 64
        layers: list[nn.Module] = []
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_features = units
        layers.append(nn.Linear(in_features, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images.flatten(1))
