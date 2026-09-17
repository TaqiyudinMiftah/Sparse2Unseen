from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG16_BN_Weights, vgg16_bn


class VGG16Density(nn.Module):
    """Simple VGG16-BN density regressor for controlled baseline experiments.

    This is not an implementation of MPCount. It is a compact standalone baseline
    that makes B0/B1/B3 experiments executable while MPCount is kept external.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vgg16_bn(weights=weights)
        # Stop before the final pooling layer; output stride is 16.
        self.features = nn.Sequential(*list(backbone.features.children())[:-1])
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Softplus(beta=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get("name", "vgg16_density")
    if name != "vgg16_density":
        raise ValueError(f"Unknown model: {name}")
    return VGG16Density(pretrained=bool(cfg.get("pretrained", True)))
