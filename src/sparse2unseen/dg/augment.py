from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter


class DomainDiversifier:
    """Photometric source-domain diversification with geometry preserved."""

    def __init__(self, cfg: dict) -> None:
        self.jitter = ColorJitter(
            brightness=float(cfg.get("brightness", 0.35)),
            contrast=float(cfg.get("contrast", 0.35)),
            saturation=float(cfg.get("saturation", 0.25)),
            hue=0.05,
        )
        self.blur_probability = float(cfg.get("blur_probability", 0.25))

    def __call__(self, normalized_batch: torch.Tensor) -> torch.Tensor:
        # Convert [-1,1] normalized tensors back to [0,1] for torchvision ops.
        imgs = (normalized_batch * 0.5 + 0.5).clamp(0, 1)
        out = []
        for img in imgs:
            x = self.jitter(img)
            if torch.rand(()) < self.blur_probability:
                x = TF.gaussian_blur(x, kernel_size=[3, 3], sigma=[0.3, 1.5])
            out.append((x - 0.5) / 0.5)
        return torch.stack(out, dim=0)
