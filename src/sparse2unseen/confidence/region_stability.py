from __future__ import annotations

import torch
import torch.nn.functional as F


def region_counts(density: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Return regional counts with shape [B, Gh, Gw]."""
    if density.ndim != 4 or density.shape[1] != 1:
        raise ValueError("density must have shape [B,1,H,W]")
    gh, gw = grid
    pooled = F.adaptive_avg_pool2d(density, (gh, gw))
    h, w = density.shape[-2:]
    # Average * approximate pixels per adaptive cell = count.
    return pooled[:, 0] * (h * w / (gh * gw))


def stability_weights(
    predictions: torch.Tensor,
    grid: tuple[int, int] = (4, 4),
    beta: float = 4.0,
) -> torch.Tensor:
    """Compute region weights from multiple teacher predictions.

    Args:
        predictions: [K, B, 1, H, W] density predictions from K diversified views.
    Returns:
        Pixel weight map [B,1,H,W] in (0,1].
    """
    if predictions.ndim != 5:
        raise ValueError("predictions must have shape [K,B,1,H,W]")
    regional = torch.stack([region_counts(p, grid) for p in predictions], dim=0)
    variance = regional.var(dim=0, unbiased=False)
    weights = torch.exp(-beta * variance)
    b, _, _, h, w = predictions.shape
    return F.interpolate(weights.unsqueeze(1), size=(h, w), mode="nearest")
