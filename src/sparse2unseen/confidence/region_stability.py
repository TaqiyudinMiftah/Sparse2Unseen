from __future__ import annotations

import torch


def region_counts(density: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Return regional counts with shape [B, Gh, Gw]."""
    if density.ndim != 4 or density.shape[1] != 1:
        raise ValueError("density must have shape [B,1,H,W]")
    gh, gw = grid
    h, w = density.shape[-2:]
    if not 1 <= gh <= h or not 1 <= gw <= w:
        raise ValueError("grid must define nonempty regions")
    # Disjoint partitions also preserve counts at non-divisible resolutions.
    return torch.stack([
        torch.stack([
            density[:, 0, i*h//gh:(i+1)*h//gh, j*w//gw:(j+1)*w//gw].sum((-2, -1))
            for j in range(gw)
        ], dim=-1) for i in range(gh)
    ], dim=-2)


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
    if predictions.ndim != 5 or predictions.shape[0] < 1 or beta < 0:
        raise ValueError("predictions must have shape [K,B,1,H,W]")
    regional = torch.stack([region_counts(p, grid) for p in predictions], dim=0)
    variance = regional.var(dim=0, unbiased=False)
    weights = torch.exp(-beta * variance)
    _, b, _, h, w = predictions.shape
    result = predictions.new_empty(b, 1, h, w)
    gh, gw = grid
    for i in range(gh):
        for j in range(gw):
            result[:, :, i*h//gh:(i+1)*h//gh, j*w//gw:(j+1)*w//gw] = weights[:, i, j, None, None, None]
    return result
