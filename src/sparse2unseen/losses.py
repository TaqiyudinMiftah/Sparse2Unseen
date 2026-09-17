from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_density_preserve_count(target: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if target.shape[-2:] == shape:
        return target
    old_sum = target.sum(dim=(-2, -1), keepdim=True)
    resized = F.interpolate(target, size=shape, mode="bilinear", align_corners=False)
    new_sum = resized.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return resized * (old_sum / new_sum)


def supervised_density_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = resize_density_preserve_count(target, pred.shape[-2:])
    return F.mse_loss(pred, target)


def weighted_consistency_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    weight_map: torch.Tensor | None = None,
) -> torch.Tensor:
    if teacher.shape[-2:] != student.shape[-2:]:
        teacher = resize_density_preserve_count(teacher, student.shape[-2:])
    sq = (student - teacher.detach()).pow(2)
    if weight_map is not None:
        if weight_map.shape[-2:] != student.shape[-2:]:
            weight_map = F.interpolate(weight_map, size=student.shape[-2:], mode="nearest")
        sq = sq * weight_map
        return sq.sum() / weight_map.sum().clamp_min(1.0)
    return sq.mean()
