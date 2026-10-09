from __future__ import annotations

import torch
from torch.utils.data import Sampler


class BalancedEpochSampler(Sampler):
    """Exact exposure budget, with per-image visit counts differing by at most one."""

    def __init__(self, n_images: int, samples: int, generator: torch.Generator):
        if n_images < 1 or samples < n_images:
            raise ValueError("Exposure budget must cover every source image")
        self.n_images, self.samples, self.generator = n_images, samples, generator

    def __len__(self):
        return self.samples

    def __iter__(self):
        repetitions, remainder = divmod(self.samples, self.n_images)
        indices = torch.arange(self.n_images).repeat(repetitions)
        if remainder:
            extras = torch.randperm(self.n_images, generator=self.generator)[:remainder]
            indices = torch.cat([indices, extras])
        ordering = torch.randperm(len(indices), generator=self.generator)
        return iter(indices[ordering].tolist())
