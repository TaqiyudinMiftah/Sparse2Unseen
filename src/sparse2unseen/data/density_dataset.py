from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as TF

from .manifest import Sample


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(img), MEAN, STD)


def _pad_to_minimum(img: Image.Image, density: torch.Tensor | None, size: int):
    w, h = img.size
    pad_w = max(0, size - w)
    pad_h = max(0, size - h)
    padding = (0, 0, pad_w, pad_h)
    if pad_w or pad_h:
        img = TF.pad(img, padding)
        if density is not None:
            density = TF.pad(density, padding)
    return img, density


class DensityDataset(Dataset):
    """Manifest-backed density dataset.

    For labeled samples, returns `(image, density, id)`.
    For unlabeled samples, returns `(weak_image, strong_image, id)` with only
    photometric changes between the two views, keeping pseudo labels spatially aligned.
    """

    def __init__(
        self,
        samples: list[Sample],
        ids: set[str] | None,
        crop_size: int,
        labeled: bool,
        train: bool,
    ) -> None:
        self.samples = [s for s in samples if ids is None or s.id in ids]
        self.crop_size = crop_size
        self.labeled = labeled
        self.train = train
        self.strong_jitter = ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05)
        if not self.samples:
            raise ValueError("Dataset selection is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, sample: Sample):
        img = Image.open(sample.image).convert("RGB")
        density = None
        if self.labeled:
            if not sample.density:
                raise ValueError(f"No density map for labeled sample {sample.id}")
            density_np = np.load(sample.density).astype(np.float32)
            density = torch.from_numpy(density_np).unsqueeze(0)
        return img, density

    def _aligned_train_crop(self, img: Image.Image, density: torch.Tensor | None):
        img, density = _pad_to_minimum(img, density, self.crop_size)
        w, h = img.size
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        img = TF.crop(img, top, left, self.crop_size, self.crop_size)
        if density is not None:
            density = TF.crop(density, top, left, self.crop_size, self.crop_size)
        if random.random() < 0.5:
            img = TF.hflip(img)
            if density is not None:
                density = TF.hflip(density)
        return img, density

    def __getitem__(self, index: int):
        sample = self.samples[index]
        img, density = self._load(sample)
        if self.train:
            img, density = self._aligned_train_crop(img, density)

        if self.labeled:
            return _to_tensor(img), density.float(), sample.id

        weak = _to_tensor(img)
        strong_img = self.strong_jitter(img)
        if random.random() < 0.25:
            strong_img = TF.gaussian_blur(strong_img, kernel_size=[3, 3], sigma=[0.3, 1.5])
        strong = _to_tensor(strong_img)
        return weak, strong, sample.id
