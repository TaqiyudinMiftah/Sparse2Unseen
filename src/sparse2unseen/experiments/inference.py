from __future__ import annotations

import math
import sys

import torch

from .common import ROOT


def upstream():
    root = str(ROOT / "external/MPCount")
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.models import DGModel_base, DGModel_final
    from datasets.den_cls_dataset import DenClsDataset
    from utils.misc import seed_everything, seed_worker, divide_img_into_patches
    return DGModel_base, DGModel_final, DenClsDataset, seed_everything, seed_worker, divide_img_into_patches


def density_prediction(model, images):
    output = model(images)
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def predict_count(model, image, patch_size: int, log_para: float) -> float:
    divide = upstream()[-1]
    patches = divide(image, patch_size)[0] if max(image.shape[-2:]) >= patch_size else [image]
    return sum(float(density_prediction(model, patch).sum().cpu()) / log_para for patch in patches)


@torch.no_grad()
def count_metrics(model, loader, device, patch_size: int, log_para: float) -> dict:
    model.eval()
    absolute, squared, records = 0.0, 0.0, []
    for image, _, points, names, _ in loader:
        predicted = predict_count(model, image.to(device), patch_size, log_para)
        count = points.shape[1]
        error = predicted - count
        absolute += abs(error)
        squared += error**2
        records.append({"id": names[0], "predicted": predicted, "count": count})
    if not records:
        raise ValueError("Counting evaluation partition is empty")
    n = len(records)
    return {"mae": absolute/n, "mse": squared/n, "rmse": math.sqrt(squared/n), "n": n, "predictions": records}
