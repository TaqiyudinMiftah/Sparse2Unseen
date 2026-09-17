#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from sparse2unseen.config import load_config
from sparse2unseen.data.manifest import read_manifest
from sparse2unseen.metrics import CountMetrics
from sparse2unseen.models.density import build_model
from sparse2unseen.utils import resolve_device


def image_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = TF.to_tensor(img)
    x = TF.normalize(x, [0.5] * 3, [0.5] * 3)
    return x.unsqueeze(0).to(device)


@torch.no_grad()
def evaluate_manifest(model, manifest: str, device: torch.device) -> dict[str, float]:
    samples = read_manifest(manifest)
    metrics = CountMetrics()
    model.eval()
    for sample in samples:
        pred = model(image_tensor(sample.image, device))
        metrics.update(float(pred.sum().cpu()), sample.count)
    result = metrics.compute()
    result["n"] = len(samples)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(str(cfg.get("device", "cuda")))
    model = build_model(cfg.get("model", {})).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    weights = state.get("teacher") or state["model"]
    model.load_state_dict(weights)

    manifests = {"source": cfg["data"]["source_test_manifest"]}
    manifests.update(cfg["data"].get("target_test_manifests", {}))
    results = {name: evaluate_manifest(model, path, device) for name, path in manifests.items()}
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
