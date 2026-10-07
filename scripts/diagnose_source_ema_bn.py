#!/usr/bin/env python3
"""Observe source-only EMA/BN sensitivity without changing production runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from sparse2unseen.experiments.common import (
    ROOT, SOURCE_COUNTS, SSL_METHODS, check_source_inputs, digest, load_json, now, write_json,
)
from sparse2unseen.experiments.inference import count_metrics, density_prediction, upstream
from sparse2unseen.experiments.training import atomic_save, make_model


def validate_spec(spec):
    if spec["source"] not in SOURCE_COUNTS or spec["method"] not in SSL_METHODS:
        raise ValueError("This diagnostic requires a declared source and SSL model")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", spec["name"]):
        raise ValueError("Unsafe experiment name")
    for phase in ("train", "val"):
        expected = f"data/manifests/{spec['source']}_{phase}.jsonl"
        if spec[f"{phase}_manifest"] != expected:
            raise ValueError("Diagnostic manifests must belong to the declared source")


def validate_snapshot(snapshot, spec_sha, epochs):
    if snapshot["spec_hash"] != spec_sha or snapshot.get("smoke") is not False:
        raise ValueError("Snapshot is not from this frozen production recipe")
    if snapshot.get("teacher") is None or not 0 <= snapshot["epoch"] < epochs:
        raise ValueError("An active, completed-epoch EMA snapshot is required")


def source_image_batches(rows, source, count, batch_size, crop_size):
    if count <= 0 or count > len(rows) or count % batch_size:
        raise ValueError("Calibration count must be positive, in-pool, and a whole batch")
    selected = sorted(rows, key=lambda row: row["id"])[:count]
    expected_parent = (ROOT / f"data/processed/mpcount/{source}/train").resolve()
    for offset in range(0, count, batch_size):
        images = []
        for row in selected[offset:offset + batch_size]:
            path = Path(row["image"])
            if path.resolve().parent != expected_parent:
                raise ValueError("BN calibration may only open source training images")
            # Never read points, density, or count fields, including for unlabeled images.
            with Image.open(path) as original:
                crop = TF.center_crop(original.convert("RGB"), [crop_size, crop_size])
                images.append(TF.normalize(TF.to_tensor(crop), [.5] * 3, [.5] * 3))
        yield torch.stack(images)


@torch.no_grad()
def recalibrate_batch_norm(model, batches):
    layers = [module for module in model.modules()
              if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not layers:
        raise ValueError("The diagnostic model has no batch normalization")
    model.eval()  # Keep dropout and all non-BN modules deterministic.
    momenta = [module.momentum for module in layers]
    for module in layers:
        module.reset_running_stats()
        module.momentum = None  # Cumulative average over equal-sized source batches.
        module.train()
    samples, updates = 0, 0
    try:
        for images in batches:
            density_prediction(model, images)
            samples += len(images)
            updates += 1
            print({"phase": "source_train_bn_calibration", "images": samples}, flush=True)
        if not updates:
            raise ValueError("BN calibration received no source images")
    finally:
        model.eval()
        for module, momentum in zip(layers, momenta):
            module.momentum = momentum
    return {"images": samples, "batches": updates, "bn_layers": len(layers)}


def evaluate_source(model, loader, spec, phase):
    def observed_batches():
        for index, batch in enumerate(loader, 1):
            yield batch
            if index % 5 == 0:
                print({"phase": phase, "source_val_images": index}, flush=True)
    result = count_metrics(model, observed_batches(), torch.device("cpu"),
                           spec["settings"]["validation_patch_sizes"][spec["source"]],
                           spec["settings"]["log_para"])
    if result["n"] != SOURCE_COUNTS[spec["source"]][1]:
        raise ValueError("Diagnostic did not cover the full source validation partition")
    print({"phase": phase, **{key: value for key, value in result.items()
                              if key != "predictions"}}, flush=True)
    return result


def diagnose(spec_path, calibration_images=32, threads=2):
    if not 1 <= threads <= 4:
        raise ValueError("Use one to four CPU threads; diagnostics never allocate a GPU")
    torch.set_num_threads(threads)
    spec = load_json(spec_path)
    validate_spec(spec)
    source = check_source_inputs(spec)
    batch_size = spec["settings"]["physical_batch_size"]
    if calibration_images <= 0 or calibration_images > len(source["pool"]) or calibration_images % batch_size:
        raise ValueError("Calibration count must be positive, in-pool, and a whole batch")
    run_root = ROOT / "runs/benchmark_v1/jobs" / spec["name"]
    if load_json(run_root / "spec.json") != spec:
        raise ValueError("Live run does not belong to the requested frozen recipe")
    spec_sha = digest(spec_path)
    snapshot = torch.load(run_root / "resume.pt", map_location="cpu")
    validate_snapshot(snapshot, spec_sha, spec["settings"]["epochs"])
    model = make_model(spec["method"], pretrained=False).eval()
    model.load_state_dict(snapshot["teacher"], strict=True)
    model.requires_grad_(False)
    original_weights = snapshot["teacher"]
    copied_buffers = all(torch.equal(value, snapshot["model"][key])
                         for key, value in model.named_buffers())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "runs/benchmark_v1/diagnostics" / spec["name"] / f"ema_bn_epoch{snapshot['epoch']}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "input_teacher.pt"
    atomic_save({"model": original_weights, "epoch": snapshot["epoch"],
                 "spec_sha256": spec_sha, "diagnostic_only": True}, checkpoint)
    provenance = {
        "kind": "source_only_bn_diagnostic", "experiment": spec["name"], "source": spec["source"],
        "created_at": now(), "snapshot_epoch": snapshot["epoch"], "spec_sha256": spec_sha,
        "checkpoint_sha256": digest(checkpoint), "diagnostic_code_sha256": digest(Path(__file__)),
        "train_manifest_sha256": digest(ROOT / spec["train_manifest"]),
        "val_manifest_sha256": digest(ROOT / spec["val_manifest"]),
        "calibration_ids": [row["id"] for row in sorted(source["pool"], key=lambda row: row["id"])[:calibration_images]],
        "calibration_batch_size": batch_size, "calibration_crop_size": spec["settings"]["crop_size"],
        "calibration_transform": "deterministic center crop; native 0.5 normalization; no dropout",
        "teacher_buffers_equal_student_at_snapshot": copied_buffers,
        "target_data_used": False, "excluded_from_primary_results": True,
    }
    write_json(output / "source_spec.json", spec)
    write_json(output / "provenance.json", provenance)
    del snapshot
    print({"output": str(output), **provenance}, flush=True)
    dataset_type = upstream()[2]
    validation = dataset_type(method="val", root=str(ROOT / spec["labeled_root"]),
                              crop_size=spec["settings"]["crop_size"], downsample=1,
                              is_grey=False, unit_size=16, pre_resize=1)
    loader = DataLoader(validation, batch_size=1, num_workers=0)
    original = evaluate_source(model, loader, spec, "original_ema_source_val")
    write_json(output / "original_source_val.json", original)
    calibration = recalibrate_batch_norm(model, source_image_batches(
        source["pool"], spec["source"], calibration_images, batch_size, spec["settings"]["crop_size"]))
    if not all(torch.equal(value, original_weights[key]) for key, value in model.named_parameters()):
        raise RuntimeError("Diagnostic unexpectedly modified model parameters")
    recalibrated = evaluate_source(model, loader, spec, "scratch_recalibrated_ema_source_val")
    result = {**provenance, "completed_at": now(), "calibration": calibration,
              "parameters_unchanged": True, "live_models_and_checkpoints_unchanged": True,
              "original": original, "scratch_recalibrated": recalibrated}
    write_json(output / "diagnostic.json", result)
    print({"completed_diagnostic": str(output / "diagnostic.json")}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    diagnose(args.spec, args.calibration_images, args.threads)
