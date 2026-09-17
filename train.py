#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparse2unseen.config import load_config
from sparse2unseen.confidence.region_stability import stability_weights
from sparse2unseen.data.density_dataset import DensityDataset
from sparse2unseen.data.manifest import read_manifest
from sparse2unseen.data.splits import load_split
from sparse2unseen.dg.augment import DomainDiversifier
from sparse2unseen.losses import supervised_density_loss, weighted_consistency_loss
from sparse2unseen.models.density import build_model
from sparse2unseen.ssl.ema import make_teacher, update_ema
from sparse2unseen.utils import ensure_dir, resolve_device, seed_everything


def make_loaders(cfg: dict):
    samples = read_manifest(cfg["data"]["source_train_manifest"])
    split = load_split(cfg["data"]["split"])
    manifest_ids = {s.id for s in samples}
    labeled_ids = set(split["labeled_ids"])
    unlabeled_ids = set(split["unlabeled_ids"])
    missing = (labeled_ids | unlabeled_ids) - manifest_ids
    if missing:
        raise ValueError(f"Split references IDs not present in manifest: {sorted(missing)[:5]}")

    crop = int(cfg["data"].get("crop_size", 320))
    workers = int(cfg["data"].get("num_workers", 4))
    labeled_ds = DensityDataset(samples, labeled_ids, crop, labeled=True, train=True)
    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=int(cfg["train"].get("batch_size", 8)),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=len(labeled_ds) >= int(cfg["train"].get("batch_size", 8)),
    )

    unlabeled_loader = None
    if cfg.get("ssl", {}).get("enabled") and unlabeled_ids:
        unlabeled_ds = DensityDataset(samples, unlabeled_ids, crop, labeled=False, train=True)
        unlabeled_loader = DataLoader(
            unlabeled_ds,
            batch_size=int(cfg["train"].get("unlabeled_batch_size", 8)),
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=len(unlabeled_ds) >= int(cfg["train"].get("unlabeled_batch_size", 8)),
        )
    return labeled_loader, unlabeled_loader, split


def validate_config(cfg: dict) -> dict:
    result = {
        "experiment": cfg["experiment"]["name"],
        "regime": cfg["experiment"]["regime"],
        "source_manifest": cfg["data"]["source_train_manifest"],
        "split": cfg["data"]["split"],
        "targets": cfg["data"].get("target_test_manifests", {}),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 1)))
    print(json.dumps(validate_config(cfg), indent=2))

    if args.dry_run:
        paths = [cfg["data"]["source_train_manifest"], cfg["data"]["split"]]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            print("Dry-run OK: configuration schema parsed. Data not prepared yet:")
            for p in missing:
                print(f"  - {p}")
            return
        labeled_loader, unlabeled_loader, split = make_loaders(cfg)
        print(
            f"Dry-run OK: labeled={split['n_labeled']}, unlabeled={len(split['unlabeled_ids'])}, "
            f"labeled_batches={len(labeled_loader)}, "
            f"unlabeled_batches={len(unlabeled_loader) if unlabeled_loader else 0}"
        )
        return

    labeled_loader, unlabeled_loader, _ = make_loaders(cfg)
    device = resolve_device(str(cfg.get("device", "cuda")))
    model = build_model(cfg.get("model", {})).to(device)
    teacher = make_teacher(model) if unlabeled_loader is not None else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = ensure_dir(cfg["train"]["output_dir"])
    dg = DomainDiversifier(cfg.get("dg", {})) if cfg.get("dg", {}).get("enabled") else None

    best_loss = float("inf")
    epochs = int(cfg["train"].get("epochs", 180))
    warmup = int(cfg.get("ssl", {}).get("warmup_epochs", 0))
    unsup_weight = float(cfg.get("ssl", {}).get("unsup_weight", 1.0))
    ema_decay = float(cfg.get("ssl", {}).get("ema_decay", 0.999))

    for epoch in range(epochs):
        model.train()
        running = 0.0
        unlabeled_iter = itertools.cycle(unlabeled_loader) if unlabeled_loader is not None else None
        progress = tqdm(labeled_loader, desc=f"epoch {epoch+1}/{epochs}")
        for images, density, _ in progress:
            images = images.to(device, non_blocking=True)
            density = density.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(images)
                loss_sup = supervised_density_loss(pred, density)
                loss_unsup = torch.zeros((), device=device)

                if unlabeled_iter is not None and epoch >= warmup:
                    weak, strong, _ = next(unlabeled_iter)
                    weak = weak.to(device, non_blocking=True)
                    strong = strong.to(device, non_blocking=True)
                    with torch.no_grad():
                        teacher_pred = teacher(weak)

                    student_pred = model(strong)
                    weight_map = None
                    if dg is not None:
                        k = int(cfg["dg"].get("teacher_views", 4))
                        views = []
                        with torch.no_grad():
                            for _ in range(k):
                                views.append(teacher(dg(weak)))
                        stacked = torch.stack(views, dim=0)
                        grid = tuple(int(x) for x in cfg["dg"].get("region_grid", [4, 4]))
                        beta = float(cfg["dg"].get("stability_beta", 4.0))
                        weight_map = stability_weights(stacked, grid=grid, beta=beta)
                    loss_unsup = weighted_consistency_loss(student_pred, teacher_pred, weight_map)

                ramp = min(1.0, max(0.0, (epoch + 1 - warmup) / 10.0))
                loss = loss_sup + unsup_weight * ramp * loss_unsup

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if teacher is not None:
                update_ema(model, teacher, ema_decay)
            running += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.5f}")

        epoch_loss = running / max(1, len(labeled_loader))
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "teacher": teacher.state_dict() if teacher is not None else None,
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "train_loss": epoch_loss,
        }
        torch.save(state, output_dir / "last.pt")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(state, output_dir / "best.pt")


if __name__ == "__main__":
    main()
