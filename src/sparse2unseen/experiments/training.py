from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sparse2unseen.confidence.region_stability import stability_weights
from sparse2unseen.data.density_dataset import DensityDataset
from sparse2unseen.data.manifest import Sample
from sparse2unseen.dg.augment import DomainDiversifier
from sparse2unseen.losses import weighted_consistency_loss
from sparse2unseen.ssl.ema import make_teacher, update_ema
from .common import ROOT, SSL_METHODS, DG_METHODS, SOURCE_COUNTS, check_source_inputs, digest, load_json, now, write_json
from .inference import upstream, density_prediction, count_metrics
from .sampling import BalancedEpochSampler


def atomic_save(value, path: Path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def make_model(method: str, pretrained: bool):
    base, final, *_ = upstream()
    cls = final if method in DG_METHODS else base
    return cls(pretrained=pretrained, deterministic=True)


def make_loaders(spec: dict):
    source = check_source_inputs(spec)
    settings = spec["settings"]
    _, _, dataset_type, _, seed_worker, _ = upstream()
    params = dict(root=str(ROOT / spec["labeled_root"]), crop_size=settings["crop_size"],
                  downsample=1, is_grey=False, unit_size=16, pre_resize=1)
    labeled = dataset_type(method="train", **params)
    validation = dataset_type(method="val", **params)
    if len(labeled) != spec["labeled_images"] or len(validation) != SOURCE_COUNTS[spec["source"]][1]:
        raise ValueError("Upstream dataset sizes do not match the frozen source partitions")
    generators = {name: torch.Generator().manual_seed(spec["seed"] + offset)
                  for offset, name in enumerate(("l_sampler", "l_workers", "u_sampler", "u_workers", "val_workers"))}
    train_loader = DataLoader(
        labeled, batch_size=settings["physical_batch_size"],
        sampler=BalancedEpochSampler(len(labeled), spec["samples_per_epoch"], generators["l_sampler"]),
        collate_fn=dataset_type.collate, num_workers=settings["train_workers"],
        worker_init_fn=seed_worker, generator=generators["l_workers"], pin_memory=True,
    )
    val_loader = DataLoader(validation, batch_size=1, num_workers=settings["validation_workers"],
                            worker_init_fn=seed_worker, generator=generators["val_workers"])
    unlabeled_loader = None
    if spec["method"] in SSL_METHODS:
        if not spec["unlabeled_images"]:
            raise ValueError("SSL experiments require an unlabeled source partition")
        # Annotation paths and counts are discarded before dataset construction.
        images_only = [Sample(row["id"], row["image"], None, None, 0.0) for row in source["pool"]]
        unlabeled = DensityDataset(images_only, set(source["split"]["unlabeled_ids"]),
                                    settings["crop_size"], labeled=False, train=True)
        unlabeled_loader = DataLoader(
            unlabeled, batch_size=settings["physical_batch_size"],
            sampler=BalancedEpochSampler(len(unlabeled), spec["samples_per_epoch"], generators["u_sampler"]),
            num_workers=settings["unlabeled_workers"], worker_init_fn=seed_worker,
            generator=generators["u_workers"], pin_memory=True,
        )
    return train_loader, unlabeled_loader, val_loader, generators


def supervised_loss(model, method: str, batch, device, log_para: float):
    weak, strong, (_, density, regions) = batch
    weak, strong = weak.to(device), strong.to(device)
    density = density.to(device) * log_para
    if method in DG_METHODS:
        regions = regions.to(device)
        p1, p2, c1, c2, _, consistency, _ = model.forward_train(weak, strong, regions)
        return (F.mse_loss(p1, density) + F.mse_loss(p2, density)
                + 10 * (F.binary_cross_entropy(c1, regions) + F.binary_cross_entropy(c2, regions))
                + 10 * consistency)
    return F.mse_loss(model(weak), density) + F.mse_loss(model(strong), density)


def ema_decay_for_step(maximum: float, completed_updates: int) -> float:
    """Mean Teacher's original startup correction, with a fixed decay ceiling."""
    if completed_updates < 1 or not 0 <= maximum < 1:
        raise ValueError("EMA requires a positive completed step and decay in [0,1)")
    return min(maximum, 1.0 - 1.0/(completed_updates + 1))


def train_epoch(model, teacher, optimizer, labeled_loader, unlabeled_loader, spec, epoch: int, device):
    settings = spec["settings"]
    model.train()
    if teacher is not None:
        teacher.eval()
    active = teacher is not None and epoch >= settings["warmup_epochs"]
    unlabeled = iter(unlabeled_loader) if active else None
    diversifier = DomainDiversifier({}) if spec["method"] == "domain_stable" else None
    batches, accumulation = len(labeled_loader), settings["accumulation_steps"]
    supervised_total, unsupervised_total, weight_total = 0.0, 0.0, 0.0
    updates, labeled_samples, unlabeled_samples = 0, 0, 0
    for index, batch in enumerate(labeled_loader):
        group_start = index // accumulation * accumulation
        group_size = min(accumulation, batches - group_start)
        if index == group_start:
            optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(model, spec["method"], batch, device, settings["log_para"])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite supervised loss")
        supervised_total += float(loss.detach())
        labeled_samples += batch[0].shape[0]
        (loss / group_size).backward()
        # Free the supervised graph before the SSL pass to bound peak memory.
        del loss
        if active:
            weak, strong, _ = next(unlabeled)
            weak, strong = weak.to(device), strong.to(device)
            weights = None
            with torch.no_grad():
                pseudo = density_prediction(teacher, weak).clamp_min(0)
                if diversifier is not None:
                    views = torch.stack([
                        density_prediction(teacher, diversifier(weak)) / settings["log_para"]
                        for _ in range(settings["teacher_views"])
                    ])
                    weights = stability_weights(views, tuple(settings["region_grid"]), settings["stability_beta"])
            prediction = density_prediction(model, strong)
            ssl_loss = weighted_consistency_loss(prediction, pseudo, weights)
            if not torch.isfinite(ssl_loss):
                raise RuntimeError("Non-finite pseudo-label loss")
            ramp = min(1.0, (epoch + 1 - settings["warmup_epochs"]) / settings["ramp_epochs"])
            (settings["unsupervised_weight"] * ramp * ssl_loss / group_size).backward()
            unsupervised_total += float(ssl_loss.detach())
            weight_total += float(weights.mean()) if weights is not None else 1.0
            unlabeled_samples += weak.shape[0]
            del prediction, ssl_loss, pseudo, weights
        if index + 1 == group_start + group_size:
            optimizer.step()
            updates += 1
            if teacher is not None:
                total_updates = epoch * spec["updates_per_epoch"] + updates
                decay = (ema_decay_for_step(settings["ema_decay"], total_updates)
                         if settings.get("ema_startup_correction", True) else settings["ema_decay"])
                update_ema(model, teacher, decay)
    if updates != spec["updates_per_epoch"] or labeled_samples != spec["samples_per_epoch"]:
        raise RuntimeError("Training did not meet the frozen optimizer/exposure budget")
    if active and unlabeled_samples != spec["samples_per_epoch"]:
        raise RuntimeError("SSL did not meet its explicit source exposure budget")
    return {"supervised_loss": supervised_total/batches, "unsupervised_loss": unsupervised_total/batches,
            "stability_weight_mean": weight_total/batches if active else None,
            "optimizer_updates": updates, "labeled_samples": labeled_samples,
            "unlabeled_samples": unlabeled_samples}


def capture_rng(generators):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": {key: value.get_state() for key, value in generators.items()}}


def restore_rng(state, generators):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    for key, value in generators.items():
        value.set_state(state["generators"][key].cpu())


def cpu_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train(spec_path: Path, *, resume=False, smoke_epochs: int | None = None, smoke_name=None, wandb_enabled=False):
    spec = load_json(spec_path)
    check_source_inputs(spec)
    total_epochs = spec["settings"]["epochs"]
    name = spec["name"]
    if smoke_epochs is not None:
        if not 1 <= smoke_epochs <= 2:
            raise ValueError("Smoke runs must use one or two epochs")
        name += "_smoke"
        if smoke_name:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", smoke_name):
                raise ValueError("Smoke name must be a single safe path component")
            name += "_" + smoke_name
        total_epochs = smoke_epochs
        spec = copy.deepcopy(spec)
        spec["settings"]["warmup_epochs"] = 0  # Exercise SSL in isolated smoke runs.
    elif smoke_name:
        raise ValueError("A smoke name requires --smoke-epochs")
    output = ROOT / "runs/benchmark_v1/jobs" / name
    completed = output / "training_complete.json"
    if completed.exists():
        if resume:
            marker = load_json(completed)
            if digest(output / "best.pt") != marker["checkpoint_sha256"]:
                raise RuntimeError("Completed checkpoint hash differs")
            return marker
        raise FileExistsError(f"Training is already complete: {name}")
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to reuse a training run: {name}")
    if output.exists() and resume:
        if not (output / "spec.json").exists() or load_json(output / "spec.json") != spec:
            raise RuntimeError("Existing run belongs to a different frozen recipe")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "spec.json", spec)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for production benchmark training")
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    upstream()[3](spec["seed"])
    labeled, unlabeled, validation, generators = make_loaders(spec)
    resume_path = output / "resume.pt"
    restoring = resume and resume_path.is_file()
    model = make_model(spec["method"], pretrained=not restoring).to(device)
    teacher = make_teacher(model) if spec["method"] in SSL_METHODS else None
    settings = spec["settings"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    # Preserve MPCount's published config and its upstream once-per-epoch step.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=settings["learning_rate"], epochs=300,
        steps_per_epoch=spec["updates_per_epoch"], final_div_factor=1000,
    )
    best_mae, best_epoch, best_weights = float("inf"), -1, None
    start_epoch, loaded = 0, None
    if restoring:
        loaded = torch.load(resume_path, map_location="cpu")
        if loaded["spec_hash"] != digest(spec_path) or loaded["smoke"] != (smoke_epochs is not None):
            raise RuntimeError("Resume recipe does not match the frozen experiment")
        model.load_state_dict(loaded["model"], strict=True)
        if teacher is not None:
            teacher.load_state_dict(loaded["teacher"], strict=True)
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        start_epoch = loaded["epoch"] + 1
        best_mae, best_epoch, best_weights = loaded["best_mae"], loaded["best_epoch"], loaded["best_weights"]
        if best_weights is not None:
            atomic_save({"model": best_weights, "epoch": best_epoch, "source_val_mae": best_mae,
                         "model_kind": "ema" if teacher is not None else "student"}, output / "best.pt")
    run = None
    if wandb_enabled:
        import wandb
        previous = load_json(output / "wandb.json") if (output / "wandb.json").exists() else {}
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "Sparse2Unseen"),
                         entity=os.environ.get("WANDB_ENTITY"), name=name, id=previous.get("id"),
                         resume="allow", job_type="train", dir=str(output),
                         config={"source": spec["source"], "method": spec["method"], "fraction": spec["fraction"],
                                 "seed": spec["seed"], "target_data_used": False,
                                 "labeled_images": spec["labeled_images"], "source_val_images": len(validation.dataset),
                                 "samples_per_epoch": spec["samples_per_epoch"], "settings": settings},
                         tags=["benchmark-v1", spec["method"], spec["source"], "source-only"])
        write_json(output / "wandb.json", {"id": run.id, "url": run.url})
    if loaded is not None:
        restore_rng(loaded["rng"], generators)
        del loaded

    def snapshot(epoch):
        return {"model": model.state_dict(), "teacher": teacher.state_dict() if teacher is not None else None,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best_mae": best_mae, "best_epoch": best_epoch, "best_weights": best_weights,
                "rng": capture_rng(generators), "spec_hash": digest(spec_path), "smoke": smoke_epochs is not None}

    if start_epoch == 0 and not restoring:
        atomic_save(snapshot(-1), resume_path)
    try:
        for epoch in range(start_epoch, total_epochs):
            started = time.monotonic()
            metrics = train_epoch(model, teacher, optimizer, labeled, unlabeled, spec, epoch, device)
            scheduler.step()
            evaluation_model = teacher if teacher is not None else model
            val = count_metrics(evaluation_model, validation, device,
                                settings["validation_patch_sizes"][spec["source"]], settings["log_para"])
            if not math.isfinite(val["mae"]):
                raise RuntimeError("Non-finite source validation MAE")
            if val["mae"] < best_mae:
                best_mae, best_epoch, best_weights = val["mae"], epoch, cpu_weights(evaluation_model)
                atomic_save({"model": best_weights, "epoch": epoch, "source_val_mae": best_mae,
                             "model_kind": "ema" if teacher is not None else "student"}, output / "best.pt")
            record = {"epoch": epoch, **metrics, "source_val_mae": val["mae"], "source_val_rmse": val["rmse"],
                      "best_source_val_mae": best_mae, "best_epoch": best_epoch,
                      "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.monotonic()-started,
                      "peak_gpu_mib": torch.cuda.max_memory_allocated()/1024**2, "updated_at": now()}
            write_json(output / "progress.json", record)
            with (output / "epochs.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            print(json.dumps({"experiment": name, **record}), flush=True)
            if run is not None:
                run.log({key: value for key, value in record.items() if isinstance(value, (int, float))}, step=epoch)
                run.summary["best_source_val_epoch"] = best_epoch
                run.summary["best_source_val_mae"] = best_mae
            atomic_save(snapshot(epoch), resume_path)
        marker = {"experiment": name, "completed_at": now(), "epochs": total_epochs,
                  "smoke": smoke_epochs is not None, "selected_epoch": best_epoch, "source_val_mae": best_mae,
                  "selection_domain": f"{spec['source']}_val", "checkpoint_sha256": digest(output / "best.pt"),
                  "spec_sha256": digest(spec_path), "model_kind": "ema" if teacher is not None else "student"}
        write_json(completed, marker)
        # Completed jobs retain selected and final weights; only active jobs need
        # large optimizer/RNG snapshots. This caps storage for the full matrix.
        atomic_save({"model": cpu_weights(model), "epoch": total_epochs-1, "completed": True,
                     "spec_hash": digest(spec_path)}, resume_path)
        if run is not None:
            run.finish()
        return marker
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
