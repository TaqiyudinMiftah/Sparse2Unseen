#!/usr/bin/env python3
"""Train upstream MPCount final mode with project-owned gradient accumulation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
import shutil
import sys
from pathlib import Path

import torch.nn.functional as F
import yaml

from sparse2unseen.monitoring.mpcount_log import parse_completed_epochs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPCOUNT_ROOT = PROJECT_ROOT / "external/MPCount"
sys.path.insert(0, str(MPCOUNT_ROOT))

from main import load_config  # noqa: E402
from trainers.dgtrainer import DGTrainer  # noqa: E402


class AccumDGTrainer(DGTrainer):
    """Preserve MPCount's final-mode loss, validation, and epoch scheduler."""

    def __init__(self, *args, accumulation_steps: int, wandb_run=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode != "final":
            raise ValueError("Accumulation adapter supports MPCount final mode only")
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        self.accumulation_steps = accumulation_steps
        self.micro_batches = 0
        self.micro_index = 0
        self.optimizer_steps = 0
        self.train_loss_sum = 0.0
        self.wandb_run = wandb_run

    def configure_epoch(self, micro_batches: int) -> None:
        if micro_batches < 1:
            raise ValueError("Training loader must contain at least one minibatch")
        self.micro_batches = micro_batches
        self.micro_index = 0
        self.optimizer_steps = 0
        self.train_loss_sum = 0.0

    def train_epoch(self, model, loss, train_dataloader, val_dataloader, optimizer, scheduler, epoch, best_criterion, best_epoch):
        self.configure_epoch(len(train_dataloader))
        result = super().train_epoch(
            model, loss, train_dataloader, val_dataloader, optimizer,
            scheduler, epoch, best_criterion, best_epoch
        )
        expected = math.ceil(self.micro_batches / self.accumulation_steps)
        if self.micro_index != self.micro_batches or self.optimizer_steps != expected:
            raise RuntimeError("Incomplete gradient-accumulation epoch")
        self.log(
            f"Epoch {epoch}: accumulation={self.accumulation_steps}, "
            f"microbatches={self.micro_batches}, optimizer steps={self.optimizer_steps}"
        )
        if self.wandb_run is not None:
            try:
                log_text = (Path(self.log_dir) / "log.txt").read_text(encoding="utf-8")
                record = parse_completed_epochs(log_text)[-1]
                if record.epoch != epoch:
                    raise RuntimeError("Latest completed log epoch does not match training")
                metrics = record.wandb_values()
                metrics["train/mean_loss"] = self.train_loss_sum / self.micro_batches
                metrics["optimizer/lr"] = optimizer.param_groups[0]["lr"]
                metrics["source_val/best_mae"] = result[0]
                self.wandb_run.log(metrics, step=epoch, commit=True)
                self.wandb_run.summary["best_source_val_mae"] = result[0]
                self.wandb_run.summary["best_source_val_epoch"] = result[1]
            except Exception as error:
                self.log(
                    "W&B logging disabled after a monitoring error "
                    f"({type(error).__name__}); training continues."
                )
                self.wandb_run = None
        return result

    def train_step(self, model, loss, optimizer, batch, epoch):
        if not self.micro_batches or self.micro_index >= self.micro_batches:
            raise RuntimeError("Call configure_epoch before training minibatches")
        index = self.micro_index
        group_start = (index // self.accumulation_steps) * self.accumulation_steps
        group_size = min(self.accumulation_steps, self.micro_batches - group_start)
        if index == group_start:
            optimizer.zero_grad()

        imgs1, imgs2, gt_datas = batch
        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)
        gt_cmaps = gt_datas[-1].to(self.device)
        dmaps1, dmaps2, cmaps1, cmaps2, _, loss_con, _ = model.forward_train(
            imgs1, imgs2, gt_cmaps
        )
        loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(
            loss, dmaps2, gt_datas
        )
        loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(
            cmaps2, gt_cmaps
        )
        loss_total = loss_den + 10 * loss_cls + 10 * loss_con
        (loss_total / group_size).backward()
        self.micro_index += 1
        if self.micro_index == group_start + group_size:
            optimizer.step()
            self.optimizer_steps += 1
        raw_loss = loss_total.detach().item()
        self.train_loss_sum += raw_loss
        return raw_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="Override epochs for a smoke run")
    parser.add_argument("--version", help="Override run name for a smoke run")
    parser.add_argument("--wandb", action="store_true", help="Log source-only metrics to W&B")
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "Sparse2Unseen")
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args()
    config = args.config.expanduser().resolve(strict=True)
    with config.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    accumulation_steps = raw_config["accumulation_steps"]
    if args.epochs is not None and (
        not args.version or args.version == raw_config["version"]
    ):
        raise ValueError("--epochs requires a distinct --version for smoke runs")
    version = args.version or raw_config["version"]
    run_dir = MPCOUNT_ROOT / "logs" / version
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse MPCount run directory: {run_dir}")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")

    os.chdir(MPCOUNT_ROOT)
    init_params, task_params = load_config(str(config), "train")
    init_params["version"] = version
    if args.epochs is not None:
        task_params["num_epochs"] = args.epochs
    if args.wandb:
        try:
            import wandb
        except ImportError as error:
            raise SystemExit("Install monitoring with `uv sync --extra wandb` first") from error
        run_context = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=version,
            job_type="train",
            tags=["MPCount", "STB", "source-only", "effective-batch-16"],
            config={
                "seed": raw_config["seed"],
                "epochs": task_params["num_epochs"],
                "physical_batch_size": raw_config["train_loader"]["batch_size"],
                "accumulation_steps": accumulation_steps,
                "model_deterministic": raw_config["model"]["params"]["deterministic"],
                "source_domain": "STB",
                "target_data_used": False,
            },
            dir=str(MPCOUNT_ROOT),
        )
    else:
        run_context = nullcontext(None)
    with run_context as wandb_run:
        trainer = AccumDGTrainer(
            **init_params, accumulation_steps=accumulation_steps, wandb_run=wandb_run
        )
        shutil.copy2(config, trainer.log_dir)
        trainer.train(**task_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
