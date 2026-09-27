"""Read completed source-validation epochs from MPCount's training log."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


TRAIN = re.compile(r"^Epoch (\d+): Training loss: ([\d.eE+-]+) Version: .+$")
VALIDATION = re.compile(
    r"^Epoch (\d+): Val criterion: ([\d.eE+-]+) mse: ([\d.eE+-]+) best: .+$"
)
ACCUMULATION = re.compile(
    r"^Epoch (\d+): accumulation=(\d+), microbatches=(\d+), optimizer steps=(\d+)$"
)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    last_batch_loss: float
    source_val_mae: float
    source_val_mse: float
    accumulation_steps: int
    microbatches: int
    optimizer_steps: int

    def wandb_values(self) -> dict[str, float | int]:
        return {
            "epoch": self.epoch,
            "train/last_batch_loss": self.last_batch_loss,
            "source_val/mae": self.source_val_mae,
            "source_val/mse": self.source_val_mse,
            "source_val/rmse": math.sqrt(self.source_val_mse),
            "optimizer/accumulation_steps": self.accumulation_steps,
            "optimizer/microbatches_per_epoch": self.microbatches,
            "optimizer/updates_per_epoch": self.optimizer_steps,
        }


def latest_training_section(log_text: str) -> str:
    """Discard older runs if MPCount appended to an existing log file."""
    lines = log_text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("Start training at ")]
    return "\n".join(lines[starts[-1] :]) if starts else log_text


def parse_completed_epochs(log_text: str) -> list[EpochMetrics]:
    """Ignore unfinished epochs and any previous run appended to the same log."""
    lines = latest_training_section(log_text).splitlines()

    train: dict[int, float] = {}
    validation: dict[int, tuple[float, float]] = {}
    accumulation: dict[int, tuple[int, int, int]] = {}
    for line in lines:
        if match := TRAIN.match(line):
            train[int(match.group(1))] = float(match.group(2))
        elif match := VALIDATION.match(line):
            validation[int(match.group(1))] = (
                float(match.group(2)), float(match.group(3))
            )
        elif match := ACCUMULATION.match(line):
            accumulation[int(match.group(1))] = tuple(
                int(match.group(i)) for i in (2, 3, 4)
            )

    complete = sorted(train.keys() & validation.keys() & accumulation.keys())
    records = []
    for epoch in complete:
        mae, mse = validation[epoch]
        steps, microbatches, updates = accumulation[epoch]
        values = (train[epoch], mae, mse)
        if not all(math.isfinite(value) for value in values) or mse < 0:
            raise ValueError(f"Non-finite or negative metric in epoch {epoch}")
        if steps < 1 or microbatches < 1 or updates < 1:
            raise ValueError(f"Invalid accumulation record in epoch {epoch}")
        records.append(
            EpochMetrics(epoch, train[epoch], mae, mse, steps, microbatches, updates)
        )
    return records
