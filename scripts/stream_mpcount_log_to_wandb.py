#!/usr/bin/env python3
"""Stream an existing MPCount source-training log to a separate W&B run.

This companion does not modify the training process, read target data, or
upload datasets/checkpoints. It can replay completed epochs, then follow a
live training log. Authenticate using W&B's normal login, not a CLI key.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sparse2unseen.monitoring.mpcount_log import (  # noqa: E402
    latest_training_section,
    parse_completed_epochs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "Sparse2Unseen"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    log_path = run_dir / "log.txt"
    if not log_path.is_file():
        raise FileNotFoundError(log_path)

    import wandb

    with wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_dir.name,
        job_type="training-log-companion",
        tags=["MPCount", "STB", "source-only", "log-companion"],
        config={
            "source_domain": "STB",
            "training_run": run_dir.name,
            "metric_source": "MPCount source-training log",
            "target_data_used": False,
        },
        dir=str(run_dir),
    ) as run:
        print(f"W&B run: {run.url}", flush=True)
        last_epoch = -1
        while True:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            for record in parse_completed_epochs(log_text):
                if record.epoch <= last_epoch:
                    continue
                run.log(record.wandb_values(), step=record.epoch, commit=True)
                last_epoch = record.epoch
                best = run.summary.get("best_source_val_mae")
                if best is None or record.source_val_mae < best:
                    run.summary["best_source_val_mae"] = record.source_val_mae
                    run.summary["best_source_val_epoch"] = record.epoch
                print(
                    f"Uploaded epoch {record.epoch}: source val MAE "
                    f"{record.source_val_mae:.4f}",
                    flush=True,
                )
            if not args.follow or "End training at " in latest_training_section(log_text):
                break
            time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
