#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sparse2unseen.experiments.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a frozen source-only experiment")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--smoke-name")
    args = parser.parse_args()
    train(args.spec, resume=args.resume, smoke_epochs=args.smoke_epochs,
          smoke_name=args.smoke_name, wandb_enabled=args.wandb)
