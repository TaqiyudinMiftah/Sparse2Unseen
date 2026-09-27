#!/usr/bin/env python3
"""Strictly verify the official MPCount B-source checkpoint before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data/checkpoints/mpcount/stb_original.pth"
OFFICIAL_STB_SHA256 = "9f2e73c62cd289d4a63786663e8fd1ae0833c365b6f84714c5541f8c6fc92b4e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_compatible(model_state: Mapping, checkpoint_state: Mapping) -> None:
    missing = sorted(model_state.keys() - checkpoint_state.keys())
    unexpected = sorted(checkpoint_state.keys() - model_state.keys())
    incompatible = sorted(
        key
        for key in model_state.keys() & checkpoint_state.keys()
        if model_state[key].shape != checkpoint_state[key].shape
        or model_state[key].dtype != checkpoint_state[key].dtype
    )
    if missing or unexpected or incompatible:
        raise ValueError(
            "Checkpoint does not match original MPCount B model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape/dtype mismatch={incompatible[:5]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-sha256", default=OFFICIAL_STB_SHA256)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    digest = sha256(checkpoint)
    if digest != args.expected_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 differs from pinned official B weight: {digest}"
        )

    sys.path.insert(0, str(PROJECT_ROOT / "external/MPCount"))
    from models.models import DGModel_final  # noqa: E402

    model = DGModel_final(pretrained=False, deterministic=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("Expected a plain MPCount state_dict checkpoint")
    assert_compatible(model.state_dict(), state)
    model.load_state_dict(state, strict=True)
    print(f"Verified original MPCount B checkpoint: {checkpoint}")
    print(f"SHA-256: {digest}; state entries: {len(state)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
