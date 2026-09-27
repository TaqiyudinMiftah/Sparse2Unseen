from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_mpcount_checkpoint.py"
spec = importlib.util.spec_from_file_location("verify_mpcount_checkpoint", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_matching_state_dict_passes() -> None:
    state = {"weight": torch.ones(2, 3), "bias": torch.zeros(2)}
    module.assert_compatible(state, state)


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"weight": torch.ones(2, 3)},
        {"weight": torch.ones(2, 3), "bias": torch.zeros(2), "extra": torch.zeros(1)},
        {"weight": torch.ones(3, 2), "bias": torch.zeros(2)},
        {"weight": torch.ones(2, 3, dtype=torch.float64), "bias": torch.zeros(2)},
    ],
)
def test_mismatched_state_dict_fails(checkpoint: dict) -> None:
    model = {"weight": torch.ones(2, 3), "bias": torch.zeros(2)}
    with pytest.raises(ValueError, match="does not match"):
        module.assert_compatible(model, checkpoint)


@pytest.mark.parametrize(
    ("domain", "patch_size"),
    [("stb", 10000), ("sta", 10000), ("qnrf", 1024)],
)
def test_official_diagnostic_config_uses_original_model(
    domain: str, patch_size: int
) -> None:
    config = yaml.safe_load(
        (
            SCRIPT.parents[1]
            / f"configs/mpcount/stb_official_original_test_{domain}.yml"
        ).read_text()
    )
    assert config["model"]["params"]["pretrained"] is False
    assert config["model"]["params"]["deterministic"] is False
    assert config["test_dataset"]["params"]["root"] == f"data/{domain}"
    assert config["patch_size"] == patch_size
