from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("evaluate_sparse_mpcount", SCRIPTS / "evaluate_sparse_mpcount.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("domain,patch_size", [("stb", 10000), ("sta", 10000), ("qnrf", 1024)])
def test_sparse_eval_config_uses_fixed_domain_protocol(tmp_path, seed, domain, patch_size):
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"weights")
    config = module.build_eval_config(seed, domain, checkpoint)
    assert config["seed"] == seed
    assert config["version"].startswith(f"stb_10_seed{seed}_effbs16_test_{domain}")
    assert config["checkpoint"] == str(checkpoint)
    assert config["test_dataset"]["params"]["root"] == f"data/{domain}"
    assert config["patch_size"] == patch_size
    assert config["model"]["params"]["deterministic"] is True


def test_rejects_unregistered_sparse_seed(tmp_path):
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"weights")
    with pytest.raises(ValueError, match="seed"):
        module.build_eval_config(4, "stb", checkpoint)


@pytest.mark.parametrize("domain,patch_size", [("stb", 10000), ("sta", 10000), ("qnrf", 1024)])
def test_clean_retry_keeps_inference_protocol_and_separate_logs(tmp_path, domain, patch_size):
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"weights")
    version = "stb_10_seed3_effbs16_retry1"
    config = module.build_eval_config(3, domain, checkpoint, version)
    assert config["version"].startswith(f"{version}_test_{domain}")
    assert config["patch_size"] == patch_size
    assert config["test_dataset"]["params"]["root"] == f"data/{domain}"
    assert config["model"]["params"]["deterministic"] is True


@pytest.mark.parametrize("version", ["../escape", "stb_10_seed1_effbs16_retry1", "stb_10_seed3_effbs160"])
def test_retry_version_must_match_seed_without_path_traversal(version):
    with pytest.raises(ValueError):
        module.validate_run_version(3, version)
