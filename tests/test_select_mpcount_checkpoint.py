from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_mpcount_checkpoint.py"
spec = importlib.util.spec_from_file_location("select_mpcount_checkpoint", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_uses_completed_source_validation_selection(tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text(
        "Start training at 2026-09-26\n"
        "Epoch 5: Val criterion: 42.0\n"
        "Best epoch: 5, best criterion: 42.0\n"
        "End training at 2026-09-26\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best_5.pth"
    checkpoint.touch()

    assert module.select_checkpoint(tmp_path) == checkpoint


def test_rejects_incomplete_training(tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text(
        "Start training at 2026-09-26\n"
        "Best epoch: 5, best criterion: 42.0\n",
        encoding="utf-8",
    )
    (tmp_path / "best_5.pth").touch()

    with pytest.raises(RuntimeError, match="has not completed"):
        module.select_checkpoint(tmp_path)


def test_rejects_previous_checkpoint_during_new_training(tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text(
        "Start training at first run\n"
        "Best epoch: 5, best criterion: 42.0\n"
        "End training at first run\n"
        "Start training at second run\n",
        encoding="utf-8",
    )
    (tmp_path / "best_5.pth").touch()

    with pytest.raises(RuntimeError, match="No source-validation best epoch"):
        module.select_checkpoint(tmp_path)
