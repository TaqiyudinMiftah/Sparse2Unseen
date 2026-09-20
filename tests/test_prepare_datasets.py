from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_datasets.py"
spec = importlib.util.spec_from_file_location("prepare_datasets", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_raw_dataset_discovery(tmp_path: Path) -> None:
    a = tmp_path / "shanghaitech" / "ShanghaiTech" / "part_A_final"
    b = tmp_path / "shanghaitech" / "ShanghaiTech" / "part_B_final"
    for part in (a, b):
        (part / "train_data").mkdir(parents=True)
        (part / "test_data").mkdir()

    qnrf = tmp_path / "ucf_qnrf" / "nested" / "UCF-QNRF_ECCV18"
    (qnrf / "Train").mkdir(parents=True)
    (qnrf / "Test").mkdir()

    assert module.locate_shanghaitech_part(tmp_path, "part_A") == a.resolve()
    assert module.locate_shanghaitech_part(tmp_path, "part_B") == b.resolve()
    assert module.locate_qnrf_root(tmp_path) == qnrf.resolve()


def test_shanghaitech_mpcount_alias(tmp_path: Path) -> None:
    target = tmp_path / "source" / "part_A_final"
    target.mkdir(parents=True)

    alias = module.ensure_shanghai_alias(
        tmp_path, "part_A", target, dry_run=False
    )

    assert alias.is_symlink()
    assert alias.resolve() == target.resolve()


def test_selected_keys() -> None:
    assert module.selected_keys("all") == ["sta", "stb", "qnrf"]
    assert module.selected_keys("stb") == ["stb"]
