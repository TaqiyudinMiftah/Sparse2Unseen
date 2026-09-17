from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_datasets.py"
spec = importlib.util.spec_from_file_location("download_datasets", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_dataset_selection() -> None:
    assert [d.key for d in module.selected_specs("shanghaitech")] == ["shanghaitech"]
    assert [d.key for d in module.selected_specs("all")] == ["shanghaitech", "ucf_qnrf"]


def test_zip_validation_and_marker_detection(tmp_path: Path) -> None:
    archive = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("part_A_final/example.txt", "a")
        zf.writestr("part_B_final/example.txt", "b")

    assert module.looks_like_zip(archive)

    extracted = tmp_path / "extracted"
    module.extract_zip(archive, extracted, force=False)
    assert module.find_marker(extracted, "part_A_final")
    assert module.find_marker(extracted, "part_B_final")


def test_human_bytes() -> None:
    assert module.human_bytes(1024) == "1.0 KiB"
    assert module.human_bytes(1024 * 1024) == "1.0 MiB"
