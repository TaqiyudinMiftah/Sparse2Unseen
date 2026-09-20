from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_datasets.py"
spec = importlib.util.spec_from_file_location("download_datasets", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_dataset_selection() -> None:
    assert [d.key for d in module.selected_specs("shanghaitech")] == ["shanghaitech"]
    assert [d.key for d in module.selected_specs("all")] == ["shanghaitech", "ucf_qnrf"]


def test_sources_are_configured() -> None:
    shanghai = module.DATASETS["shanghaitech"]
    qnrf = module.DATASETS["ucf_qnrf"]
    assert shanghai.provider == "gdrive"
    assert shanghai.gdrive_id == "1DLgEpNEPp3UqPnEtzW0BSMdS151kRNCs"
    assert qnrf.provider == "http"
    assert qnrf.url and "crcv.ucf.edu" in qnrf.url


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


def test_stale_html_partial_is_removed(tmp_path: Path) -> None:
    archive = tmp_path / "ShanghaiTech.zip"
    partial = tmp_path / "ShanghaiTech.zip.part"
    partial.write_text("<html>Dropbox error</html>", encoding="utf-8")

    returned_partial, existing = module.prepare_existing_download(archive, force=False)
    assert returned_partial == partial
    assert existing == 0
    assert not partial.exists()


def test_human_bytes() -> None:
    assert module.human_bytes(1024) == "1.0 KiB"
    assert module.human_bytes(1024 * 1024) == "1.0 MiB"


def test_ucf_uses_official_source() -> None:
    spec = module.DATASETS["ucf_qnrf"]
    assert spec.provider == "http"
    assert spec.gdrive_id is None
    assert spec.url == "https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip"
