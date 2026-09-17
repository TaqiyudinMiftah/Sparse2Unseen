#!/usr/bin/env python3
"""Download raw ShanghaiTech and UCF-QNRF crowd-counting datasets.

ShanghaiTech is downloaded from the public Google Drive mirror linked by the
official SASNet repository. UCF-QNRF is downloaded from the official UCF CRCV
archive. Downloads are validated as ZIP files before extraction.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

MIB = 1024 * 1024
GIB = 1024 * MIB

Provider = Literal["http", "gdrive"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    provider: Provider
    archive_name: str
    extract_dir: str
    expected_markers: tuple[str, ...]
    approximate_required_space_gib: float
    source_note: str
    url: str | None = None
    gdrive_id: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "shanghaitech": DatasetSpec(
        key="shanghaitech",
        display_name="ShanghaiTech Part A + Part B",
        provider="gdrive",
        gdrive_id="1DLgEpNEPp3UqPnEtzW0BSMdS151kRNCs",
        archive_name="ShanghaiTech.zip",
        extract_dir="shanghaitech",
        expected_markers=("part_A_final", "part_B_final"),
        approximate_required_space_gib=2.0,
        source_note=(
            "Public Google Drive mirror linked by the official TencentYoutuResearch "
            "SASNet repository."
        ),
    ),
    "ucf_qnrf": DatasetSpec(
        key="ucf_qnrf",
        display_name="UCF-QNRF",
        provider="http",
        url="https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip",
        archive_name="UCF-QNRF_ECCV18.zip",
        extract_dir="ucf_qnrf",
        expected_markers=("Train", "Test"),
        approximate_required_space_gib=10.0,
        source_note="Official UCF Center for Research in Computer Vision archive.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw crowd-counting datasets for Sparse2Unseen."
    )
    parser.add_argument(
        "dataset",
        choices=["all", *DATASETS.keys()],
        help="Dataset to download.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw"),
        help="Destination root (default: data/raw).",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download archives but do not extract them.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep ZIP files after successful extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload/re-extract even when output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without downloading anything.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP socket timeout in seconds for direct HTTP downloads (default: 60).",
    )
    return parser.parse_args()


def selected_specs(name: str) -> list[DatasetSpec]:
    if name == "all":
        return [DATASETS["shanghaitech"], DATASETS["ucf_qnrf"]]
    return [DATASETS[name]]


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def check_disk_space(dest: Path, specs: Iterable[DatasetSpec]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    recommended = sum(s.approximate_required_space_gib for s in specs) * GIB
    if free < recommended:
        print(
            f"WARNING: only {human_bytes(free)} free under {dest}. "
            f"Approximately {recommended / GIB:.1f} GiB is recommended for the "
            "selected download(s), including extraction space.",
            file=sys.stderr,
        )


def looks_like_zip(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as handle:
        signature = handle.read(4)
    return signature in {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}


def find_marker(root: Path, marker: str) -> bool:
    if not root.exists():
        return False
    if (root / marker).exists():
        return True
    try:
        return any(p.name == marker for p in root.rglob(marker))
    except OSError:
        return False


def extraction_is_complete(spec: DatasetSpec, extract_root: Path) -> bool:
    return extract_root.exists() and all(
        find_marker(extract_root, marker) for marker in spec.expected_markers
    )


def build_request(url: str, start: int = 0) -> urllib.request.Request:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 Sparse2Unseen-dataset-downloader/1.1 "
            "(+https://github.com/TaqiyudinMiftah/Sparse2Unseen)"
        )
    }
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.Request(url, headers=headers)


def print_progress(downloaded: int, total: int | None, started_at: float) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-6)
    speed = downloaded / elapsed
    if total:
        pct = 100.0 * downloaded / total
        msg = (
            f"\r  {pct:6.2f}%  {human_bytes(downloaded)} / {human_bytes(total)} "
            f"at {human_bytes(int(speed))}/s"
        )
    else:
        msg = f"\r  {human_bytes(downloaded)} at {human_bytes(int(speed))}/s"
    print(msg, end="", flush=True)


def prepare_existing_download(output: Path, force: bool) -> tuple[Path, int]:
    partial = output.with_suffix(output.suffix + ".part")

    if force:
        output.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    if output.exists():
        if looks_like_zip(output):
            print(f"Archive already present: {output}")
            return partial, -1
        raise RuntimeError(
            f"Existing file is not a ZIP archive: {output}. "
            "Delete it or rerun with --force."
        )

    if partial.exists() and not looks_like_zip(partial):
        print(f"Removing stale non-ZIP partial download: {partial}")
        partial.unlink()

    return partial, partial.stat().st_size if partial.exists() else 0


def download_http(url: str, output: Path, timeout: int, force: bool) -> None:
    partial, existing = prepare_existing_download(output, force)
    if existing == -1:
        return

    request = build_request(url, existing)
    print(f"Downloading: {url}")
    if existing:
        print(f"Resuming from {human_bytes(existing)}: {partial}")

    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if existing and exc.code == 416:
            partial.replace(output)
            return
        raise RuntimeError(f"HTTP {exc.code} while downloading {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download {url}: {exc.reason}") from exc

    status = getattr(response, "status", response.getcode())
    content_length = response.headers.get("Content-Length")
    response_size = int(content_length) if content_length and content_length.isdigit() else None

    if existing and status == 206:
        mode = "ab"
        base = existing
    else:
        mode = "wb"
        base = 0

    total = base + response_size if response_size is not None else None
    started_at = time.monotonic()
    downloaded_this_session = 0
    partial.parent.mkdir(parents=True, exist_ok=True)

    try:
        with response, partial.open(mode) as handle:
            while True:
                chunk = response.read(4 * MIB)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded_this_session += len(chunk)
                print_progress(base + downloaded_this_session, total, started_at)
    except KeyboardInterrupt:
        print(f"\nDownload interrupted. Partial file kept for resume: {partial}")
        raise

    print()
    partial.replace(output)
    validate_download(output, url)


def download_gdrive(file_id: str, output: Path, force: bool) -> None:
    partial, existing = prepare_existing_download(output, force)
    if existing == -1:
        return

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive download support requires 'gdown'. Run 'uv sync' and retry."
        ) from exc

    print(f"Downloading Google Drive file: {file_id}")
    result = gdown.download(
        id=file_id,
        output=str(output),
        quiet=False,
        resume=True,
    )
    if result is None:
        raise RuntimeError(
            "gdown could not download the public ShanghaiTech archive. "
            "Check network access to Google Drive and retry."
        )

    if partial.exists() and not looks_like_zip(partial):
        partial.unlink(missing_ok=True)
    validate_download(output, f"Google Drive file {file_id}")


def validate_download(output: Path, source: str) -> None:
    if not looks_like_zip(output):
        bad = output.with_suffix(output.suffix + ".bad")
        output.replace(bad)
        raise RuntimeError(
            f"Downloaded content from {source} is not a ZIP archive. "
            f"Saved the invalid response as {bad}."
        )
    print(f"Saved: {output} ({human_bytes(output.stat().st_size)})")


def download_dataset(spec: DatasetSpec, output: Path, timeout: int, force: bool) -> None:
    if spec.provider == "http":
        if not spec.url:
            raise RuntimeError(f"Missing HTTP URL for {spec.key}")
        download_http(spec.url, output, timeout=timeout, force=force)
        return
    if spec.provider == "gdrive":
        if not spec.gdrive_id:
            raise RuntimeError(f"Missing Google Drive file ID for {spec.key}")
        download_gdrive(spec.gdrive_id, output, force=force)
        return
    raise RuntimeError(f"Unsupported provider: {spec.provider}")


def extract_zip(archive: Path, destination: Path, force: bool) -> None:
    if destination.exists() and force:
        print(f"Removing previous extraction: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive.name} -> {destination}")
    try:
        with zipfile.ZipFile(archive, "r", allowZip64=True) as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Corrupt ZIP member detected: {bad_member}")
            zf.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Invalid ZIP archive: {archive}") from exc


def process_dataset(
    spec: DatasetSpec,
    dest: Path,
    extract: bool,
    keep_archive: bool,
    force: bool,
    dry_run: bool,
    timeout: int,
) -> None:
    archive = dest / spec.archive_name
    extract_root = dest / spec.extract_dir

    print("\n" + "=" * 72)
    print(spec.display_name)
    print(spec.source_note)
    print(f"Archive: {archive}")
    if extract:
        print(f"Extract to: {extract_root}")

    if dry_run:
        if spec.provider == "gdrive":
            print(f"[dry-run] would download Google Drive file {spec.gdrive_id}")
        else:
            print(f"[dry-run] would download {spec.url}")
        if extract:
            print(f"[dry-run] would extract {archive} to {extract_root}")
        return

    if extract and not force and extraction_is_complete(spec, extract_root):
        print(f"Dataset already appears extracted: {extract_root}")
        print("Use --force to redownload/re-extract it.")
        return

    download_dataset(spec, archive, timeout=timeout, force=force)

    if not extract:
        return

    extract_zip(archive, extract_root, force=force)

    missing = [
        marker
        for marker in spec.expected_markers
        if not find_marker(extract_root, marker)
    ]
    if missing:
        print(
            "WARNING: extraction completed, but expected marker(s) were not "
            f"found: {', '.join(missing)}. Inspect {extract_root} manually.",
            file=sys.stderr,
        )
    else:
        print(f"Verified expected dataset structure under: {extract_root}")

    if not keep_archive:
        archive.unlink(missing_ok=True)
        print(f"Removed archive after successful extraction: {archive}")


def main() -> int:
    args = parse_args()
    specs = selected_specs(args.dataset)
    args.dest = args.dest.expanduser().resolve()

    print("Sparse2Unseen raw dataset downloader")
    print(f"Destination: {args.dest}")
    print(
        "Please use these datasets only under their respective research/use "
        "terms and cite the original dataset papers."
    )

    if not args.dry_run:
        check_disk_space(args.dest, specs)

    failures: list[tuple[str, str]] = []
    for spec in specs:
        try:
            process_dataset(
                spec=spec,
                dest=args.dest,
                extract=not args.no_extract,
                keep_archive=args.keep_archive,
                force=args.force,
                dry_run=args.dry_run,
                timeout=args.timeout,
            )
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # noqa: BLE001 - CLI should give a clean error.
            print(f"ERROR [{spec.display_name}]: {exc}", file=sys.stderr)
            failures.append((spec.display_name, str(exc)))

    if failures:
        print("\nCompleted with errors:", file=sys.stderr)
        for name, message in failures:
            print(f"  - {name}: {message}", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
