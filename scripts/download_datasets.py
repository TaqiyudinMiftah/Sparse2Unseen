#!/usr/bin/env python3
"""Download raw ShanghaiTech and UCF-QNRF crowd-counting datasets.

ShanghaiTech is downloaded from a public Google Drive mirror linked by the
official SASNet repository. UCF-QNRF is downloaded from the canonical UCF CRCV
archive. The UCF server currently presents a certificate chain that may fail in
some containers/HPC environments; use --insecure-ssl only as an explicit fallback.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import certifi
import requests

MIB = 1024 * 1024
GIB = 1024 * MIB
Provider = Literal["http", "gdrive", "kaggle"]


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
        provider="kaggle",
        kaggle_handle="faihajalamtopu/ucf-qnrf",
        url="https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip",
        archive_name="UCF-QNRF_ECCV18.zip",
        extract_dir="ucf_qnrf",
        expected_markers=("Train", "Test"),
        approximate_required_space_gib=10.0,
        source_note=(
            "Public Kaggle mirror of the raw UCF-QNRF tree. "
            "Canonical source: UCF Center for Research in Computer Vision."
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw crowd-counting datasets for Sparse2Unseen."
    )
    parser.add_argument("dataset", choices=["all", *DATASETS.keys()])
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw"),
        help="Destination root (default: data/raw).",
    )
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP connect/read timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help=(
            "Disable TLS certificate verification for direct HTTPS downloads. "
            "Only relevant when --ucf-source official is selected."
        ),
    )
    parser.add_argument(
        "--ucf-source",
        choices=["kaggle", "official"],
        default="kaggle",
        help=(
            "Transport for UCF-QNRF (default: kaggle). "
            "Use 'official' to download from UCF CRCV directly."
        ),
    )
    return parser.parse_args()


def ucf_spec(source: str = "kaggle") -> DatasetSpec:
    if source == "kaggle":
        return DATASETS["ucf_qnrf"]
    if source == "official":
        base = DATASETS["ucf_qnrf"]
        return DatasetSpec(
            key=base.key,
            display_name=base.display_name,
            provider="http",
            url=base.url,
            archive_name=base.archive_name,
            extract_dir=base.extract_dir,
            expected_markers=base.expected_markers,
            approximate_required_space_gib=base.approximate_required_space_gib,
            source_note=(
                "Canonical archive from the UCF Center for Research in Computer Vision."
            ),
        )
    raise ValueError(f"Unknown UCF source: {source}")


def selected_specs(name: str, ucf_source: str = "kaggle") -> list[DatasetSpec]:
    if name == "all":
        return [DATASETS["shanghaitech"], ucf_spec(ucf_source)]
    if name == "ucf_qnrf":
        return [ucf_spec(ucf_source)]
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
            f"Approximately {recommended / GIB:.1f} GiB is recommended.",
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

    # A valid partial ZIP still starts with the ZIP signature. HTML/error responses do not.
    if partial.exists() and not looks_like_zip(partial):
        print(f"Removing stale non-ZIP partial download: {partial}")
        partial.unlink()

    return partial, partial.stat().st_size if partial.exists() else 0


def validate_download(output: Path, source: str) -> None:
    if not looks_like_zip(output):
        bad = output.with_suffix(output.suffix + ".bad")
        output.replace(bad)
        raise RuntimeError(
            f"Downloaded content from {source} is not a ZIP archive. "
            f"Saved the invalid response as {bad}."
        )
    print(f"Saved: {output} ({human_bytes(output.stat().st_size)})")


def download_http(
    url: str,
    output: Path,
    timeout: int,
    force: bool,
    insecure_ssl: bool = False,
) -> None:
    partial, existing = prepare_existing_download(output, force)
    if existing == -1:
        return

    headers = {
        "User-Agent": (
            "Mozilla/5.0 Sparse2Unseen-dataset-downloader/1.2 "
            "(+https://github.com/TaqiyudinMiftah/Sparse2Unseen)"
        )
    }
    if existing:
        headers["Range"] = f"bytes={existing}-"

    verify: str | bool = False if insecure_ssl else certifi.where()
    print(f"Downloading: {url}")
    if insecure_ssl:
        print(
            "WARNING: TLS certificate verification is disabled for this download.",
            file=sys.stderr,
        )
    else:
        print(f"TLS CA bundle: {certifi.where()}")
    if existing:
        print(f"Resuming from {human_bytes(existing)}: {partial}")

    try:
        response = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(timeout, timeout),
            verify=verify,
            allow_redirects=True,
        )
    except requests.exceptions.SSLError as exc:
        raise RuntimeError(
            "TLS certificate verification failed even with the certifi CA bundle. "
            "Because this is the official UCF URL, you may retry explicitly with "
            "'--insecure-ssl' if you accept the transport-security tradeoff."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    if existing and response.status_code == 416:
        partial.replace(output)
        validate_download(output, url)
        return

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {response.status_code} while downloading {url}"
        ) from exc

    if existing and response.status_code == 206:
        mode = "ab"
        base = existing
    else:
        mode = "wb"
        base = 0

    content_length = response.headers.get("Content-Length")
    response_size = int(content_length) if content_length and content_length.isdigit() else None
    total = base + response_size if response_size is not None else None
    started_at = time.monotonic()
    downloaded_this_session = 0
    partial.parent.mkdir(parents=True, exist_ok=True)

    try:
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=4 * MIB):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded_this_session += len(chunk)
                print_progress(base + downloaded_this_session, total, started_at)
    except KeyboardInterrupt:
        print(f"\nDownload interrupted. Partial file kept for resume: {partial}")
        raise
    finally:
        response.close()

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
    result = gdown.download(id=file_id, output=str(output), quiet=False, resume=True)
    if result is None:
        raise RuntimeError(
            "gdown could not download the public dataset archive. "
            "Check network access to Google Drive and retry."
        )

    if partial.exists() and not looks_like_zip(partial):
        partial.unlink(missing_ok=True)
    validate_download(output, f"Google Drive file {file_id}")


def download_kaggle(handle: str, destination: Path, force: bool) -> None:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle download support requires 'kagglehub'. Run 'uv sync' and retry."
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading public Kaggle dataset: {handle}")
    try:
        result = kagglehub.dataset_download(
            handle,
            output_dir=str(destination),
            force_download=force,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Kaggle download failed for {handle}: {exc}"
        ) from exc
    print(f"Kaggle dataset available at: {result}")


def download_dataset(
    spec: DatasetSpec,
    output: Path,
    timeout: int,
    force: bool,
    insecure_ssl: bool,
) -> None:
    if spec.provider == "http":
        if not spec.url:
            raise RuntimeError(f"Missing HTTP URL for {spec.key}")
        download_http(
            spec.url,
            output,
            timeout=timeout,
            force=force,
            insecure_ssl=insecure_ssl,
        )
        return
    if spec.provider == "gdrive":
        if not spec.gdrive_id:
            raise RuntimeError(f"Missing Google Drive file ID for {spec.key}")
        download_gdrive(spec.gdrive_id, output, force=force)
        return
    if spec.provider == "kaggle":
        raise RuntimeError(
            "Kaggle providers are downloaded directly into the extraction directory."
        )
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
    insecure_ssl: bool,
) -> None:
    archive = dest / spec.archive_name
    extract_root = dest / spec.extract_dir

    print("\n" + "=" * 72)
    print(spec.display_name)
    print(spec.source_note)
    if spec.provider == "kaggle":
        print(f"Download to: {extract_root}")
    else:
        print(f"Archive: {archive}")
        if extract:
            print(f"Extract to: {extract_root}")

    if dry_run:
        if spec.provider == "gdrive":
            print(f"[dry-run] would download Google Drive file {spec.gdrive_id}")
        elif spec.provider == "kaggle":
            print(f"[dry-run] would download Kaggle dataset {spec.kaggle_handle}")
        else:
            print(f"[dry-run] would download {spec.url}")
        if extract and spec.provider != "kaggle":
            print(f"[dry-run] would extract {archive} to {extract_root}")
        return

    if extract and not force and extraction_is_complete(spec, extract_root):
        print(f"Dataset already appears extracted: {extract_root}")
        print("Use --force to redownload/re-extract it.")
        return

    if spec.provider == "kaggle":
        if not extract:
            raise RuntimeError(
                "--no-extract is not supported for Kaggle datasets because "
                "kagglehub materializes the dataset tree directly."
            )
        if not spec.kaggle_handle:
            raise RuntimeError(f"Missing Kaggle handle for {spec.key}")
        download_kaggle(spec.kaggle_handle, extract_root, force=force)
    else:
        download_dataset(
            spec,
            archive,
            timeout=timeout,
            force=force,
            insecure_ssl=insecure_ssl,
        )

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

    if spec.provider != "kaggle" and not keep_archive:
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
                insecure_ssl=args.insecure_ssl,
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
