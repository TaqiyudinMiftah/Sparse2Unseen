#!/usr/bin/env python3
"""Prepare downloaded crowd-counting datasets with official MPCount utilities."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetPlan:
    key: str
    mpcount_name: str
    raw_origin: Path
    processed_root: Path
    split_prefix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare raw datasets using official MPCount preprocessing."
    )
    parser.add_argument("dataset", choices=["all", "sta", "stb", "qnrf"])
    parser.add_argument(
        "--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "mpcount",
    )
    parser.add_argument(
        "--mpcount-dir", type=Path, default=PROJECT_ROOT / "external" / "MPCount"
    )
    parser.add_argument("--skip-density", action="store_true")
    parser.add_argument("--skip-manifests", action="store_true")
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--no-link-mpcount", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete selected processed outputs and regenerate them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    print("$ " + " ".join(str(x) for x in command))
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def bootstrap_mpcount(mpcount_dir: Path, no_bootstrap: bool, dry_run: bool) -> None:
    marker = mpcount_dir / "utils" / "preprocess_data.py"
    if marker.exists():
        print(f"MPCount checkout: {mpcount_dir}")
        return
    if no_bootstrap:
        raise FileNotFoundError(
            f"MPCount is missing at {mpcount_dir}. Run bash scripts/bootstrap_mpcount.sh."
        )
    bootstrap = PROJECT_ROOT / "scripts" / "bootstrap_mpcount.sh"
    run_command(["bash", str(bootstrap)], cwd=PROJECT_ROOT, dry_run=dry_run)
    if not dry_run and not marker.exists():
        raise RuntimeError(f"MPCount bootstrap did not create {marker}")


def has_children(path: Path, children: Iterable[str]) -> bool:
    return path.is_dir() and all((path / child).exists() for child in children)


def locate_shanghaitech_part(raw_root: Path, part: str) -> Path:
    base = raw_root / "shanghaitech"
    names = (part, f"{part}_final")
    likely = [
        base / part,
        base / f"{part}_final",
        base / "ShanghaiTech" / part,
        base / "ShanghaiTech" / f"{part}_final",
    ]
    for candidate in likely:
        if has_children(candidate, ("train_data", "test_data")):
            return candidate.resolve()
    if base.exists():
        for name in names:
            for candidate in base.rglob(name):
                if ".mpcount_alias" not in candidate.parts and has_children(
                    candidate, ("train_data", "test_data")
                ):
                    return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate ShanghaiTech {part} under {base}; expected train_data/test_data."
    )


def locate_qnrf_root(raw_root: Path) -> Path:
    base = raw_root / "ucf_qnrf"
    likely = [
        base,
        base / "UCF-QNRF_ECCV18",
        base / "UCF-QNRF_ECCV18" / "UCF-QNRF_ECCV18",
    ]
    for candidate in likely:
        if has_children(candidate, ("Train", "Test")):
            return candidate.resolve()
    if base.exists():
        for train_dir in base.rglob("Train"):
            candidate = train_dir.parent
            if has_children(candidate, ("Train", "Test")):
                return candidate.resolve()
    raise FileNotFoundError(f"Could not locate UCF-QNRF Train/Test under {base}.")


def ensure_shanghai_alias(raw_root: Path, part: str, target: Path, dry_run: bool) -> Path:
    # MPCount checks the basename literally: part_A => STA, otherwise STB.
    alias = raw_root / ".mpcount_alias" / "ShanghaiTech" / part
    if alias.is_symlink():
        if alias.resolve() == target.resolve():
            return alias
        if not dry_run:
            alias.unlink()
    elif alias.exists():
        raise RuntimeError(f"Cannot replace non-symlink alias path: {alias}")
    print(f"Alias: {alias} -> {target}")
    if not dry_run:
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(target.resolve(), target_is_directory=True)
    return alias


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def expected_phase_counts(plan: DatasetPlan, mpcount_dir: Path) -> dict[str, int]:
    if plan.key in {"sta", "stb"}:
        train = count_nonempty_lines(mpcount_dir / f"{plan.split_prefix}_train.txt")
        val = count_nonempty_lines(mpcount_dir / f"{plan.split_prefix}_val.txt")
        test = len(list((plan.raw_origin / "test_data" / "images").glob("*.jpg")))
    else:
        train = count_nonempty_lines(mpcount_dir / "ucf_train.txt")
        val = count_nonempty_lines(mpcount_dir / "ucf_val.txt")
        test = len(list((plan.raw_origin / "Test").glob("*.jpg")))
    return {"train": train, "val": val, "test": test}


def phase_images(root: Path, phase: str) -> list[Path]:
    d = root / phase
    return sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))


def points_complete(root: Path, expected: dict[str, int]) -> bool:
    for phase, expected_count in expected.items():
        images = phase_images(root, phase)
        if len(images) != expected_count:
            return False
        if any(not image.with_suffix(".npy").exists() for image in images):
            return False
    return True


def density_complete(root: Path, expected: dict[str, int]) -> bool:
    for phase, expected_count in expected.items():
        images = phase_images(root, phase)
        if len(images) != expected_count:
            return False
        if any(
            not image.with_name(image.stem + "_dmap.npy").exists() for image in images
        ):
            return False
    return True


def ensure_mpcount_data_link(
    mpcount_dir: Path, key: str, target: Path, dry_run: bool
) -> None:
    link = mpcount_dir / "data" / key
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            print(f"MPCount data link already correct: {link}")
            return
        if not dry_run:
            link.unlink()
    elif link.exists():
        raise RuntimeError(
            f"Refusing to replace existing non-symlink MPCount data path: {link}. "
            "Move it manually or use --no-link-mpcount."
        )
    print(f"MPCount data link: {link} -> {target}")
    if not dry_run:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)


def build_manifest(root: Path, key: str, phase: str, dry_run: bool) -> None:
    output = PROJECT_ROOT / "data" / "manifests" / f"{key}_{phase}.jsonl"
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "build_manifest.py"),
            "--root",
            str(root),
            "--phase",
            phase,
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        dry_run=dry_run,
    )


def selected_keys(name: str) -> list[str]:
    return ["sta", "stb", "qnrf"] if name == "all" else [name]


def resolve_plans(
    raw_root: Path, processed_root: Path, keys: list[str], dry_run: bool
) -> list[DatasetPlan]:
    plans: list[DatasetPlan] = []
    if "sta" in keys:
        target = locate_shanghaitech_part(raw_root, "part_A")
        origin = ensure_shanghai_alias(raw_root, "part_A", target, dry_run)
        plans.append(DatasetPlan("sta", "sta", origin, processed_root / "sta", "sta"))
    if "stb" in keys:
        target = locate_shanghaitech_part(raw_root, "part_B")
        origin = ensure_shanghai_alias(raw_root, "part_B", target, dry_run)
        plans.append(DatasetPlan("stb", "stb", origin, processed_root / "stb", "stb"))
    if "qnrf" in keys:
        origin = locate_qnrf_root(raw_root)
        plans.append(DatasetPlan("qnrf", "qnrf", origin, processed_root / "qnrf", "ucf"))
    return plans


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    processed_root = args.processed_root.expanduser().resolve()
    mpcount_dir = args.mpcount_dir.expanduser().resolve()

    print("Sparse2Unseen dataset preparation")
    print(f"Raw root:       {raw_root}")
    print(f"Processed root: {processed_root}")
    print(f"MPCount:        {mpcount_dir}")

    bootstrap_mpcount(mpcount_dir, args.no_bootstrap, args.dry_run)
    if args.dry_run and not (mpcount_dir / "utils" / "preprocess_data.py").exists():
        print("Dry-run stops here because MPCount is not cloned yet.")
        return 0

    plans = resolve_plans(
        raw_root, processed_root, selected_keys(args.dataset), args.dry_run
    )

    for plan in plans:
        print("\n" + "=" * 72)
        print(f"Preparing {plan.key.upper()}")
        print(f"Raw origin: {plan.raw_origin}")
        print(f"Processed:  {plan.processed_root}")

        if args.force and plan.processed_root.exists():
            print(f"Removing processed output: {plan.processed_root}")
            if not args.dry_run:
                shutil.rmtree(plan.processed_root)

        expected = expected_phase_counts(plan, mpcount_dir)
        print(
            "Expected phase sizes: "
            + ", ".join(f"{phase}={count}" for phase, count in expected.items())
        )

        if points_complete(plan.processed_root, expected):
            print("Point preprocessing already complete; skipping.")
        else:
            run_command(
                [
                    sys.executable,
                    "utils/preprocess_data.py",
                    "--dataset",
                    plan.mpcount_name,
                    "--origin-dir",
                    str(plan.raw_origin),
                    "--data-dir",
                    str(plan.processed_root),
                ],
                cwd=mpcount_dir,
                dry_run=args.dry_run,
            )
            if not args.dry_run and not points_complete(plan.processed_root, expected):
                actual = {
                    phase: len(phase_images(plan.processed_root, phase))
                    for phase in expected
                }
                raise RuntimeError(
                    f"Unexpected processed counts for {plan.key}: "
                    f"expected={expected}, actual={actual}"
                )

        if not args.skip_density:
            if density_complete(plan.processed_root, expected):
                print("Density maps already complete; skipping.")
            else:
                run_command(
                    [
                        sys.executable,
                        "utils/dmap_gen.py",
                        "--path",
                        str(plan.processed_root),
                    ],
                    cwd=mpcount_dir,
                    dry_run=args.dry_run,
                )
                if not args.dry_run and not density_complete(
                    plan.processed_root, expected
                ):
                    raise RuntimeError(f"Density-map generation incomplete for {plan.key}")

        if not args.no_link_mpcount:
            ensure_mpcount_data_link(
                mpcount_dir, plan.key, plan.processed_root, args.dry_run
            )

        if not args.skip_manifests:
            for phase in ("train", "val", "test"):
                build_manifest(plan.processed_root, plan.key, phase, args.dry_run)

    print("\nPreparation complete.")
    print(
        "Next milestone: reproduce MPCount with 100% STB labels, "
        "then evaluate unchanged on STA and QNRF."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
