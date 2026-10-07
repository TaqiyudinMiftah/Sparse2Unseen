#!/usr/bin/env python3
"""Pin validated, committed code before the first production benchmark launch."""
from __future__ import annotations

import subprocess

import yaml

from sparse2unseen.experiments.common import ROOT, check_source_inputs, digest, load_json, now, write_json
from build_benchmark_suite import CODE_INPUTS


def finalize():
    suite = ROOT / "runs/benchmark_v1"
    matrix = load_json(suite / "matrix.json")
    if "frozen_at" in matrix or (suite / "queue_state.json").exists():
        raise RuntimeError("Production has already been finalized or launched")
    settings = yaml.safe_load((ROOT / "configs/suite/matrix.yaml").read_text())
    if matrix["settings"] != settings:
        raise RuntimeError("The prepared scientific recipe changed; create a new protocol version")
    for job in matrix["jobs"]:
        output = suite / "jobs" / job["name"]
        if any((output / filename).exists() for filename in ("resume.pt", "best.pt", "training_complete.json")):
            raise RuntimeError("Cannot change provenance after production training starts")
    paths = set(CODE_INPUTS) | {"scripts/finalize_benchmark_suite.py", "scripts/report_benchmark_suite.py"}
    paths |= {load_json(ROOT / job["spec"])["split"] for job in matrix["jobs"]}
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *sorted(paths)], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    if subprocess.check_output(["git", "status", "--porcelain", "--", *sorted(paths)], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit tested benchmark code and splits before finalizing")
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT / "external/MPCount", check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    frozen = {path: digest(ROOT / path) for path in CODE_INPUTS}
    for folder in ("models", "datasets", "utils"):
        for path in sorted((ROOT / "external/MPCount" / folder).glob("*.py")):
            frozen[str(path.relative_to(ROOT))] = digest(path)
    frozen_at = now()
    write_json(suite / "preparation_matrix.json", matrix)
    for job in matrix["jobs"]:
        path = ROOT / job["spec"]
        spec = load_json(path)
        spec["preparation_commit"] = spec["project_commit"]
        spec["project_commit"] = commit
        spec["frozen_at"] = frozen_at
        spec["input_sha256"] = {**frozen, spec["split"]: digest(ROOT / spec["split"]),
                                spec["train_manifest"]: digest(ROOT / spec["train_manifest"]),
                                spec["val_manifest"]: digest(ROOT / spec["val_manifest"])}
        check_source_inputs(spec)
        write_json(path, spec)
    # Pair methods for each seed early, while retaining source/fraction priority.
    fractions = matrix["settings"]["fractions"] + [1.0]
    def ordering(job):
        spec = load_json(ROOT / job["spec"])
        return (settings["sources"].index(spec["source"]), fractions.index(spec["fraction"]),
                spec["seed"], settings["methods"].index(spec["method"]))
    matrix["jobs"].sort(key=ordering)
    matrix.update(inputs=frozen, frozen_at=frozen_at, production_commit=commit)
    write_json(suite / "matrix.json", matrix)
    print(f"Production frozen at {commit}: {len(matrix['jobs'])} entries")


if __name__ == "__main__":
    finalize()
