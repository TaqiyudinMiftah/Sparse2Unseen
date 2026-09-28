#!/usr/bin/env python3
"""Detach an audited sparse MPCount job until GPU memory is available."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPCOUNT_ROOT = PROJECT_ROOT / "external/MPCount"
QUEUE_ROOT = PROJECT_ROOT / "runs/queues"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_version(seed: int, version: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", version):
        raise ValueError("Run version must be a single safe path component")
    if not version.startswith(f"stb_10_seed{seed}_effbs16_"):
        raise ValueError("Retry version must identify the matching sparse seed")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_gpus(output: str) -> list[dict]:
    rows = []
    for fields in csv.reader(output.splitlines()):
        if len(fields) != 4:
            raise ValueError("Unexpected nvidia-smi GPU inventory")
        index, uuid, used, total = (field.strip() for field in fields)
        rows.append({"index": int(index), "uuid": uuid, "free_mib": int(total) - int(used)})
    return rows


def query_gpus() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return parse_gpus(result.stdout)


def wait_for_gpu(request, update, cancelled, *, query=query_gpus, sleep=time.sleep):
    previous = None
    stable = 0
    while not cancelled():
        try:
            inventory = query()
            eligible = [gpu for gpu in inventory
                        if gpu["free_mib"] >= request["min_free_mib"]
                        and (not request["gpus"] or gpu["index"] in request["gpus"])]
            candidate = max(eligible, key=lambda gpu: (gpu["free_mib"], -gpu["index"])) if eligible else None
            current = candidate["uuid"] if candidate else None
            stable = stable + 1 if current is not None and current == previous else int(current is not None)
            previous = current
            update(gpu_inventory=inventory, candidate=candidate, stable_checks=stable,
                   gpu_query_error=None)
            if candidate and stable >= request["stable_checks"]:
                return candidate
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:
            previous, stable = None, 0
            update(gpu_query_error=type(error).__name__, stable_checks=0)
        sleep(request["poll_seconds"])
    raise InterruptedError("Queue cancellation requested")


def prepare_request(args) -> dict:
    validate_version(args.seed, args.run_version)
    if args.min_free_mib < 1 or args.poll_seconds < 1 or args.stable_checks < 1:
        raise ValueError("Memory threshold, poll interval, and stable checks must be positive")
    if (MPCOUNT_ROOT / "logs" / args.run_version).exists():
        raise FileExistsError("Retry training directory already exists")
    config_rel = f"configs/mpcount/stb_10_train_accum4_seed{args.seed}.yml"
    config = yaml.safe_load((PROJECT_ROOT / config_rel).read_text(encoding="utf-8"))
    if (config["seed"], config["num_epochs"], config["checkpoint"]) != (args.seed, 180, None):
        raise ValueError("Queue requires a clean, predeclared 180-epoch sparse run")
    train_root = MPCOUNT_ROOT / config["train_dataset"]["params"]["root"] / "train"
    val_root = MPCOUNT_ROOT / config["val_dataset"]["params"]["root"] / "val"
    split_rel = f"splits/stb_10_seed{args.seed}.json"
    split = json.loads((PROJECT_ROOT / split_rel).read_text(encoding="utf-8"))
    images = list(train_root.glob("*.jpg")) + list(train_root.glob("*.png"))
    if len(images) != 32 or {image.stem for image in images} != set(split["labeled_ids"]):
        raise ValueError("Sparse training root does not match the committed labeled split")
    for image in images:
        if not image.with_suffix(".npy").is_file() or not image.with_name(image.stem + "_dmap.npy").is_file():
            raise ValueError("Sparse source annotations or density maps are incomplete")
    if len(list(val_root.glob("*.jpg")) + list(val_root.glob("*.png"))) != 80:
        raise ValueError("Expected the fixed 80-image source validation partition")
    files = [config_rel, split_rel, "uv.lock", "scripts/queue_sparse_mpcount.py",
             "scripts/train_mpcount_accum.py",
             "scripts/evaluate_sparse_mpcount.py", "scripts/select_mpcount_checkpoint.py",
             "src/sparse2unseen/monitoring/mpcount_log.py"]
    files.extend(f"configs/mpcount/stb_100_effbs16_test_{domain}.yml"
                 for domain in ("stb", "sta", "qnrf"))
    return {
        "created_at": utc_now(), "seed": args.seed, "version": args.run_version,
        "config": config_rel, "min_free_mib": args.min_free_mib,
        "poll_seconds": args.poll_seconds, "stable_checks": args.stable_checks,
        "gpus": args.gpu or [], "wandb": args.wandb, "evaluate": args.evaluate,
        "input_sha256": {path: sha256(PROJECT_ROOT / path) for path in files},
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "mpcount_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MPCOUNT_ROOT, text=True).strip(),
    }


def check_inputs(request) -> None:
    for path, digest in request["input_sha256"].items():
        if sha256(PROJECT_ROOT / path) != digest:
            raise RuntimeError(f"Queued input changed; refusing to launch: {path}")
    upstream = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MPCOUNT_ROOT, text=True).strip()
    if upstream != request["mpcount_commit"]:
        raise RuntimeError("Upstream MPCount checkout changed while queued")


def run_child(command, gpu, log_path, update, cancelled) -> int:
    if cancelled():
        raise InterruptedError("Queue cancellation requested")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as output:
        child = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        update(child_pid=child.pid, command=command, gpu=gpu)
        while child.poll() is None:
            if cancelled():
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                update(child_pid=None, child_exit_code=child.returncode)
                raise InterruptedError("Queue cancellation requested")
            time.sleep(1)
        update(child_pid=None, child_exit_code=child.returncode)
        return child.returncode


def worker(queue_dir: Path) -> int:
    request = json.loads((queue_dir / "request.json").read_text(encoding="utf-8"))
    state = {"worker_pid": os.getpid(), "version": request["version"]}
    stopped = {"value": False}
    signal.signal(signal.SIGTERM, lambda *_: stopped.update(value=True))
    signal.signal(signal.SIGINT, lambda *_: stopped.update(value=True))
    cancelled = lambda: stopped["value"] or (queue_dir / "cancel.requested").exists()

    def update(**values):
        state.update(values, updated_at=utc_now())
        write_json(queue_dir / "status.json", state)

    with (queue_dir / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            check_inputs(request)
            update(state="waiting_for_training_memory")
            gpu = wait_for_gpu(request, update, cancelled)
            check_inputs(request)
            command = [sys.executable, str(PROJECT_ROOT / "scripts/train_mpcount_accum.py"),
                       "--config", str(PROJECT_ROOT / request["config"]), "--version", request["version"]]
            if request["wandb"]:
                command.append("--wandb")
            update(state="training", training_started_at=utc_now())
            if run_child(command, gpu, queue_dir / "training.log", update, cancelled) != 0:
                raise RuntimeError("Training exited unsuccessfully; no automatic retry")
            log = MPCOUNT_ROOT / "logs" / request["version"] / "log.txt"
            if not any(line.startswith("End training at ") for line in log.read_text().splitlines()):
                raise RuntimeError("Training did not log successful completion")
            update(training_finished_at=utc_now())
            if request["evaluate"]:
                update(state="waiting_for_evaluation_memory")
                gpu = wait_for_gpu(request, update, cancelled)
                check_inputs(request)
                command = [sys.executable, str(PROJECT_ROOT / "scripts/evaluate_sparse_mpcount.py"),
                           "--seed", str(request["seed"]), "--run-version", request["version"], "--domain", "all"]
                update(state="evaluating")
                if run_child(command, gpu, queue_dir / "evaluation.log", update, cancelled) != 0:
                    raise RuntimeError("Test evaluation exited unsuccessfully")
            update(state="complete", completed_at=utc_now())
            return 0
        except InterruptedError as error:
            update(state="cancelled", error=str(error))
            return 2
        except Exception as error:
            update(state="failed", error=str(error))
            traceback.print_exc()
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cancel", help="Request cancellation of this queue/run version")
    parser.add_argument("--seed", type=int, choices=(1, 2, 3))
    parser.add_argument("--run-version")
    parser.add_argument("--min-free-mib", type=int, default=7168)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--gpu", action="append", type=int, help="Eligible GPU index (repeatable)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run fixed tests after successful training")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve(strict=True))
    if args.cancel:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.cancel):
            raise ValueError("Invalid queue version")
        queue_dir = QUEUE_ROOT / args.cancel
        if not (queue_dir / "request.json").is_file():
            raise FileNotFoundError("Queue request does not exist")
        (queue_dir / "cancel.requested").touch()
        print(f"Cancellation requested: {queue_dir}")
        return 0
    if args.seed is None or args.run_version is None:
        parser.error("--seed and --run-version are required")
    request = prepare_request(args)
    queue_dir = QUEUE_ROOT / args.run_version
    queue_dir.mkdir(parents=True, exist_ok=False)
    write_json(queue_dir / "request.json", request)
    write_json(queue_dir / "status.json", {"state": "queued", "updated_at": utc_now()})
    with (queue_dir / "queue.log").open("a", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(queue_dir)],
            cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    print(f"Detached queue PID: {process.pid}\nStatus: {queue_dir / 'status.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
