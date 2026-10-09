#!/usr/bin/env python3
"""Run the frozen benchmark on available GPUs, with durable state and retries."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

from sparse2unseen.experiments.common import ROOT, digest, load_json, now, write_json

sys.path.insert(0, str(ROOT / "scripts"))
from queue_sparse_mpcount import query_gpus, wait_for_gpu

SUITE = ROOT / "runs/benchmark_v1"


def process_matches(pid, script_name: str) -> bool:
    if not pid:
        return False
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return str(ROOT / "scripts" / script_name).encode() in arguments
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def worker(args):
    matrix = load_json(SUITE / "matrix.json")
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    mutex = threading.RLock()
    state_path = SUITE / "queue_state.json"
    state = load_json(state_path) if state_path.exists() else {"jobs": matrix["jobs"]}
    state.update(worker_pid=os.getpid(), state="running", started_at=now())
    for job in state["jobs"]:
        if job["state"] in ("training", "evaluating", "waiting_for_evaluation_memory", "orphan_live"):
            if process_matches(job.get("child_pid"), "train_benchmark.py") or process_matches(job.get("child_pid"), "evaluate_benchmark.py"):
                job["state"] = "orphan_live"
            else:
                job.update(state="pending", child_pid=None)

    def cancelled():
        return stopped.is_set() or (SUITE / "stop.requested").exists()

    def save():
        with mutex:
            state["updated_at"] = now()
            write_json(state_path, state)

    def claim(gpu):
        with mutex:
            for job in state["jobs"]:
                if job["state"] == "orphan_live":
                    pid = job.get("child_pid")
                    if not (process_matches(pid, "train_benchmark.py") or process_matches(pid, "evaluate_benchmark.py")):
                        job.update(state="pending", child_pid=None)
                if job["state"] == "pending":
                    job.update(state="training", gpu=gpu, claimed_at=now(), attempts=job.get("attempts", 0)+1)
                    save()
                    return job
        return None

    def child(job, phase, gpu):
        spec = ROOT / job["spec"]
        job_dir = SUITE / "jobs" / job["name"]
        command = [sys.executable, str(ROOT / "scripts" / f"{phase}_benchmark.py"), "--spec", str(spec)]
        if phase == "train":
            command.append("--wandb")
            if (job_dir / "spec.json").exists():
                command.append("--resume")
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        console_dir = SUITE / "console" / job["name"]
        console_dir.mkdir(parents=True, exist_ok=True)
        with (console_dir / f"{phase}.log").open("a", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            with mutex:
                job.update(child_pid=process.pid, command=command, phase_started_at=now())
                save()
            while process.poll() is None:
                if cancelled():
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    with mutex:
                        job.update(child_pid=None, state="pending", last_exit_code=process.returncode)
                        save()
                    raise InterruptedError("Benchmark stop requested")
                save()
                time.sleep(5)
            with mutex:
                job.update(child_pid=None, last_exit_code=process.returncode, phase_finished_at=now())
                save()
            if process.returncode:
                raise RuntimeError(f"{phase} exited {process.returncode}; see {console_dir / (phase + '.log')}")

    def thread_work():
        while not cancelled():
            with mutex:
                unfinished = [job for job in state["jobs"] if job["state"] not in ("complete", "imported_complete", "failed")]
            if not unfinished:
                return
            try:
                for relative, expected in matrix["inputs"].items():
                    if digest(ROOT / relative) != expected:
                        raise RuntimeError(f"Frozen suite input changed: {relative}")
                try:
                    candidates = sorted(query_gpus(), key=lambda gpu: gpu["free_mib"], reverse=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:
                    with mutex:
                        state["gpu_query_error"] = type(error).__name__
                        save()
                    time.sleep(args.poll_seconds)
                    continue
                claimed_gpu = False
                for gpu in candidates:
                    if gpu["free_mib"] < args.min_free_mib:
                        continue
                    lock_path = SUITE / f"gpu-{gpu['uuid']}.lock"
                    with lock_path.open("w") as gpu_lock:
                        try:
                            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        request = {"min_free_mib": args.min_free_mib, "stable_checks": args.stable_checks,
                                   "poll_seconds": args.poll_seconds, "gpus": [gpu["index"]]}
                        gpu = wait_for_gpu(request, lambda **_: save(), cancelled)
                        job = claim(gpu)
                        if job is None:
                            break
                        claimed_gpu = True
                        try:
                            child(job, "train", gpu)
                            with mutex:
                                job["state"] = "waiting_for_evaluation_memory"
                                save()
                            gpu = wait_for_gpu(request, lambda **_: save(), cancelled)
                            with mutex:
                                job["state"] = "evaluating"
                                save()
                            child(job, "evaluate", gpu)
                            result = load_json(SUITE / "jobs" / job["name"] / "results.json")
                            if set(result["domains"]) != {"stb", "sta", "qnrf"}:
                                raise RuntimeError("Final evaluation did not cover every fixed domain")
                            with mutex:
                                job.update(state="complete", completed_at=now())
                                save()
                        except InterruptedError:
                            return
                        except Exception as error:
                            with mutex:
                                job.update(state="pending" if job["attempts"] < 3 else "failed", error=str(error))
                                save()
                            traceback.print_exc()
                        break
                if not claimed_gpu:
                    time.sleep(args.poll_seconds)
            except InterruptedError:
                return
            except Exception as error:
                with mutex:
                    state.update(state="error", error=str(error))
                    save()
                traceback.print_exc()
                stopped.set()
                return

    with (SUITE / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        threads = [threading.Thread(target=thread_work) for _ in range(args.workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if state["state"] != "error":
            failed = any(job["state"] == "failed" for job in state["jobs"])
            finished = all(job["state"] in ("complete", "imported_complete") for job in state["jobs"])
            state["state"] = "complete" if finished else "failed" if failed else "stopped"
        save()
    return 0 if state["state"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=8192)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.stop:
        (SUITE / "stop.requested").touch()
        return 0
    if args.worker:
        return worker(args)
    if not args.launch:
        parser.error("Use --launch or --stop")
    if not 1 <= args.workers <= 2 or min(args.min_free_mib, args.stable_checks, args.poll_seconds) < 1:
        parser.error("Use one or two workers and positive memory/poll settings")
    state_path = SUITE / "queue_state.json"
    if state_path.exists() and process_matches(load_json(state_path).get("worker_pid"), "run_benchmark_suite.py"):
        raise RuntimeError("Benchmark worker is already live")
    if (SUITE / "stop.requested").exists():
        raise RuntimeError("Explicit stop marker exists; resolve it before relaunching")
    load_json(SUITE / "matrix.json")
    with (SUITE / "worker_console.log").open("a", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "--workers", str(args.workers),
             "--min-free-mib", str(args.min_free_mib), "--stable-checks", str(args.stable_checks),
             "--poll-seconds", str(args.poll_seconds)], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
    print(f"Detached benchmark PID {process.pid}; status {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
