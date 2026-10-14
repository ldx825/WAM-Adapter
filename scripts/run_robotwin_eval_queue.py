#!/usr/bin/env python3
"""Resumable RoboTwin task evaluation queue across one or more GPUs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Queue
from threading import Lock


@dataclass(frozen=True)
class EvalJob:
    task_name: str
    phase: str
    trial_start: int
    num_trials: int
    seeds: tuple[int, ...] | None = None
    # When ``seeds`` is None (no prior accepted-seed manifest), this shard
    # starts RoboTwin's accepted-seed search at ``100000 * (1 + seed_override)``
    # instead of the shared default.  Cold-start shards therefore scan disjoint
    # seed intervals and can run in parallel without a manifest.
    seed_override: int | None = None
    gpu_id: int = -1

    @property
    def trial_end(self) -> int:
        return self.trial_start + self.num_trials

    @property
    def shard_tag(self) -> str:
        return f"shard{self.trial_start:03d}-{self.trial_end:03d}"


def parse_rate(path: Path) -> float:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"No numeric success rate in {path}")
    value = values[-1]
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Success rate outside [0,1] in {path}: {value}")
    return value


def result_path(output_dir: Path, task_name: str, phase: str) -> Path:
    suffix = "clean" if phase == "clean" else "random"
    return output_dir / task_name / f"_result_{suffix}.txt"


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
SEED_PATTERN = re.compile(r"current seed:\s*(\d+)")


def success_count_from_rate(rate: float, num_trials: int) -> int:
    count = int(round(rate * num_trials))
    if abs(rate - count / num_trials) > 1e-6:
        raise ValueError(
            f"Success rate {rate} is incompatible with {num_trials} integer trials."
        )
    return count


def parse_seeds_from_log(path: Path) -> list[int]:
    text = ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="ignore"))
    return [int(value) for value in SEED_PATTERN.findall(text)]


def extract_seed_manifest(
    eval_dir: Path,
    *,
    tasks: list[str],
    num_trials: int,
) -> dict[str, dict[str, list[int]]]:
    manifest_path = eval_dir / "seed_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = payload.get("seeds", payload)
        manifest = {
            task: {
                phase: [int(value) for value in raw[task][phase]]
                for phase in ("clean", "random")
            }
            for task in tasks
        }
    else:
        launcher_dir = eval_dir / "launcher_logs"
        manifest: dict[str, dict[str, list[int]]] = {}
        for task in tasks:
            manifest[task] = {}
            for phase in ("clean", "random"):
                candidates = sorted(
                    launcher_dir.glob(f"{task}_{phase}_gpu*_attempt*.log"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                seeds = next(
                    (
                        values[:num_trials]
                        for values in (parse_seeds_from_log(path) for path in candidates)
                        if len(values) >= num_trials
                    ),
                    None,
                )
                if seeds is None:
                    raise FileNotFoundError(
                        f"No complete {num_trials}-trial seed log for {task}/{phase} in {launcher_dir}"
                    )
                manifest[task][phase] = seeds

    for task in tasks:
        for phase in ("clean", "random"):
            seeds = manifest[task][phase]
            if len(seeds) != num_trials:
                raise ValueError(
                    f"Seed manifest has {len(seeds)} seeds for {task}/{phase}; expected {num_trials}."
                )
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"Seed manifest contains duplicate seeds for {task}/{phase}.")
    return manifest


def discover_prior_seed_manifest(
    output_dir: Path,
    *,
    tasks: list[str],
    num_trials: int,
) -> tuple[Path, dict[str, dict[str, list[int]]]] | None:
    candidates = sorted(
        (
            path
            for path in output_dir.parent.glob(f"eval_robotwin_*_trials{num_trials}_step_*")
            if path.resolve() != output_dir.resolve()
        ),
        key=lambda path: path.name,
    )
    for candidate in candidates:
        try:
            return candidate, extract_seed_manifest(
                candidate,
                tasks=tasks,
                num_trials=num_trials,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def write_rate(path: Path, rate: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Instruction Type: unseen\n\n" + f"{rate:.12g}\n",
        encoding="utf-8",
    )


def run_job(
    job: EvalJob,
    *,
    args: argparse.Namespace,
    status_path: Path,
) -> tuple[int, int]:
    task_config = "demo_clean" if job.phase == "clean" else "demo_randomized"
    # Every shard gets an isolated staging directory.  The exact accepted seed
    # list and global episode offset make shards disjoint while preserving the
    # same 50 evaluation scenes used by the original monolithic evaluator.
    job_output_dir = (
        args.output_dir
        / "_job_outputs"
        / job.task_name
        / job.phase
        / job.shard_tag
        / args.checkpoint.stem
    )
    job_expected = result_path(job_output_dir, job.task_name, job.phase)
    seed_list_path = job_output_dir / "eval_seeds.json"
    if job.seeds is not None:
        seed_list_path.parent.mkdir(parents=True, exist_ok=True)
        expected_seeds = list(job.seeds)
        if seed_list_path.is_file():
            existing_seeds = json.loads(seed_list_path.read_text(encoding="utf-8"))
            if existing_seeds != expected_seeds:
                raise RuntimeError(f"Seed shard changed across resume: {seed_list_path}")
        else:
            seed_list_path.write_text(
                json.dumps(expected_seeds, indent=2) + "\n",
                encoding="utf-8",
            )

    def publish_job_outputs() -> tuple[int, int]:
        rate = parse_rate(job_expected)
        successes = success_count_from_rate(rate, job.num_trials)
        source_task_dir = job_output_dir / job.task_name
        phase_output_dir = args.output_dir / job.task_name / job.phase
        phase_output_dir.mkdir(parents=True, exist_ok=True)
        for artifact in source_task_dir.iterdir():
            if artifact == job_expected:
                continue
            target = phase_output_dir / artifact.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(artifact), str(target))
        (job_output_dir / "shard_summary.json").write_text(
            json.dumps(
                {
                    "task_name": job.task_name,
                    "phase": job.phase,
                    "trial_start": job.trial_start,
                    "num_trials": job.num_trials,
                    "successes": successes,
                    "success_rate": rate,
                    "seeds": list(job.seeds) if job.seeds is not None else None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return successes, job.num_trials

    if job_expected.is_file():
        try:
            result = publish_job_outputs()
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"SKIP_COMPLETE task={job.task_name} phase={job.phase} "
                    f"shard={job.shard_tag} gpu={job.gpu_id} "
                    f"successes={result[0]}/{result[1]}\n"
                )
            return result
        except ValueError:
            job_expected.unlink()

    for attempt in range(1, args.max_attempts + 1):
        launcher_log = (
            args.output_dir
            / "launcher_logs"
            / (
                f"{job.task_name}_{job.phase}_{job.shard_tag}_"
                f"gpu{job.gpu_id}_attempt{attempt}.log"
            )
        )
        launcher_log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "EVAL_MODE": "single",
                "CKPT": str(args.checkpoint),
                "GPU_ID": str(job.gpu_id),
                "ROBOTWIN_TASK_NAME": job.task_name,
                "TASK_CONFIG": task_config,
                "EVAL_NUM_EPISODES": str(job.num_trials),
                "EVAL_TOTAL_EPISODES": str(args.num_trials),
                "EVAL_SEED_LIST_PATH": (
                    str(seed_list_path) if job.seeds is not None else "null"
                ),
                "EVAL_EPISODE_OFFSET": str(job.trial_start),
                "OUTPUT_DIR": str(job_output_dir),
                "DATASET_STATS_PATH": str(args.dataset_stats),
                "USE_TRAINING_RUN_CONFIG": "true",
                "REPLAN_STEPS": str(args.replan_steps),
                "INSTRUCTION_TYPE": "unseen",
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
            }
        )
        if job.seed_override is not None:
            # Cold-start shard: give RoboTwin a disjoint starting seed so this
            # shard explores its own accepted-seed interval (no manifest).
            env["SEED_OVERRIDE"] = str(job.seed_override)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"START task={job.task_name} phase={job.phase} gpu={job.gpu_id} "
                f"shard={job.shard_tag} trials={job.num_trials} "
                f"attempt={attempt}/{args.max_attempts}\n"
            )
        with launcher_log.open("a", encoding="utf-8") as log_handle:
            result = subprocess.run(
                ["bash", str(args.eval_script)],
                cwd=str(args.repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds,
                check=False,
            )
        if job_expected.is_file():
            try:
                rate = publish_job_outputs()
                with status_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"COMPLETE task={job.task_name} phase={job.phase} "
                        f"shard={job.shard_tag} gpu={job.gpu_id} "
                        f"successes={rate[0]}/{rate[1]} returncode={result.returncode}\n"
                    )
                return rate
            except ValueError:
                job_expected.unlink()
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"RETRY task={job.task_name} phase={job.phase} gpu={job.gpu_id} "
                f"shard={job.shard_tag} attempt={attempt} "
                f"returncode={result.returncode} log={launcher_log}\n"
            )
        time.sleep(args.retry_sleep_seconds)
    raise RuntimeError(
        f"Evaluation failed for {job.task_name}/{job.phase}/{job.shard_tag} "
        f"after {args.max_attempts} attempts."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="open_microwave,dump_bin_bigbin,press_stapler,place_cans_plasticbox")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument(
        "--trials-per-shard",
        type=int,
        default=10,
        help=(
            "Split each task/condition into exact-seed shards of this size. "
            "Sharding activates once a complete prior seed manifest is available."
        ),
    )
    parser.add_argument(
        "--cold-seed-base",
        type=int,
        default=0,
        help=(
            "Deterministic RoboTwin seed override for the first cold-start "
            "shard (default: 0). RoboTwin maps override s to "
            "100000*(1+s), so the generated value must fit NumPy uint32."
        ),
    )
    parser.add_argument(
        "--cold-seed-stride",
        type=int,
        default=1,
        help="Per-shard cold-start seed-override step (default: 1).",
    )
    parser.add_argument(
        "--seed-source-dir",
        type=Path,
        help="Optional prior complete evaluation directory used to recover exact accepted seeds.",
    )
    parser.add_argument(
        "--max-processes-per-gpu",
        type=int,
        default=2,
        help="Maximum concurrent evaluator processes on each GPU.",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/robotwin2.0/dataset_stats.json",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--eval-script", type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.eval_script = (args.eval_script or (args.repo_root / "scripts" / "eval_robotwin.sh")).resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.dataset_stats = args.dataset_stats.resolve()
    if args.seed_source_dir is not None:
        args.seed_source_dir = args.seed_source_dir.resolve()

    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    gpu_ids = [int(value.strip()) for value in args.gpu_ids.split(",") if value.strip()]
    if not tasks or not gpu_ids:
        raise ValueError(f"At least one task and one GPU are required, got tasks={tasks}, gpu_ids={gpu_ids}")
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if args.max_processes_per_gpu <= 0:
        raise ValueError("--max-processes-per-gpu must be positive")
    if args.trials_per_shard <= 0:
        raise ValueError("--trials-per-shard must be positive")
    if args.cold_seed_base < 0:
        raise ValueError("--cold-seed-base must be non-negative")
    if args.cold_seed_stride <= 0:
        raise ValueError("--cold-seed-stride must be positive")
    cold_shard_count = (args.num_trials + args.trials_per_shard - 1) // args.trials_per_shard
    cold_last_seed = args.cold_seed_base + max(0, cold_shard_count - 1) * args.cold_seed_stride
    if 100000 * (1 + cold_last_seed) > (2**32 - 1):
        raise ValueError(
            "Cold-start seed range exceeds RoboTwin/NumPy uint32 range: "
            f"last override={cold_last_seed}"
        )
    for required in (args.checkpoint, args.dataset_stats, args.eval_script):
        if not required.exists():
            raise FileNotFoundError(required)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "queue_status.log"

    seed_source: Path | None = None
    seed_manifest: dict[str, dict[str, list[int]]] | None = None
    if args.num_trials > args.trials_per_shard:
        if args.seed_source_dir is not None:
            seed_source = args.seed_source_dir
            seed_manifest = extract_seed_manifest(
                seed_source,
                tasks=tasks,
                num_trials=args.num_trials,
            )
        else:
            discovered = discover_prior_seed_manifest(
                args.output_dir,
                tasks=tasks,
                num_trials=args.num_trials,
            )
            if discovered is not None:
                seed_source, seed_manifest = discovered

    if seed_manifest is not None:
        (args.output_dir / "seed_manifest.json").write_text(
            json.dumps(
                {
                    "source_eval_dir": str(seed_source),
                    "num_trials_per_task_per_condition": args.num_trials,
                    "seeds": seed_manifest,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"SHARDING_ENABLED trials_per_shard={args.trials_per_shard} "
                f"seed_source={seed_source}\n"
            )
    else:
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "SHARDING_COLD_START reason=no_complete_prior_seed_manifest "
                f"num_trials={args.num_trials} trials_per_shard={args.trials_per_shard} "
                f"seed_base={args.cold_seed_base} seed_stride={args.cold_seed_stride}\n"
            )

    # Every GPU owns N workers.  Jobs are trial shards rather than whole
    # task/condition pairs, so any free worker can immediately take over work
    # from a long task without duplicating or changing evaluation seeds.
    job_queue: Queue[EvalJob] = Queue()
    result_counts: dict[str, dict[str, list[int]]] = {}
    for task in tasks:
        for phase in ("clean", "random"):
            canonical_result = result_path(args.output_dir, task, phase)
            if canonical_result.is_file():
                try:
                    rate = parse_rate(canonical_result)
                    successes = success_count_from_rate(rate, args.num_trials)
                    result_counts.setdefault(task, {})[phase] = [successes, args.num_trials]
                    with status_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            f"SKIP_CONDITION_COMPLETE task={task} phase={phase} "
                            f"successes={successes}/{args.num_trials}\n"
                        )
                    continue
                except ValueError:
                    canonical_result.unlink()

            if seed_manifest is None:
                # Cold start: no prior accepted-seed manifest exists yet, so
                # every task/condition is split into deterministic seed-range
                # shards.  Each shard starts RoboTwin's accepted-seed search at
                # a disjoint interval, letting the slowest task (e.g.
                # open_microwave) run on many workers instead of one worker
                # blocking the whole queue for hours.
                for shard_idx, start in enumerate(
                    range(0, args.num_trials, args.trials_per_shard)
                ):
                    shard_size = min(
                        args.trials_per_shard, args.num_trials - start
                    )
                    job_queue.put(
                        EvalJob(
                            task_name=task,
                            phase=phase,
                            trial_start=start,
                            num_trials=shard_size,
                            seeds=None,
                            seed_override=(
                                args.cold_seed_base
                                + shard_idx * args.cold_seed_stride
                            ),
                        )
                    )
                continue

            condition_seeds = seed_manifest[task][phase]
            for start in range(0, args.num_trials, args.trials_per_shard):
                shard_seeds = tuple(condition_seeds[start : start + args.trials_per_shard])
                job_queue.put(
                    EvalJob(
                        task_name=task,
                        phase=phase,
                        trial_start=start,
                        num_trials=len(shard_seeds),
                        seeds=shard_seeds,
                    )
                )

    failures: list[str] = []
    results_lock = Lock()

    def gpu_worker(gpu_id: int) -> None:
        while True:
            try:
                queued_job = job_queue.get_nowait()
            except Empty:
                return
            job = replace(queued_job, gpu_id=gpu_id)
            try:
                successes, num_trials = run_job(
                    job,
                    args=args,
                    status_path=status_path,
                )
                with results_lock:
                    condition = result_counts.setdefault(job.task_name, {}).setdefault(
                        job.phase,
                        [0, 0],
                    )
                    condition[0] += successes
                    condition[1] += num_trials
            except Exception as exc:  # noqa: BLE001 - preserve all worker failures in the queue log
                with results_lock:
                    failures.append(
                        f"{job.task_name}/{job.phase}/{job.shard_tag}: {exc}"
                    )
            finally:
                job_queue.task_done()

    # Round-robin worker order fills every GPU once before adding its second
    # resident evaluator.
    worker_gpu_ids = [
        gpu_id
        for _ in range(args.max_processes_per_gpu)
        for gpu_id in gpu_ids
    ]
    queued_jobs = job_queue.qsize()
    if queued_jobs:
        with ThreadPoolExecutor(max_workers=min(queued_jobs, len(worker_gpu_ids))) as executor:
            futures = [executor.submit(gpu_worker, gpu_id) for gpu_id in worker_gpu_ids]
            for future in as_completed(futures):
                future.result()

    if failures:
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write("FAILED " + " | ".join(failures) + "\n")
        raise RuntimeError("; ".join(failures))

    results: dict[str, dict[str, float]] = {}
    for task in tasks:
        results[task] = {}
        for phase in ("clean", "random"):
            successes, evaluated = result_counts[task][phase]
            if evaluated != args.num_trials:
                raise RuntimeError(
                    f"Incomplete aggregate for {task}/{phase}: "
                    f"{evaluated}/{args.num_trials} trials."
                )
            rate = successes / evaluated
            results[task][phase] = rate
            write_rate(result_path(args.output_dir, task, phase), rate)

    clean_mean = sum(results[task]["clean"] for task in tasks) / len(tasks)
    random_mean = sum(results[task]["random"] for task in tasks) / len(tasks)
    overall_mean = (clean_mean + random_mean) / 2.0
    payload = {
        "checkpoint": str(args.checkpoint),
        "num_trials_per_task_per_condition": args.num_trials,
        "trials_per_shard": args.trials_per_shard if seed_manifest is not None else args.num_trials,
        "seed_source_eval_dir": str(seed_source) if seed_source is not None else None,
        "per_task": [
            {
                "task_name": task,
                "clean_success_rate": results[task]["clean"],
                "random_success_rate": results[task]["random"],
                "average_success_rate": (results[task]["clean"] + results[task]["random"]) / 2.0,
            }
            for task in tasks
        ],
        "overall": {
            "clean_mean_success_rate": clean_mean,
            "random_mean_success_rate": random_mean,
            "average_success_rate": overall_mean,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_name", "clean_success_rate", "random_success_rate", "average_success_rate"])
        for row in payload["per_task"]:
            writer.writerow(
                [row["task_name"], row["clean_success_rate"], row["random_success_rate"], row["average_success_rate"]]
            )
        writer.writerow(["__overall__", clean_mean, random_mean, overall_mean])
    with status_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"PIPELINE_COMPLETE clean={clean_mean:.6f} random={random_mean:.6f} overall={overall_mean:.6f}\n"
        )
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
