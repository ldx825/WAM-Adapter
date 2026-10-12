#!/usr/bin/env python3
"""Fast structural validation for the 50-task RoboTwin training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=27_500)
    parser.add_argument("--expected-task-records", type=int, default=921_032)
    parser.add_argument("--episodes-per-benchmark-task", type=int, default=550)
    args = parser.parse_args()

    root = args.dataset_dir.resolve()
    for relative in ("data", "meta", "videos", "meta/info.json", "meta/tasks.jsonl"):
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Incomplete RoboTwin dataset: missing {path}")

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = int(info.get("total_episodes", -1))
    if episodes != args.expected_episodes:
        raise ValueError(
            f"Episode-count mismatch: expected {args.expected_episodes}, got {episodes}"
        )

    task_records = 0
    with (root / "meta" / "tasks.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "task" not in record:
                raise KeyError(f"Missing task at tasks.jsonl:{line_number}")
            task_records += 1
    if task_records != args.expected_task_records:
        raise ValueError(
            f"Task-record mismatch: expected {args.expected_task_records}, got {task_records}"
        )
    if episodes % args.episodes_per_benchmark_task != 0:
        raise ValueError(
            f"{episodes} episodes is not divisible by "
            f"{args.episodes_per_benchmark_task} episodes per benchmark task"
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_dir": str(root),
                "episodes": episodes,
                "benchmark_tasks": episodes // args.episodes_per_benchmark_task,
                "episodes_per_benchmark_task": args.episodes_per_benchmark_task,
                "text_task_records": task_records,
                "total_frames": int(info.get("total_frames", -1)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
