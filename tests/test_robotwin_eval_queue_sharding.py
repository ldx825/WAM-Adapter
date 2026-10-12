from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_SCRIPT = REPO_ROOT / "scripts" / "run_robotwin_fourtask_eval_queue.py"


def load_queue_module():
    spec = importlib.util.spec_from_file_location("robotwin_eval_queue", QUEUE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_seed_log(path: Path, seeds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"current seed: \x1b[90m{seed}\x1b[0m" for seed in seeds) + "\n",
        encoding="utf-8",
    )


def test_extract_seed_manifest_strips_ansi(tmp_path: Path) -> None:
    queue = load_queue_module()
    eval_dir = tmp_path / "eval_robotwin_test_trials3_step_020000"
    write_seed_log(eval_dir / "launcher_logs" / "task_a_clean_gpu0_attempt1.log", [10, 12, 15])
    write_seed_log(eval_dir / "launcher_logs" / "task_a_random_gpu1_attempt1.log", [20, 21, 25])

    manifest = queue.extract_seed_manifest(eval_dir, tasks=["task_a"], num_trials=3)

    assert manifest == {
        "task_a": {
            "clean": [10, 12, 15],
            "random": [20, 21, 25],
        }
    }


def test_sharded_queue_aggregates_exact_trials(tmp_path: Path) -> None:
    source = tmp_path / "run" / "eval_robotwin_test_trials6_step_020000"
    clean_seeds = [100, 102, 105, 109, 110, 114]
    random_seeds = [200, 203, 205, 208, 211, 214]
    write_seed_log(source / "launcher_logs" / "task_a_clean_gpu0_attempt1.log", clean_seeds)
    write_seed_log(source / "launcher_logs" / "task_a_random_gpu1_attempt1.log", random_seeds)

    checkpoint = tmp_path / "step_040000.pt"
    checkpoint.write_bytes(b"checkpoint")
    stats = tmp_path / "dataset_stats.json"
    stats.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "run" / "eval_robotwin_test_trials6_step_040000"
    fake_eval = tmp_path / "fake_eval.sh"
    fake_eval.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUTPUT_DIR"]) / os.environ["ROBOTWIN_TASK_NAME"]
out.mkdir(parents=True, exist_ok=True)
phase = "clean" if os.environ["TASK_CONFIG"] == "demo_clean" else "random"
seeds = json.loads(Path(os.environ["EVAL_SEED_LIST_PATH"]).read_text())
offset = int(os.environ["EVAL_EPISODE_OFFSET"])
successes = 0
for local_index, seed in enumerate(seeds):
    success = seed % 2 == 0
    successes += int(success)
    randomized = phase == "random"
    (out / f"episode{offset + local_index}_randomized-{str(randomized).lower()}_success-{str(success).lower()}.mp4").write_bytes(b"video")
(out / f"_result_{phase}.txt").write_text(f"{successes / len(seeds)}\\n")
PY
""",
        encoding="utf-8",
    )
    fake_eval.chmod(0o755)

    subprocess.run(
        [
            sys.executable,
            str(QUEUE_SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--tasks",
            "task_a",
            "--gpu-ids",
            "0,1",
            "--num-trials",
            "6",
            "--trials-per-shard",
            "2",
            "--max-processes-per-gpu",
            "2",
            "--seed-source-dir",
            str(source),
            "--dataset-stats",
            str(stats),
            "--repo-root",
            str(tmp_path),
            "--eval-script",
            str(fake_eval),
        ],
        check=True,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["trials_per_shard"] == 2
    assert summary["per_task"][0]["clean_success_rate"] == 4 / 6
    assert summary["per_task"][0]["random_success_rate"] == 3 / 6
    for phase in ("clean", "random"):
        videos = sorted((output / "task_a" / phase).glob("*.mp4"))
        assert len(videos) == 6
        assert {path.name.split("_", 1)[0] for path in videos} == {
            f"episode{index}" for index in range(6)
        }
