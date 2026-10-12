#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


TASK_BLOCK_SIZE = 550
CLEAN_EPISODES = 50
RANDOMIZED_EPISODES = 500
TASK_POSITIONS = {
    "place_cans_plasticbox": 26,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_POSITIONS), default="place_cans_plasticbox")
    args = parser.parse_args()

    episodes_meta = args.dataset_root / "meta" / "episodes.jsonl"
    info_meta = args.dataset_root / "meta" / "info.json"
    if not episodes_meta.is_file() or not info_meta.is_file():
        raise FileNotFoundError(f"Invalid LeRobot dataset root: {args.dataset_root}")

    block_start = TASK_POSITIONS[args.task] * TASK_BLOCK_SIZE
    records = [
        {
            "episode_index": episode_index,
            "task_name": args.task,
            "condition": "clean" if offset < CLEAN_EPISODES else "randomized",
            "task_block_start": block_start,
        }
        for offset, episode_index in enumerate(range(block_start, block_start + TASK_BLOCK_SIZE))
    ]
    manifest = {
        "name": f"robotwin_{args.task}_full550_v1",
        "num_tasks": 1,
        "num_episodes": TASK_BLOCK_SIZE,
        "num_clean_episodes": CLEAN_EPISODES,
        "num_randomized_episodes": RANDOMIZED_EPISODES,
        "tasks": [args.task],
        "mapping_contract": "50 clean followed by 500 randomized episodes per 550-episode task block",
        "source": {
            "dataset_root": str(args.dataset_root.resolve()),
            "info_sha256": sha256(info_meta),
            "episodes_sha256": sha256(episodes_meta),
        },
        "episode_indices": [record["episode_index"] for record in records],
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} episodes for {args.task} to {args.output}")


if __name__ == "__main__":
    main()
