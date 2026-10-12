#!/usr/bin/env python3
"""Validate every RoboTwin task prompt in a WAM-Adapter text cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


# Single source of truth: keep the prompt template in sync with the training
# dataset and precompute_text_embeds.py (the formal wrapper exports
# PYTHONPATH=src, so this import is available in the official runtime).
from wam_adapter.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


def cache_id(model_id: str) -> str:
    base = str(model_id).rstrip("/").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "", base.lower()) or "textenc"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--expected-hidden-size", type=int, default=4096)
    parser.add_argument("--expected-num-prompts", type=int, default=921_032)
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "force the exhaustive per-entry validation even when the `_SUCCESS` marker "
            "exists (default: sampled check, because the marker is only written after "
            "a successful exhaustive pass)"
        ),
    )
    args = parser.parse_args()

    tasks_path = args.dataset_dir.resolve() / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing RoboTwin task metadata: {tasks_path}")

    prompts: list[str] = []
    seen: set[str] = set()
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "task" not in record:
                raise KeyError(f"Missing task at {tasks_path}:{line_number}")
            prompt = DEFAULT_PROMPT.format(task=str(record["task"]))
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No task prompts found in {tasks_path}")
    if len(prompts) != args.expected_num_prompts:
        raise ValueError(
            f"Prompt-count mismatch: expected {args.expected_num_prompts}, got {len(prompts)}"
        )

    resolved_cache = args.cache_dir.resolve()
    resolved_id = cache_id(args.model_id)

    reuse_marker = resolved_cache / "_SUCCESS"
    quick = reuse_marker.is_file() and not args.full

    def _validate_entry(prompt: str) -> None:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = resolved_cache / f"{digest}.t5_len{args.context_len}.{resolved_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Text cache entry is missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        context = payload.get("context")
        mask = payload.get("mask")
        expected_context = (args.context_len, args.expected_hidden_size)
        if not isinstance(context, torch.Tensor) or tuple(context.shape) != expected_context:
            raise ValueError(
                f"Invalid context in {path}: expected {expected_context}, "
                f"got {getattr(context, 'shape', None)}"
            )
        if context.dtype != torch.bfloat16:
            raise ValueError(f"Invalid context dtype in {path}: {context.dtype}")
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (args.context_len,):
            raise ValueError(f"Invalid mask in {path}: {getattr(mask, 'shape', None)}")
        if mask.dtype != torch.bool:
            raise ValueError(f"Invalid mask dtype in {path}: {mask.dtype}")

    if quick:
        # Fast reuse path: `_SUCCESS` is only written after a successful
        # exhaustive validation of the freshly built cache, so re-loading all
        # 921k entries on every resume only burns ~45 min of GPFS time. Sample
        # deterministically, verify payloads, and require the full entry count;
        # `--full` restores the exhaustive pass for manual audits.
        stride = max(1, len(prompts) // 64)
        sampled = prompts[::stride]
        if prompts[-1] not in sampled:
            sampled.append(prompts[-1])
        for prompt in sampled:
            _validate_entry(prompt)
        num_entries = sum(1 for _ in resolved_cache.iterdir())
        if num_entries < len(prompts):
            raise FileNotFoundError(
                f"Text cache is incomplete: directory holds {num_entries} entries, "
                f"expected at least {len(prompts)} under {resolved_cache}"
            )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "quick",
                    "dataset_dir": str(args.dataset_dir.resolve()),
                    "cache_dir": str(resolved_cache),
                    "model_id": args.model_id,
                    "cache_id": resolved_id,
                    "num_prompts": len(prompts),
                    "num_sampled_entries": len(sampled),
                    "num_directory_entries": num_entries,
                    "context_shape": [args.context_len, args.expected_hidden_size],
                    "dtype": "bfloat16",
                },
                indent=2,
            )
        )
        return

    missing: list[str] = []
    for prompt in prompts:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = resolved_cache / f"{digest}.t5_len{args.context_len}.{resolved_id}.pt"
        if not path.is_file():
            missing.append(str(path))
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        context = payload.get("context")
        mask = payload.get("mask")
        expected_context = (args.context_len, args.expected_hidden_size)
        if not isinstance(context, torch.Tensor) or tuple(context.shape) != expected_context:
            raise ValueError(
                f"Invalid context in {path}: expected {expected_context}, "
                f"got {getattr(context, 'shape', None)}"
            )
        if context.dtype != torch.bfloat16:
            raise ValueError(f"Invalid context dtype in {path}: {context.dtype}")
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (args.context_len,):
            raise ValueError(f"Invalid mask in {path}: {getattr(mask, 'shape', None)}")
        if mask.dtype != torch.bool:
            raise ValueError(f"Invalid mask dtype in {path}: {mask.dtype}")

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"Text cache is incomplete: {len(missing)}/{len(prompts)} prompts missing.\n{preview}"
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_dir": str(args.dataset_dir.resolve()),
                "cache_dir": str(resolved_cache),
                "model_id": args.model_id,
                "cache_id": resolved_id,
                "num_prompts": len(prompts),
                "context_shape": [args.context_len, args.expected_hidden_size],
                "dtype": "bfloat16",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
