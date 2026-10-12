#!/usr/bin/env python3
"""Validate an indexed WAM-Adapter video-latent cache before formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--expected-factor", type=int, required=True)
    parser.add_argument("--expected-shape", type=int, nargs=4, required=True)
    parser.add_argument("--expected-num-samples", type=int)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    meta_path = cache_dir / "meta.json"
    index_path = cache_dir / "index.pt"
    if not meta_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"Incomplete cache: expected {meta_path} and {index_path}."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    factor = int(meta.get("latent_spatial_downsample_factor", 1))
    if factor != args.expected_factor:
        raise ValueError(f"Cache factor mismatch: expected {args.expected_factor}, got {factor}.")

    index = torch.load(index_path, map_location="cpu")
    sample_to_shard = index["sample_to_shard"]
    sample_to_offset = index["sample_to_offset"]
    shard_paths = index["shard_paths"]
    if (
        args.expected_num_samples is not None
        and int(sample_to_shard.numel()) != args.expected_num_samples
    ):
        raise ValueError(
            "Cache sample-count mismatch: expected "
            f"{args.expected_num_samples}, got {sample_to_shard.numel()}."
        )
    missing = int(((sample_to_shard < 0) | (sample_to_offset < 0)).sum().item())
    if missing:
        raise ValueError(
            f"Cache index is incomplete: {missing}/{sample_to_shard.numel()} samples missing."
        )
    if not shard_paths:
        raise ValueError("Cache index contains no shard paths.")

    expected_shape = tuple(args.expected_shape)
    for relpath in (shard_paths[0], shard_paths[-1]):
        path = cache_dir / relpath
        payload = torch.load(path, map_location="cpu", mmap=True)
        latents = payload["video_latents"]
        if latents.ndim != 5 or tuple(latents.shape[1:]) != expected_shape:
            raise ValueError(
                f"Unexpected latent shape in {path}: expected [N,{expected_shape}], "
                f"got {tuple(latents.shape)}."
            )
        if latents.dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 cache, got {latents.dtype} in {path}.")

    print(
        json.dumps(
            {
                "status": "ok",
                "cache_dir": str(cache_dir),
                "storage_format": index.get("storage_format"),
                "num_samples": int(sample_to_shard.numel()),
                "num_shards": len(shard_paths),
                "latent_shape": list(expected_shape),
                "latent_spatial_downsample_factor": factor,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
