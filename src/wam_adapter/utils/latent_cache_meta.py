from __future__ import annotations

from typing import Any


def _pick_existing(mapping: dict[str, Any] | None, keys: list[str]) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, Any] = {}
    for key in keys:
        if key in mapping:
            out[key] = mapping[key]
    return out


def _canonical_model_meta(model_cfg: dict[str, Any] | None) -> dict[str, Any]:
    return _pick_existing(
        model_cfg,
        [
            "video_backbone_type",
            "video_backbone_name",
        ],
    )


def _canonical_processor_meta(processor_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(processor_cfg, dict):
        return {}
    out = _pick_existing(
        processor_cfg,
        [
            "_target_",
            "train_transforms",
            "val_transforms",
        ],
    )
    return out


def _canonical_data_train_meta(data_train_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data_train_cfg, dict):
        return {}

    out = _pick_existing(
        data_train_cfg,
        [
            "_target_",
            "dataset_dirs",
            "num_frames",
            "global_sample_stride",
            "action_video_freq_ratio",
            "video_size",
            "camera_key",
            "concat_multi_camera",
        ],
    )

    shape_meta = data_train_cfg.get("shape_meta")
    if isinstance(shape_meta, dict):
        images = shape_meta.get("images")
        if images is not None:
            out["shape_meta"] = {"images": images}

    processor_cfg = _canonical_processor_meta(data_train_cfg.get("processor"))
    if processor_cfg:
        out["processor"] = processor_cfg

    return out


def build_canonical_video_latent_meta_payload(
    *,
    format_name: str,
    storage_format: str,
    video_only: bool,
    shard_size: int | None,
    cache_dtype: str,
    mixed_precision: str | None,
    tiled: bool,
    model_cfg: dict[str, Any] | None,
    data_train_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": format_name,
        "storage_format": storage_format,
        "video_only": video_only,
        "shard_size": shard_size,
        "cache_dtype": cache_dtype,
        "mixed_precision": mixed_precision,
        "tiled": tiled,
    }

    model_meta = _canonical_model_meta(model_cfg)
    if model_meta:
        payload["model"] = model_meta

    data_train_meta = _canonical_data_train_meta(data_train_cfg)
    if data_train_meta:
        payload["data_train"] = data_train_meta

    return payload


def canonicalize_existing_video_latent_meta_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return build_canonical_video_latent_meta_payload(
        format_name=str(payload.get("format")),
        storage_format=str(payload.get("storage_format", "sharded_v1")),
        video_only=bool(payload.get("video_only", True)),
        shard_size=payload.get("shard_size"),
        cache_dtype=str(payload.get("cache_dtype", "unknown")),
        mixed_precision=(None if payload.get("mixed_precision") is None else str(payload.get("mixed_precision"))),
        tiled=bool(payload.get("tiled", False)),
        model_cfg=payload.get("model"),
        data_train_cfg=payload.get("data_train"),
    )
