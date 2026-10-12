import gc
import json
import inspect
import logging
import os
import time
import uuid
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from wam_adapter.runtime import (
    create_wan22_model,
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
)
from wam_adapter.utils.latent_cache_meta import build_canonical_video_latent_meta_payload
from wam_adapter.utils.config_resolvers import register_default_resolvers
from wam_adapter.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = str(os.environ.get("WAM_ADAPTER_PRECOMPUTE_DIST_BACKEND", "gloo")).strip().lower()
    if backend not in {"gloo", "nccl"}:
        raise ValueError(
            f"Unsupported WAM_ADAPTER_PRECOMPUTE_DIST_BACKEND={backend}. Expected one of ['gloo', 'nccl']."
        )
    if backend == "nccl" and not torch.cuda.is_available():
        raise ValueError("WAM_ADAPTER_PRECOMPUTE_DIST_BACKEND=nccl requires CUDA to be available.")
    timeout_minutes = int(os.environ.get("WAM_ADAPTER_PRECOMPUTE_DIST_TIMEOUT_MIN", "720"))
    if not dist.is_initialized():
        init_kwargs = {
            "backend": backend,
            "init_method": "env://",
            "timeout": timedelta(minutes=max(timeout_minutes, 1)),
        }
        if backend == "nccl" and "device_id" in inspect.signature(dist.init_process_group).parameters:
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(**init_kwargs)
    return True, dist.get_rank(), dist.get_world_size()


def _atomic_torch_save(payload: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _atomic_json_dump(payload: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def _load_existing_sharded_index(index_path: Path) -> dict | None:
    if not index_path.exists():
        return None
    payload = torch.load(str(index_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict latent cache index at {index_path}, got {type(payload)}.")
    return payload


def _torch_load_cpu_maybe_mmap(path: Path) -> dict:
    load_kwargs = {"map_location": "cpu"}
    if "mmap" in inspect.signature(torch.load).parameters:
        load_kwargs["mmap"] = True
    payload = torch.load(str(path), **load_kwargs)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict cache payload at {path}, got {type(payload)}.")
    return payload


def _to_plain_dict(cfg_node):
    return OmegaConf.to_container(cfg_node, resolve=True)


def _resolve_precompute_cache_dtype(
    cache_dtype_cfg: str | None,
    model_dtype: torch.dtype,
) -> torch.dtype:
    if cache_dtype_cfg in (None, "", "model"):
        return model_dtype

    key = str(cache_dtype_cfg).strip().lower()
    if key in {"fp32", "float32"}:
        return torch.float32
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(
        f"Unsupported `precompute_cache_dtype`: {cache_dtype_cfg}. "
        "Expected one of: ['model', 'fp32', 'fp16', 'bf16']."
    )


def _resolve_precompute_storage_format(cfg: DictConfig) -> str:
    storage_format_cfg = cfg.get("precompute_storage_format")
    if storage_format_cfg in (None, "", "null"):
        return "sharded_v1" if bool(cfg.get("precompute_use_sharded_cache", False)) else "single_file_v1"

    storage_format = str(storage_format_cfg).strip().lower()
    allowed_formats = {"single_file_v1", "sharded_v1", "episode_packed_v1"}
    if storage_format not in allowed_formats:
        raise ValueError(
            f"Unsupported `precompute_storage_format`: {storage_format_cfg}. "
            f"Expected one of: {sorted(allowed_formats)}."
        )
    return storage_format


def _downsample_latents_for_cache(
    latents: torch.Tensor,
    factor: int,
) -> torch.Tensor:
    """Apply the same deterministic spatial compression used by WAMAdapter."""
    if factor == 1:
        return latents
    if factor < 1:
        raise ValueError(
            "`precompute_latent_spatial_downsample_factor` must be >= 1, "
            f"got {factor}."
        )
    if latents.ndim != 5:
        raise ValueError(
            f"Expected VAE latents [B,C,T,H,W], got {tuple(latents.shape)}."
        )
    height, width = map(int, latents.shape[-2:])
    if height % factor or width % factor:
        raise ValueError(
            f"Latent HxW=({height},{width}) is not divisible by factor={factor}."
        )
    return F.avg_pool3d(
        latents,
        kernel_size=(1, factor, factor),
        stride=(1, factor, factor),
    )


def _timing_sync(enabled: bool, sync_cuda: bool, device: str | torch.device):
    if not enabled:
        return
    device = torch.device(device)
    if sync_cuda and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _timing_start(enabled: bool, sync_cuda: bool, device: str | torch.device):
    if not enabled:
        return None
    _timing_sync(enabled=enabled, sync_cuda=sync_cuda, device=device)
    return time.perf_counter()


def _timing_end_ms(
    start_time,
    *,
    enabled: bool,
    sync_cuda: bool,
    device: str | torch.device,
) -> float:
    if start_time is None:
        return 0.0
    _timing_sync(enabled=enabled, sync_cuda=sync_cuda, device=device)
    return float((time.perf_counter() - start_time) * 1000.0)


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except ValueError:
        logger.warning("Invalid integer for env %s=%r; falling back to %d.", name, raw, default)
        return int(default)


def _read_int_file(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _detect_cgroup_memory_dirs() -> list[Path]:
    dirs: list[Path] = []
    cgroup_root = Path("/sys/fs/cgroup")
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(":", 2)
            if len(parts) != 3:
                continue
            relative = parts[2].strip()
            candidates = [cgroup_root] if relative in ("", "/") else [cgroup_root / relative.lstrip("/"), cgroup_root]
            for candidate in candidates:
                if candidate not in dirs:
                    dirs.append(candidate)
    except OSError:
        pass
    if cgroup_root not in dirs:
        dirs.append(cgroup_root)
    return dirs


def _collect_memory_snapshot() -> dict:
    """Best-effort host memory snapshot: /proc/self/status, cgroup v2/v1 counters, /proc/meminfo."""
    snapshot: dict = {}

    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "RssAnon:", "RssFile:", "RssShmem:", "VmSize:", "VmSwap:")):
                key, value = line.split(":", 1)
                snapshot[key.strip().lower()] = int(value.strip().split()[0])  # kB
    except OSError:
        pass

    try:
        import psutil  # type: ignore

        snapshot["uss_kb"] = int(psutil.Process(os.getpid()).memory_full_info().uss // 1024)
    except Exception:
        snapshot["uss_kb"] = None

    for cgroup_dir in _detect_cgroup_memory_dirs():
        if not cgroup_dir.is_dir():
            continue
        current = _read_int_file(cgroup_dir / "memory.current")
        limit = _read_int_file(cgroup_dir / "memory.max")
        if current is None:
            current = _read_int_file(cgroup_dir / "memory.usage_in_bytes")
            limit = _read_int_file(cgroup_dir / "memory.limit_in_bytes")
        if current is None:
            continue
        snapshot.setdefault("cgroup_current_bytes", current)
        snapshot.setdefault("cgroup_max_bytes", limit)
        snapshot.setdefault("cgroup_dir", str(cgroup_dir))
        try:
            for line in (cgroup_dir / "memory.stat").read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if (
                    len(parts) == 2
                    and parts[0]
                    in {
                        "anon",
                        "file",
                        "shmem",
                        "kernel",
                        "kernel_stack",
                        "slab",
                        "sock",
                        "inactive_file",
                        "active_file",
                        "inactive_anon",
                        "active_anon",
                    }
                ):
                    snapshot.setdefault(f"cgroup_{parts[0]}_bytes", int(parts[1]))
        except (OSError, ValueError):
            pass
        break

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            parts = line.replace(":", " ").split()
            if len(parts) >= 2 and parts[0] in {"MemAvailable", "MemFree", "Cached", "Buffers", "Shmem", "Slab"}:
                snapshot.setdefault(f"meminfo_{parts[0].lower()}_kb", int(parts[1]))
    except (OSError, ValueError):
        pass

    return snapshot


class _EpisodeMemoryLogger:
    """Env-gated host-memory logger for the episode-packed precompute loop.

    Env knobs (all optional, diagnostics only):
      PRECOMPUTE_MEM_LOG_EVERY   log one snapshot every N episodes (0 / unset = disabled)
      PRECOMPUTE_MEM_LOG_DIR     optional directory for per-rank JSONL dumps
    """

    def __init__(self, *, rank: int):
        self.rank = int(rank)
        self.every = _env_int("PRECOMPUTE_MEM_LOG_EVERY", 0)
        self.enabled = self.every > 0
        if self.enabled:
            self.every = max(1, self.every)
        log_dir_raw = os.environ.get("PRECOMPUTE_MEM_LOG_DIR", "").strip()
        self.jsonl_path = None
        if log_dir_raw:
            log_dir = Path(log_dir_raw).expanduser()
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                self.jsonl_path = log_dir / f"rank{self.rank:03d}_mem.jsonl"
            except OSError as exc:
                logger.warning(
                    "Cannot create PRECOMPUTE_MEM_LOG_DIR=%s (%s); JSONL logging disabled.",
                    log_dir_raw,
                    exc,
                )

    @staticmethod
    def _mb_from_kb(value):
        return None if value is None else round(float(value) / 1024.0, 1)

    @staticmethod
    def _mb_from_bytes(value):
        return None if value is None else round(float(value) / 1048576.0, 1)

    def log(self, *, tag: str, episode_idx: int, step_idx: int, total_steps: int):
        if not self.enabled:
            return
        snapshot = _collect_memory_snapshot()
        record = {
            "ts": time.time(),
            "tag": tag,
            "rank": self.rank,
            "pid": os.getpid(),
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
            "total_steps": int(total_steps),
            **snapshot,
        }
        logger.info(
            "[precompute_mem][rank=%d] tag=%s step=%d/%d episode_idx=%d pid=%d "
            "rss_mb=%s rss_anon_mb=%s rss_file_mb=%s rss_shmem_mb=%s uss_mb=%s "
            "cgroup_current_mb=%s cgroup_max_mb=%s cgroup_anon_mb=%s cgroup_file_mb=%s "
            "cgroup_shmem_mb=%s cgroup_kernel_mb=%s cgroup_slab_mb=%s "
            "cgroup_inactive_file_mb=%s cgroup_active_file_mb=%s",
            self.rank,
            tag,
            step_idx,
            total_steps,
            episode_idx,
            os.getpid(),
            self._mb_from_kb(snapshot.get("vmrss")),
            self._mb_from_kb(snapshot.get("rssanon")),
            self._mb_from_kb(snapshot.get("rssfile")),
            self._mb_from_kb(snapshot.get("rssshmem")),
            self._mb_from_kb(snapshot.get("uss_kb")),
            self._mb_from_bytes(snapshot.get("cgroup_current_bytes")),
            self._mb_from_bytes(snapshot.get("cgroup_max_bytes")),
            self._mb_from_bytes(snapshot.get("cgroup_anon_bytes")),
            self._mb_from_bytes(snapshot.get("cgroup_file_bytes")),
            self._mb_from_bytes(snapshot.get("cgroup_shmem_bytes")),
            self._mb_from_bytes(snapshot.get("cgroup_kernel_bytes")),
            self._mb_from_bytes(snapshot.get("cgroup_slab_bytes")),
            self._mb_from_bytes(snapshot.get("cgroup_inactive_file_bytes")),
            self._mb_from_bytes(snapshot.get("cgroup_active_file_bytes")),
        )
        if self.jsonl_path is not None:
            try:
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError as exc:
                logger.warning("Failed to append memory record to %s (%s).", self.jsonl_path, exc)


def _make_rank_progress_bar(*, total: int, rank: int, desc: str):
    return tqdm(
        total=total,
        desc=f"{desc}[rank{rank}]",
        position=rank,
        leave=True,
        dynamic_ncols=True,
        disable=False,
    )


def _build_video_only_model(model_cfg: DictConfig, model_dtype: torch.dtype, device: str):
    video_scheduler = model_cfg.get("video_scheduler")
    if video_scheduler is None:
        video_scheduler = {}
    elif isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)

    return create_wan22_model(
        model_id=str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
        tokenizer_model_id=str(model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")),
        dit_config=model_cfg.video_dit_config,
        video_backbone_type=str(model_cfg.get("video_backbone_type", "wan2_2_ti2v")),
        video_backbone_name=model_cfg.get("video_backbone_name"),
        tokenizer_max_len=int(model_cfg.get("tokenizer_max_len", 512)),
        train_shift=float(video_scheduler.get("train_shift", 5.0)),
        infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
        load_text_encoder=False,
        model_dtype=model_dtype,
        device=device,
    )


def _merge_indexed_cache_manifests(
    *,
    latent_cache_dir: Path,
    index_path: Path,
    manifests_dir: Path,
    world_size: int,
    total_samples: int,
    storage_format: str,
    overwrite: bool,
    existing_sample_to_shard: torch.Tensor | None,
    existing_sample_to_offset: torch.Tensor | None,
    existing_shard_paths: list[str],
):
    if overwrite or existing_sample_to_shard is None:
        merged_sample_to_shard = torch.full((total_samples,), -1, dtype=torch.int32)
        merged_sample_to_offset = torch.full((total_samples,), -1, dtype=torch.int32)
        merged_shard_paths = []
    else:
        merged_sample_to_shard = existing_sample_to_shard.to(dtype=torch.int32).clone()
        merged_sample_to_offset = existing_sample_to_offset.to(dtype=torch.int32).clone()
        merged_shard_paths = list(existing_shard_paths)

    for manifest_rank in range(world_size):
        manifest_path = manifests_dir / f"rank{manifest_rank:03d}.pt"
        if not manifest_path.exists():
            continue
        manifest_payload = torch.load(str(manifest_path), map_location="cpu")
        if not isinstance(manifest_payload, dict):
            raise TypeError(f"Expected dict manifest at {manifest_path}, got {type(manifest_payload)}.")

        shard_paths = [str(path) for path in manifest_payload.get("shard_paths", [])]
        sample_indices = manifest_payload.get("sample_indices")
        local_shard_ids = manifest_payload.get("local_shard_ids")
        offsets = manifest_payload.get("offsets")
        if not shard_paths:
            continue
        if (
            not isinstance(sample_indices, torch.Tensor)
            or not isinstance(local_shard_ids, torch.Tensor)
            or not isinstance(offsets, torch.Tensor)
        ):
            raise TypeError(f"Invalid manifest contents in {manifest_path}.")

        base_shard_id = len(merged_shard_paths)
        merged_shard_paths.extend(shard_paths)
        global_shard_ids = local_shard_ids.to(dtype=torch.int32) + int(base_shard_id)
        merged_sample_to_shard[sample_indices.to(dtype=torch.int64)] = global_shard_ids
        merged_sample_to_offset[sample_indices.to(dtype=torch.int64)] = offsets.to(dtype=torch.int32)

    index_payload = {
        "format": "wam_adapter_video_latent_cache_index_v1",
        "storage_format": storage_format,
        "num_samples": int(total_samples),
        "sample_to_shard": merged_sample_to_shard,
        "sample_to_offset": merged_sample_to_offset,
        "shard_paths": merged_shard_paths,
    }
    _atomic_torch_save(index_payload, index_path)


def _list_existing_cache_files(storage_dir: Path) -> list[Path]:
    if not storage_dir.exists():
        return []
    if not storage_dir.is_dir():
        raise NotADirectoryError(f"Expected cache storage directory, got file: {storage_dir}")

    paths = []
    ignored_tmp_count = 0
    for path in storage_dir.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith(".") or ".tmp." in path.name:
            ignored_tmp_count += 1
            continue
        if path.suffix != ".pt":
            continue
        paths.append(path)
    if ignored_tmp_count:
        logger.warning(
            "Ignoring %d temporary latent cache file(s) under %s.",
            ignored_tmp_count,
            storage_dir,
        )
    return sorted(paths, key=lambda p: p.name)


def _validate_index_payload(
    *,
    index_payload: dict,
    index_path: Path,
    total_samples: int,
    storage_format: str,
):
    if str(index_payload.get("storage_format")) != storage_format:
        raise ValueError(
            f"Latent cache index storage format mismatch in {index_path}: "
            f"expected {storage_format}, got {index_payload.get('storage_format')}"
        )
    sample_to_shard = index_payload.get("sample_to_shard")
    sample_to_offset = index_payload.get("sample_to_offset")
    shard_paths = index_payload.get("shard_paths")
    if not isinstance(sample_to_shard, torch.Tensor) or not isinstance(sample_to_offset, torch.Tensor):
        raise TypeError(f"Invalid latent cache index tensors in {index_path}.")
    if not isinstance(shard_paths, list):
        raise TypeError(f"Invalid latent cache index shard_paths in {index_path}.")
    if int(sample_to_shard.numel()) != int(total_samples) or int(sample_to_offset.numel()) != int(total_samples):
        raise ValueError(
            f"Latent cache index sample count mismatch in {index_path}: "
            f"expected {total_samples}, got {sample_to_shard.numel()} and {sample_to_offset.numel()}."
        )


def _rebuild_index_from_cache_files(
    *,
    latent_cache_dir: Path,
    index_path: Path,
    storage_dir: Path,
    total_samples: int,
    storage_format: str,
) -> dict:
    if storage_format not in {"sharded_v1", "episode_packed_v1"}:
        raise ValueError(f"Cannot rebuild an index for storage_format={storage_format}.")

    cache_files = _list_existing_cache_files(storage_dir)
    sample_to_shard = torch.full((total_samples,), -1, dtype=torch.int32)
    sample_to_offset = torch.full((total_samples,), -1, dtype=torch.int32)
    shard_paths: list[str] = []
    recovered_samples = 0

    progress = tqdm(
        cache_files,
        total=len(cache_files),
        desc="rebuild_latent_cache_index",
        unit="shard",
    )
    for shard_idx, cache_path in enumerate(progress, start=1):
        relpath = cache_path.relative_to(latent_cache_dir).as_posix()
        try:
            payload = _torch_load_cpu_maybe_mmap(cache_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load latent cache shard while rebuilding index: {cache_path}. "
                "This file is not a valid completed torch.save payload. "
                "Remove or rename it, or rerun with `overwrite=true`."
            ) from exc

        sample_indices = payload.get("sample_indices")
        video_latents = payload.get("video_latents")
        if not isinstance(sample_indices, torch.Tensor):
            raise TypeError(f"`sample_indices` must be a tensor in cache shard: {cache_path}")
        if not isinstance(video_latents, torch.Tensor):
            raise TypeError(f"`video_latents` must be a tensor in cache shard: {cache_path}")
        if video_latents.ndim != 5:
            raise ValueError(
                f"`video_latents` must be 5D [N,C,T,H,W] in cache shard {cache_path}, "
                f"got shape {tuple(video_latents.shape)}."
            )

        sample_indices = sample_indices.to(dtype=torch.int64, device="cpu").view(-1)
        num_entries = int(sample_indices.numel())
        if num_entries == 0:
            logger.warning("Skipping empty latent cache shard while rebuilding index: %s", cache_path)
            continue
        if int(video_latents.shape[0]) != num_entries:
            raise ValueError(
                f"Shard entry count mismatch in {cache_path}: "
                f"sample_indices={num_entries}, video_latents={video_latents.shape[0]}."
            )
        if bool((sample_indices < 0).any().item()) or bool((sample_indices >= int(total_samples)).any().item()):
            raise ValueError(
                f"Shard {cache_path} contains sample index outside [0, {total_samples})."
            )
        if int(sample_indices.unique().numel()) != num_entries:
            raise ValueError(f"Shard {cache_path} contains duplicate sample indices.")

        shard_id = len(shard_paths)
        existing = sample_to_shard[sample_indices]
        duplicate_positions = torch.nonzero(existing >= 0, as_tuple=False).view(-1)
        if duplicate_positions.numel() > 0:
            duplicate_sample = int(sample_indices[int(duplicate_positions[0].item())].item())
            raise ValueError(
                f"Duplicate latent cache entry for sample idx={duplicate_sample} while scanning {cache_path}. "
                "Remove duplicate shards before resuming to avoid ambiguous training data."
            )

        offsets = torch.arange(num_entries, dtype=torch.int32)
        sample_to_shard[sample_indices] = int(shard_id)
        sample_to_offset[sample_indices] = offsets
        shard_paths.append(relpath)
        recovered_samples += num_entries
        progress.set_postfix(
            recovered_samples=recovered_samples,
            covered=f"{recovered_samples}/{total_samples}",
        )

    index_payload = {
        "format": "wam_adapter_video_latent_cache_index_v1",
        "storage_format": storage_format,
        "num_samples": int(total_samples),
        "sample_to_shard": sample_to_shard,
        "sample_to_offset": sample_to_offset,
        "shard_paths": shard_paths,
    }
    _atomic_torch_save(index_payload, index_path)
    logger.info(
        "Rebuilt latent cache index from %d %s file(s): recovered_samples=%d total_samples=%d index=%s",
        len(shard_paths),
        storage_format,
        recovered_samples,
        total_samples,
        index_path,
    )
    return index_payload


def _extract_existing_index_state(
    *,
    existing_index: dict,
    index_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    existing_sample_to_shard = existing_index.get("sample_to_shard")
    existing_sample_to_offset = existing_index.get("sample_to_offset")
    existing_shard_paths = [str(path) for path in existing_index.get("shard_paths", [])]
    if not isinstance(existing_sample_to_shard, torch.Tensor) or not isinstance(
        existing_sample_to_offset, torch.Tensor
    ):
        raise TypeError(f"Invalid latent cache index in {index_path}.")
    existing_sample_to_shard = existing_sample_to_shard.to(
        dtype=torch.int64, device="cpu"
    ).contiguous()
    existing_sample_to_offset = existing_sample_to_offset.to(
        dtype=torch.int64, device="cpu"
    ).contiguous()
    return existing_sample_to_shard, existing_sample_to_offset, existing_shard_paths


def _wait_for_index_ready(
    *,
    index_path: Path,
    total_samples: int,
    storage_format: str,
    rebuild_marker_path: Path | None = None,
    poll_interval_sec: float = 2.0,
    timeout_sec: float = 0.0,
):
    start_time = time.perf_counter()
    while True:
        if rebuild_marker_path is not None and rebuild_marker_path.exists():
            if timeout_sec > 0 and (time.perf_counter() - start_time) > timeout_sec:
                raise TimeoutError(
                    f"Timed out waiting for latent cache index rebuild marker to clear: {rebuild_marker_path}"
                )
            time.sleep(poll_interval_sec)
            continue

        if index_path.exists():
            try:
                index_payload = _load_existing_sharded_index(index_path)
                if index_payload is not None:
                    _validate_index_payload(
                        index_payload=index_payload,
                        index_path=index_path,
                        total_samples=total_samples,
                        storage_format=storage_format,
                    )
                    return
            except Exception:
                # Rank 0 may still be atomically replacing / finalizing the file.
                pass

        if timeout_sec > 0 and (time.perf_counter() - start_time) > timeout_sec:
            raise TimeoutError(
                f"Timed out waiting for latent cache index to become ready: {index_path}"
            )
        time.sleep(poll_interval_sec)


def _next_local_storage_id_from_existing_paths(
    *,
    existing_paths: list[str],
    storage_format: str,
    rank: int,
) -> int:
    if storage_format == "sharded_v1":
        prefix = f"shards/rank{rank:03d}_shard"
    else:
        prefix = f"episodes/rank{rank:03d}_"

    max_id = -1
    for path in existing_paths:
        path = str(path)
        if not path.startswith(prefix) or not path.endswith(".pt"):
            continue
        stem = Path(path).stem
        try:
            storage_id = int(stem.rsplit("_", 1)[-1].replace("shard", ""))
        except ValueError:
            continue
        max_id = max(max_id, storage_id)
    return max_id + 1


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    if _env_int("PRECOMPUTE_FAULTHANDLER", 0) > 0:
        try:
            import faulthandler
            import signal as _signal

            faulthandler.register(_signal.SIGUSR1, all_threads=True, chain=False)
            logger.info("faulthandler armed: `kill -USR1 %d` dumps all Python stacks", os.getpid())
        except Exception as exc:  # pragma: no cover - platform specific
            logger.warning("failed to arm faulthandler: %s", exc)

    is_distributed, rank, world_size = _init_distributed()
    if is_distributed and rank == 0:
        logger.info(
            "Distributed latent precompute enabled: world_size=%d backend=%s timeout_min=%s",
            world_size,
            dist.get_backend(),
            os.environ.get("WAM_ADAPTER_PRECOMPUTE_DIST_TIMEOUT_MIN", "720"),
        )

    if cfg.model is None:
        raise ValueError("`cfg.model` is required.")
    if cfg.data is None or cfg.data.get("train") is None:
        raise ValueError("`cfg.data.train` is required.")

    dataset_cfg = OmegaConf.create(_to_plain_dict(cfg.data.train))
    latent_cache_dir = dataset_cfg.get("latent_cache_dir")
    if latent_cache_dir is None or str(latent_cache_dir).strip() == "":
        raise ValueError(
            "`data.train.latent_cache_dir` must be set before running precompute_video_latents.py."
        )
    latent_cache_dir = Path(str(latent_cache_dir)).expanduser().resolve()
    storage_format = _resolve_precompute_storage_format(cfg)
    indexed_storage = storage_format in {"sharded_v1", "episode_packed_v1"}
    precompute_video_only = bool(cfg.get("precompute_video_only", False))
    if storage_format == "episode_packed_v1":
        precompute_video_only = True
    shard_size = int(cfg.get("precompute_shard_size", 128))
    precompute_timing_cfg = cfg.get("precompute_timing", {})
    precompute_timing_enabled = bool(precompute_timing_cfg.get("enabled", False))
    precompute_timing_sync_cuda = bool(precompute_timing_cfg.get("sync_cuda", True))
    precompute_timing_log_every = int(precompute_timing_cfg.get("log_every", 10))
    if storage_format == "sharded_v1" and shard_size <= 0:
        raise ValueError(f"`precompute_shard_size` must be positive, got {shard_size}.")

    dataset_cfg["use_latent_cache"] = False
    dataset_cfg["latent_cache_dir"] = None
    dataset_cfg["video_only"] = precompute_video_only
    dataset = instantiate(dataset_cfg)
    total_samples = len(dataset)
    total_episodes = int(dataset.get_num_episodes()) if hasattr(dataset, "get_num_episodes") else 0

    precompute_batch_size = int(cfg.get("precompute_batch_size", cfg.batch_size))
    precompute_num_workers = int(cfg.get("precompute_num_workers", cfg.num_workers))
    overwrite = bool(cfg.get("overwrite", False))
    precompute_resume = bool(cfg.get("precompute_resume", False))
    if overwrite and precompute_resume:
        raise ValueError("`precompute_resume=true` is incompatible with `overwrite=true`.")
    tiled = bool(cfg.get("precompute_tiled", False))
    index_path = latent_cache_dir / "index.pt"
    manifests_dir = latent_cache_dir / "manifests"
    storage_dir_name = ("shards" if storage_format == "sharded_v1" else "episodes")
    storage_dir = latent_cache_dir / storage_dir_name
    rebuild_marker_path = latent_cache_dir / ".index_rebuild_in_progress.json"
    existing_index = None
    existing_sample_to_shard = None
    existing_sample_to_offset = None
    existing_shard_paths = []
    if indexed_storage and not overwrite:
        if precompute_resume and storage_dir.exists():
            if is_distributed:
                if rank == 0:
                    _atomic_json_dump(
                        {
                            "status": "rebuilding",
                            "pid": int(os.getpid()),
                            "started_at": time.time(),
                        },
                        rebuild_marker_path,
                    )
                dist.barrier()
            if rank == 0:
                logger.info(
                    "Rebuilding latent cache index from storage before resume: storage_format=%s storage_dir=%s",
                    storage_format,
                    storage_dir,
                )
                try:
                    _rebuild_index_from_cache_files(
                        latent_cache_dir=latent_cache_dir,
                        index_path=index_path,
                        storage_dir=storage_dir,
                        total_samples=total_samples,
                        storage_format=storage_format,
                    )
                finally:
                    if rebuild_marker_path.exists():
                        rebuild_marker_path.unlink()
            elif is_distributed:
                _wait_for_index_ready(
                    index_path=index_path,
                    total_samples=total_samples,
                    storage_format=storage_format,
                    rebuild_marker_path=rebuild_marker_path,
                )
        else:
            existing_index = _load_existing_sharded_index(index_path)
            if existing_index is None and (manifests_dir.exists() or storage_dir.exists()):
                if not precompute_resume:
                    raise RuntimeError(
                        f"Found partial indexed latent cache data in {latent_cache_dir} without {index_path}. "
                        "Re-run with `precompute_resume=true`, `overwrite=true`, or clean the cache directory."
                    )
                if is_distributed:
                    if rank == 0:
                        _atomic_json_dump(
                            {
                                "status": "rebuilding",
                                "pid": int(os.getpid()),
                                "started_at": time.time(),
                            },
                            rebuild_marker_path,
                        )
                    dist.barrier()
                if rank == 0:
                    logger.info(
                        "Recovering partial latent cache before resume: storage_format=%s storage_dir=%s",
                        storage_format,
                        storage_dir,
                    )
                    try:
                        _rebuild_index_from_cache_files(
                            latent_cache_dir=latent_cache_dir,
                            index_path=index_path,
                            storage_dir=storage_dir,
                            total_samples=total_samples,
                            storage_format=storage_format,
                        )
                    finally:
                        if rebuild_marker_path.exists():
                            rebuild_marker_path.unlink()
                elif is_distributed:
                    _wait_for_index_ready(
                        index_path=index_path,
                        total_samples=total_samples,
                        storage_format=storage_format,
                        rebuild_marker_path=rebuild_marker_path,
                    )
                existing_index = _load_existing_sharded_index(index_path)
                if existing_index is None:
                    raise RuntimeError(f"Failed to rebuild latent cache index: {index_path}")
        existing_index = _load_existing_sharded_index(index_path)
        if existing_index is not None:
            _validate_index_payload(
                index_payload=existing_index,
                index_path=index_path,
                total_samples=total_samples,
                storage_format=storage_format,
            )
            existing_sample_to_shard, existing_sample_to_offset, existing_shard_paths = (
                _extract_existing_index_state(
                    existing_index=existing_index,
                    index_path=index_path,
                )
            )

    prefiltered_skipped_count = 0
    if storage_format != "episode_packed_v1":
        if is_distributed:
            precompute_indices = list(range(rank, total_samples, world_size))
        else:
            precompute_indices = list(range(total_samples))

        if not overwrite:
            original_count = len(precompute_indices)
            if storage_format == "sharded_v1" and existing_sample_to_shard is not None:
                precompute_indices = [
                    sample_idx
                    for sample_idx in precompute_indices
                    if int(existing_sample_to_shard[sample_idx].item()) < 0
                ]
            elif storage_format == "single_file_v1":
                precompute_indices = [
                    sample_idx
                    for sample_idx in precompute_indices
                    if not (latent_cache_dir / f"{sample_idx:08d}.pt").exists()
                ]
            prefiltered_skipped_count = original_count - len(precompute_indices)

        if is_distributed or prefiltered_skipped_count > 0:
            dataset = Subset(dataset, precompute_indices)

    loader = None
    if storage_format != "episode_packed_v1":
        loader = DataLoader(
            dataset,
            batch_size=precompute_batch_size,
            shuffle=False,
            num_workers=precompute_num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=precompute_num_workers > 0,
        )

    device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    cache_dtype = _resolve_precompute_cache_dtype(
        cfg.get("precompute_cache_dtype", "model"),
        model_dtype=model_dtype,
    )
    latent_spatial_downsample_factor = int(
        cfg.get("precompute_latent_spatial_downsample_factor", 1)
    )
    if latent_spatial_downsample_factor < 1:
        raise ValueError(
            "`precompute_latent_spatial_downsample_factor` must be >= 1, "
            f"got {latent_spatial_downsample_factor}."
        )
    model = _build_video_only_model(cfg.model, model_dtype=model_dtype, device=device)
    model.eval()
    if rank == 0:
        logger.info(
            "Precompute setup: storage_format=%s video_only=%s batch_size=%d workers=%d cache_dtype=%s latent_spatial_downsample_factor=%d",
            storage_format,
            precompute_video_only,
            precompute_batch_size,
            precompute_num_workers,
            str(cache_dtype).replace("torch.", ""),
            latent_spatial_downsample_factor,
        )
        if storage_format != "episode_packed_v1":
            logger.info(
                "Precompute index filter: rank=%d pending_samples=%d skipped_existing=%d",
                rank,
                len(dataset),
                prefiltered_skipped_count,
            )
        logger.info(
            "Precompute timing: enabled=%s sync_cuda=%s log_every=%d",
            precompute_timing_enabled,
            precompute_timing_sync_cuda,
            precompute_timing_log_every,
        )
        if storage_format == "episode_packed_v1":
            logger.info(
                "Episode-packed cache enabled: per-rank processing reuses full-episode decode and "
                "writes one indexed cache file per episode-like chunk. `precompute_num_workers` is "
                "not used in this mode."
            )

    if rank == 0:
        format_name = {
            "single_file_v1": "wam_adapter_video_latent_cache_v1",
            "sharded_v1": "wam_adapter_video_latent_cache_sharded_v1",
            "episode_packed_v1": "wam_adapter_video_latent_cache_episode_packed_v1",
        }[storage_format]
        meta = build_canonical_video_latent_meta_payload(
            format_name=format_name,
            storage_format=storage_format,
            video_only=precompute_video_only,
            shard_size=(shard_size if storage_format == "sharded_v1" else None),
            cache_dtype=str(cache_dtype).replace("torch.", ""),
            mixed_precision=mixed_precision,
            tiled=tiled,
            model_cfg=_to_plain_dict(cfg.model),
            data_train_cfg=_to_plain_dict(cfg.data.train),
        )
        meta["latent_spatial_downsample_factor"] = latent_spatial_downsample_factor
        _atomic_json_dump(meta, latent_cache_dir / "meta.json")

    saved_count = 0
    skipped_count = int(prefiltered_skipped_count)
    pending_sample_indices: list[int] = []
    pending_latents: list[torch.Tensor] = []
    local_manifest_sample_indices: list[int] = []
    local_manifest_shard_ids: list[int] = []
    local_manifest_offsets: list[int] = []
    local_shard_paths: list[str] = []
    if overwrite:
        next_local_storage_id = 0
    else:
        next_local_storage_id = _next_local_storage_id_from_existing_paths(
            existing_paths=existing_shard_paths,
            storage_format=storage_format,
            rank=rank,
        )

    def flush_pending(force: bool = False):
        nonlocal next_local_storage_id, saved_count
        if storage_format != "sharded_v1":
            return
        while pending_sample_indices and (force or len(pending_sample_indices) >= shard_size):
            cur_size = min(len(pending_sample_indices), shard_size)
            cur_sample_indices = pending_sample_indices[:cur_size]
            cur_latents = pending_latents[:cur_size]
            del pending_sample_indices[:cur_size]
            del pending_latents[:cur_size]

            shard_relpath = f"shards/rank{rank:03d}_shard{next_local_storage_id:06d}.pt"
            shard_payload = {
                "sample_indices": torch.tensor(cur_sample_indices, dtype=torch.int64),
                "video_latents": torch.stack(cur_latents, dim=0).contiguous(),
            }
            _atomic_torch_save(shard_payload, latent_cache_dir / shard_relpath)

            local_shard_id = len(local_shard_paths)
            local_shard_paths.append(shard_relpath)
            local_manifest_sample_indices.extend(cur_sample_indices)
            local_manifest_shard_ids.extend([local_shard_id] * cur_size)
            local_manifest_offsets.extend(list(range(cur_size)))
            saved_count += cur_size
            next_local_storage_id += 1

    if storage_format == "episode_packed_v1":
        if (
            not hasattr(dataset, "load_episode_raw_images")
            or not hasattr(dataset, "build_processed_episode_pixel_values")
            or not hasattr(dataset, "build_video_only_batch_from_processed_episode_pixel_values")
        ):
            raise TypeError(
                "Episode-packed latent precompute requires a dataset that implements "
                "`load_episode_raw_images()`, `build_processed_episode_pixel_values()`, and "
                "`build_video_only_batch_from_processed_episode_pixel_values()`."
            )

        local_episode_indices = list(range(rank, total_episodes, world_size)) if is_distributed else list(range(total_episodes))
        progress = _make_rank_progress_bar(
            total=len(local_episode_indices),
            rank=rank,
            desc="precompute_video_latents",
        )
        timing_window = {
            "episodes": 0,
            "samples": 0,
            "encoded_samples": 0,
            "episode_load_ms": 0.0,
            "process_episode_ms": 0.0,
            "build_video_ms": 0.0,
            "encode_ms": 0.0,
            "save_ms": 0.0,
            "step_total_ms": 0.0,
        }
        mem_logger = _EpisodeMemoryLogger(rank=rank)
        max_episodes = _env_int("PRECOMPUTE_MAX_EPISODES", 0)
        gc_every = _env_int("PRECOMPUTE_GC_EVERY", 0)
        malloc_trim_every = _env_int("PRECOMPUTE_MALLOC_TRIM_EVERY", 0)
        malloc_trim = None
        if malloc_trim_every > 0:
            try:
                import ctypes

                malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
            except Exception as exc:  # pragma: no cover - platform specific
                malloc_trim = None
                logger.warning("malloc_trim unavailable (%s); PRECOMPUTE_MALLOC_TRIM_EVERY ignored.", exc)
        if rank == 0:
            logger.info(
                "Episode-packed memory knobs: mem_log_every=%d mem_log_dir=%s max_episodes=%d gc_every=%d malloc_trim_every=%d",
                mem_logger.every,
                str(mem_logger.jsonl_path.parent) if mem_logger.jsonl_path is not None else "<disabled>",
                max_episodes,
                gc_every,
                malloc_trim_every,
            )
        mem_logger.log(tag="before_loop", episode_idx=-1, step_idx=0, total_steps=len(local_episode_indices))
        with torch.no_grad():
            for step_idx, episode_idx in enumerate(local_episode_indices, start=1):
                sample_start, sample_end = dataset.get_episode_sample_range(episode_idx)
                episode_sample_indices = list(range(sample_start, sample_end))
                needs_encode = []
                for sample_idx in episode_sample_indices:
                    if overwrite or existing_sample_to_shard is None:
                        needs_encode.append(True)
                    else:
                        needs_encode.append(int(existing_sample_to_shard[sample_idx].item()) < 0)

                episode_load_ms = 0.0
                process_episode_ms = 0.0
                build_video_ms = 0.0
                encode_ms = 0.0
                save_ms = 0.0
                step_start = _timing_start(
                    enabled=precompute_timing_enabled,
                    sync_cuda=precompute_timing_sync_cuda,
                    device=model.device,
                )

                encoded_sample_indices: list[int] = []
                encoded_latents: list[torch.Tensor] = []
                if any(needs_encode):
                    load_start = _timing_start(
                        enabled=precompute_timing_enabled,
                        sync_cuda=precompute_timing_sync_cuda,
                        device=model.device,
                    )
                    episode_payload = dataset.load_episode_raw_images(episode_idx)
                    episode_load_ms = _timing_end_ms(
                        load_start,
                        enabled=precompute_timing_enabled,
                        sync_cuda=precompute_timing_sync_cuda,
                        device=model.device,
                    )
                    if (
                        int(episode_payload["sample_start"]) != sample_start
                        or int(episode_payload["sample_end"]) != sample_end
                    ):
                        raise ValueError(
                            f"Episode payload sample range mismatch for episode {episode_idx}: "
                            f"expected [{sample_start}, {sample_end}), got "
                            f"[{episode_payload['sample_start']}, {episode_payload['sample_end']})"
                        )
                    episode_images = episode_payload["images"]
                    process_start = _timing_start(
                        enabled=precompute_timing_enabled,
                        sync_cuda=precompute_timing_sync_cuda,
                        device=model.device,
                    )
                    processed_pixel_values = dataset.build_processed_episode_pixel_values(
                        episode_images=episode_images
                    )
                    process_episode_ms = _timing_end_ms(
                        process_start,
                        enabled=precompute_timing_enabled,
                        sync_cuda=precompute_timing_sync_cuda,
                        device=model.device,
                    )

                    for batch_start in range(0, len(episode_sample_indices), precompute_batch_size):
                        batch_indices = episode_sample_indices[batch_start: batch_start + precompute_batch_size]
                        batch_encode_indices = [
                            sample_idx
                            for sample_idx, need in zip(batch_indices, needs_encode[batch_start: batch_start + precompute_batch_size], strict=True)
                            if need
                        ]
                        if not batch_encode_indices:
                            continue

                        build_start = _timing_start(
                            enabled=precompute_timing_enabled,
                            sync_cuda=precompute_timing_sync_cuda,
                            device=model.device,
                        )
                        video_batch = dataset.build_video_only_batch_from_processed_episode_pixel_values(
                            processed_pixel_values=processed_pixel_values,
                            sample_indices=batch_encode_indices,
                            episode_sample_start=sample_start,
                            episode_sample_end=sample_end,
                        )
                        build_video_ms += _timing_end_ms(
                            build_start,
                            enabled=precompute_timing_enabled,
                            sync_cuda=precompute_timing_sync_cuda,
                            device=model.device,
                        )

                        encode_start = _timing_start(
                            enabled=precompute_timing_enabled,
                            sync_cuda=precompute_timing_sync_cuda,
                            device=model.device,
                        )
                        latents = model._encode_video_latents(
                            video_batch.to(
                                device=model.device,
                                dtype=model.torch_dtype,
                                non_blocking=True,
                            ),
                            tiled=tiled,
                        )
                        latents = _downsample_latents_for_cache(
                            latents, latent_spatial_downsample_factor
                        )
                        latents = latents.detach().to(device="cpu", dtype=cache_dtype)
                        encode_ms += _timing_end_ms(
                            encode_start,
                            enabled=precompute_timing_enabled,
                            sync_cuda=precompute_timing_sync_cuda,
                            device=model.device,
                        )

                        encoded_sample_indices.extend(batch_encode_indices)
                        encoded_latents.extend([latents[i].clone() for i in range(latents.shape[0])])

                    save_start = _timing_start(
                        enabled=precompute_timing_enabled,
                        sync_cuda=precompute_timing_sync_cuda,
                        device=model.device,
                    )
                    if encoded_sample_indices:
                        shard_relpath = (
                            f"episodes/rank{rank:03d}_episode{int(episode_idx):06d}_"
                            f"{next_local_storage_id:06d}.pt"
                        )
                        shard_payload = {
                            "episode_index": int(episode_idx),
                            "sample_indices": torch.tensor(encoded_sample_indices, dtype=torch.int64),
                            "video_latents": torch.stack(encoded_latents, dim=0).contiguous(),
                        }
                        _atomic_torch_save(shard_payload, latent_cache_dir / shard_relpath)

                        local_shard_id = len(local_shard_paths)
                        local_shard_paths.append(shard_relpath)
                        local_manifest_sample_indices.extend(encoded_sample_indices)
                        local_manifest_shard_ids.extend([local_shard_id] * len(encoded_sample_indices))
                        local_manifest_offsets.extend(list(range(len(encoded_sample_indices))))
                        saved_count += len(encoded_sample_indices)
                        next_local_storage_id += 1
                    save_ms = _timing_end_ms(
                        save_start,
                        enabled=precompute_timing_enabled,
                        sync_cuda=precompute_timing_sync_cuda,
                        device=model.device,
                    )
                    skipped_count += len(episode_sample_indices) - len(encoded_sample_indices)
                    timing_window["encoded_samples"] += len(encoded_sample_indices)
                else:
                    skipped_count += len(episode_sample_indices)

                step_total_ms = _timing_end_ms(
                    step_start,
                    enabled=precompute_timing_enabled,
                    sync_cuda=precompute_timing_sync_cuda,
                    device=model.device,
                )
                timing_window["episodes"] += 1
                timing_window["samples"] += len(episode_sample_indices)
                timing_window["episode_load_ms"] += episode_load_ms
                timing_window["process_episode_ms"] += process_episode_ms
                timing_window["build_video_ms"] += build_video_ms
                timing_window["encode_ms"] += encode_ms
                timing_window["save_ms"] += save_ms
                timing_window["step_total_ms"] += step_total_ms

                postfix = {"saved": saved_count, "skipped": skipped_count}
                if precompute_timing_enabled:
                    postfix["step_ms"] = f"{step_total_ms:.1f}"
                    postfix["load_ms"] = f"{episode_load_ms:.1f}"
                    postfix["proc_ms"] = f"{process_episode_ms:.1f}"
                    postfix["build_ms"] = f"{build_video_ms:.1f}"
                    postfix["encode_ms"] = f"{encode_ms:.1f}"
                    postfix["save_ms"] = f"{save_ms:.1f}"
                progress.set_postfix(postfix)
                progress.update(1)
                if (
                    precompute_timing_enabled
                    and precompute_timing_log_every > 0
                    and (step_idx == 1 or step_idx % precompute_timing_log_every == 0)
                    and timing_window["episodes"] > 0
                ):
                    window_steps = int(timing_window["episodes"])
                    avg_episode_load_ms = timing_window["episode_load_ms"] / window_steps
                    avg_process_episode_ms = timing_window["process_episode_ms"] / window_steps
                    avg_build_video_ms = timing_window["build_video_ms"] / window_steps
                    avg_encode_ms = timing_window["encode_ms"] / window_steps
                    avg_save_ms = timing_window["save_ms"] / window_steps
                    avg_step_total_ms = timing_window["step_total_ms"] / window_steps
                    samples_per_sec = timing_window["samples"] / max(
                        timing_window["step_total_ms"] / 1000.0,
                        1.0e-9,
                    )
                    encoded_samples_per_sec = timing_window["encoded_samples"] / max(
                        timing_window["step_total_ms"] / 1000.0,
                        1.0e-9,
                    )
                    logger.info(
                        "[precompute_timing][rank=%d] step=%d/%d episode_load_ms=%.1f process_episode_ms=%.1f build_video_ms=%.1f "
                        "encode_ms=%.1f save_ms=%.1f step_total_ms=%.1f samples_per_sec=%.2f "
                        "encoded_samples_per_sec=%.2f",
                        rank,
                        step_idx,
                        len(local_episode_indices),
                        avg_episode_load_ms,
                        avg_process_episode_ms,
                        avg_build_video_ms,
                        avg_encode_ms,
                        avg_save_ms,
                        avg_step_total_ms,
                        samples_per_sec,
                        encoded_samples_per_sec,
                    )
                    if step_idx != 1:
                        for key in timing_window:
                            timing_window[key] = 0 if key in {"episodes", "samples", "encoded_samples"} else 0.0
                if mem_logger.enabled and (step_idx == 1 or step_idx % mem_logger.every == 0):
                    mem_logger.log(
                        tag="episode",
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        total_steps=len(local_episode_indices),
                    )
                if gc_every > 0 and step_idx % gc_every == 0:
                    gc.collect()
                if malloc_trim is not None and malloc_trim_every > 0 and step_idx % malloc_trim_every == 0:
                    malloc_trim(0)
                if max_episodes > 0 and step_idx >= max_episodes:
                    mem_logger.log(
                        tag="max_episodes_reached",
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        total_steps=len(local_episode_indices),
                    )
                    logger.info(
                        "[precompute] PRECOMPUTE_MAX_EPISODES=%d reached at step=%d (rank=%d); "
                        "finishing early with manifest/index flush.",
                        max_episodes,
                        step_idx,
                        rank,
                    )
                    break
    else:
        progress = _make_rank_progress_bar(
            total=len(loader),
            rank=rank,
            desc="precompute_video_latents",
        )
        data_iter = iter(loader)
        timing_window = {
            "batches": 0,
            "samples": 0,
            "encoded_samples": 0,
            "data_wait_ms": 0.0,
            "encode_ms": 0.0,
            "save_ms": 0.0,
            "step_total_ms": 0.0,
        }
        with torch.no_grad():
            step_idx = 0
            while True:
                data_wait_start = _timing_start(
                    enabled=precompute_timing_enabled,
                    sync_cuda=precompute_timing_sync_cuda,
                    device=model.device,
                )
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                data_wait_ms = _timing_end_ms(
                    data_wait_start,
                    enabled=precompute_timing_enabled,
                    sync_cuda=precompute_timing_sync_cuda,
                    device=model.device,
                )
                step_start = _timing_start(
                    enabled=precompute_timing_enabled,
                    sync_cuda=precompute_timing_sync_cuda,
                    device=model.device,
                )
                batch_indices = torch.as_tensor(batch["idx"], dtype=torch.long).view(-1).tolist()
                batch_sample_count = len(batch_indices)
                if storage_format == "sharded_v1":
                    needs_encode = []
                    for sample_idx in batch_indices:
                        if overwrite or existing_sample_to_shard is None:
                            needs_encode.append(True)
                        else:
                            needs_encode.append(int(existing_sample_to_shard[sample_idx].item()) < 0)
                else:
                    cache_paths = [latent_cache_dir / f"{sample_idx:08d}.pt" for sample_idx in batch_indices]
                    needs_encode = [overwrite or (not cache_path.exists()) for cache_path in cache_paths]
                encode_ms = 0.0
                save_ms = 0.0
                if not any(needs_encode):
                    skipped_count += len(batch_indices)
                else:
                    video = batch["video"]
                    if not isinstance(video, torch.Tensor) or video.ndim != 5:
                        raise ValueError(
                            f"Expected batched `video` tensor with shape [B, C, T, H, W], got {type(video)} / "
                            f"{None if not isinstance(video, torch.Tensor) else tuple(video.shape)}"
                        )

                    encode_indices = [i for i, need in enumerate(needs_encode) if need]
                    encode_start = _timing_start(
                        enabled=precompute_timing_enabled,
                        sync_cuda=precompute_timing_sync_cuda,
                        device=model.device,
                    )
                    video_to_encode = video[encode_indices].to(
                        device=model.device,
                        dtype=model.torch_dtype,
                        non_blocking=True,
                    )
                    latents = model._encode_video_latents(video_to_encode, tiled=tiled)
                    latents = _downsample_latents_for_cache(
                        latents, latent_spatial_downsample_factor
                    )
                    latents = latents.detach().to(device="cpu", dtype=cache_dtype)
                    encode_ms = _timing_end_ms(
                        encode_start,
                        enabled=precompute_timing_enabled,
                        sync_cuda=precompute_timing_sync_cuda,
                        device=model.device,
                    )

                    save_start = _timing_start(
                        enabled=precompute_timing_enabled,
                        sync_cuda=precompute_timing_sync_cuda,
                        device=model.device,
                    )
                    for local_pos, batch_pos in enumerate(encode_indices):
                        sample_idx = batch_indices[batch_pos]
                        video_latents = latents[local_pos].clone()
                        if storage_format == "sharded_v1":
                            pending_sample_indices.append(int(sample_idx))
                            pending_latents.append(video_latents)
                        else:
                            cache_path = cache_paths[batch_pos]
                            payload = {
                                "sample_idx": int(sample_idx),
                                "video_latents": video_latents,
                            }
                            _atomic_torch_save(payload, cache_path)
                            saved_count += 1

                    flush_pending(force=False)
                    save_ms = _timing_end_ms(
                        save_start,
                        enabled=precompute_timing_enabled,
                        sync_cuda=precompute_timing_sync_cuda,
                        device=model.device,
                    )
                    skipped_count += len(batch_indices) - len(encode_indices)
                    timing_window["encoded_samples"] += len(encode_indices)

                step_total_ms = _timing_end_ms(
                    step_start,
                    enabled=precompute_timing_enabled,
                    sync_cuda=precompute_timing_sync_cuda,
                    device=model.device,
                )
                timing_window["batches"] += 1
                timing_window["samples"] += batch_sample_count
                timing_window["data_wait_ms"] += data_wait_ms
                timing_window["encode_ms"] += encode_ms
                timing_window["save_ms"] += save_ms
                timing_window["step_total_ms"] += step_total_ms
                step_idx += 1
                postfix = {"saved": saved_count, "skipped": skipped_count}
                if precompute_timing_enabled:
                    postfix["wait_ms"] = f"{data_wait_ms:.1f}"
                    postfix["step_ms"] = f"{step_total_ms:.1f}"
                    postfix["encode_ms"] = f"{encode_ms:.1f}"
                    postfix["save_ms"] = f"{save_ms:.1f}"
                progress.set_postfix(postfix)
                progress.update(1)
                if (
                    precompute_timing_enabled
                    and precompute_timing_log_every > 0
                    and (step_idx == 1 or step_idx % precompute_timing_log_every == 0)
                    and timing_window["batches"] > 0
                ):
                    window_batches = int(timing_window["batches"])
                    avg_data_wait_ms = timing_window["data_wait_ms"] / window_batches
                    avg_encode_ms = timing_window["encode_ms"] / window_batches
                    avg_save_ms = timing_window["save_ms"] / window_batches
                    avg_step_total_ms = timing_window["step_total_ms"] / window_batches
                    samples_per_sec = timing_window["samples"] / max(
                        timing_window["step_total_ms"] / 1000.0,
                        1.0e-9,
                    )
                    encoded_samples_per_sec = timing_window["encoded_samples"] / max(
                        timing_window["step_total_ms"] / 1000.0,
                        1.0e-9,
                    )
                    logger.info(
                        "[precompute_timing][rank=%d] step=%d/%d data_wait_ms=%.1f encode_ms=%.1f "
                        "save_ms=%.1f step_total_ms=%.1f samples_per_sec=%.2f encoded_samples_per_sec=%.2f",
                        rank,
                        step_idx,
                        len(loader),
                        avg_data_wait_ms,
                        avg_encode_ms,
                        avg_save_ms,
                        avg_step_total_ms,
                        samples_per_sec,
                        encoded_samples_per_sec,
                    )
                    if step_idx != 1:
                        for key in timing_window:
                            timing_window[key] = 0 if key in {"batches", "samples", "encoded_samples"} else 0.0

        flush_pending(force=True)

    if indexed_storage:
        local_manifest_path = manifests_dir / f"rank{rank:03d}.pt"
        local_manifest_payload = {
            "sample_indices": torch.tensor(local_manifest_sample_indices, dtype=torch.int64),
            "local_shard_ids": torch.tensor(local_manifest_shard_ids, dtype=torch.int64),
            "offsets": torch.tensor(local_manifest_offsets, dtype=torch.int64),
            "shard_paths": local_shard_paths,
        }
        _atomic_torch_save(local_manifest_payload, local_manifest_path)

    if is_distributed:
        counts_device = model.device if dist.get_backend() == "nccl" else torch.device("cpu")
        counts = torch.tensor([saved_count, skipped_count], device=counts_device, dtype=torch.long)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        saved_count = int(counts[0].item())
        skipped_count = int(counts[1].item())
        dist.barrier()

    if indexed_storage and precompute_resume and is_distributed:
        if rank == 0:
            _atomic_json_dump(
                {
                    "status": "rebuilding",
                    "pid": int(os.getpid()),
                    "started_at": time.time(),
                },
                rebuild_marker_path,
            )
        dist.barrier()

    if indexed_storage and rank == 0:
        if precompute_resume:
            try:
                _rebuild_index_from_cache_files(
                    latent_cache_dir=latent_cache_dir,
                    index_path=index_path,
                    storage_dir=storage_dir,
                    total_samples=total_samples,
                    storage_format=storage_format,
                )
            finally:
                if rebuild_marker_path.exists():
                    rebuild_marker_path.unlink()
        else:
            _merge_indexed_cache_manifests(
                latent_cache_dir=latent_cache_dir,
                index_path=index_path,
                manifests_dir=manifests_dir,
                world_size=world_size,
                total_samples=total_samples,
                storage_format=storage_format,
                overwrite=overwrite,
                existing_sample_to_shard=existing_sample_to_shard,
                existing_sample_to_offset=existing_sample_to_offset,
                existing_shard_paths=existing_shard_paths,
            )

    if indexed_storage and is_distributed:
        if rank != 0:
            _wait_for_index_ready(
                index_path=index_path,
                total_samples=total_samples,
                storage_format=storage_format,
                rebuild_marker_path=(rebuild_marker_path if precompute_resume else None),
            )

    if storage_format == "episode_packed_v1":
        mem_logger.log(
            tag="finished",
            episode_idx=-1,
            step_idx=len(local_episode_indices),
            total_steps=len(local_episode_indices),
        )

    if rank == 0:
        logger.info(
            "Finished precomputing video latents into %s. saved=%d skipped=%d storage_format=%s",
            latent_cache_dir,
            saved_count,
            skipped_count,
            storage_format,
        )


if __name__ == "__main__":
    main()
