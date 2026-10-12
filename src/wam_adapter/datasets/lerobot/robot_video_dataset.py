import hashlib
import inspect
import os
import re
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
import pyarrow.parquet as pq
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from wam_adapter.utils.logging_config import get_logger
from wam_adapter.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _model_id_to_cache_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        text_embedding_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        episode_indices_file: Optional[str] = None,
        global_sample_stride=1,
        sample_start_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        use_latent_cache: bool = False,
        latent_cache_dir: Optional[str] = None,
        video_only: bool = False,
        video_backend: Optional[str] = None,
    ):
        self.video_only = bool(video_only)
        self.use_latent_cache = bool(use_latent_cache)
        if latent_cache_dir is None or str(latent_cache_dir).strip() == "":
            self.latent_cache_dir = None
        else:
            self.latent_cache_dir = os.path.abspath(os.path.expanduser(str(latent_cache_dir)))
        if self.use_latent_cache and self.latent_cache_dir is None:
            raise ValueError("`latent_cache_dir` must be set when `use_latent_cache=true`.")
        if self.video_only and self.use_latent_cache:
            raise ValueError("`video_only=true` is incompatible with `use_latent_cache=true`.")

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            episode_indices_file=episode_indices_file,
            global_sample_stride=global_sample_stride,
            sample_start_stride=sample_start_stride,
            return_action_state=not self.video_only,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.processor = None
        self.lerobot_dataset._set_return_images(not self.use_latent_cache)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.text_embedding_cache_id = _model_id_to_cache_id(text_embedding_model_id)
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self._latent_cache_format = "single_file_v1"
        self._latent_cache_shard_paths = None
        self._latent_cache_sample_to_shard = None
        self._latent_cache_sample_to_offset = None
        self._latent_cache_last_shard_relpath = None
        self._latent_cache_last_shard_payload = None

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            self.processor = processor
            if is_training_set:
                self.processor.train()
            else:
                self.processor.eval()

            if not self.video_only:
                if not pretrained_norm_stats:
                    if not is_training_set:
                        raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                    if PartialState().is_main_process:
                        logger.info("Calculating dataset stats for normalization...")
                        dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                        work_dir = misc.get_work_dir()
                        save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                    else:
                        dataset_stats = None
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        obj_list = [dataset_stats]
                        torch.distributed.broadcast_object_list(obj_list, src=0)
                        dataset_stats = obj_list[0]
                else:
                    dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                    logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                    if PartialState().is_main_process:
                        work_dir = misc.get_work_dir()
                        save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

                processor.set_normalizer_from_stats(dataset_stats)
                if not self.use_latent_cache:
                    self.lerobot_dataset.set_processor(processor)

        if self.use_latent_cache:
            if self.processor is None:
                raise ValueError("`processor` is required when `use_latent_cache=true`.")
            if not hasattr(self.processor, "preprocess_without_images"):
                raise ValueError(
                    "`processor` must implement `preprocess_without_images()` when "
                    "`use_latent_cache=true`."
                )
            self._init_latent_cache_reader()
            logger.info("Using latent cache for RobotVideoDataset: %s", self.latent_cache_dir)

    def __len__(self):
        return len(self.lerobot_dataset)

    def _init_latent_cache_reader(self):
        if self.latent_cache_dir is None:
            raise ValueError("`latent_cache_dir` is not set.")

        index_path = os.path.join(self.latent_cache_dir, "index.pt")
        if not os.path.exists(index_path):
            self._latent_cache_format = "single_file_v1"
            return

        index_payload = torch.load(index_path, map_location="cpu")
        if not isinstance(index_payload, dict):
            raise TypeError(
                f"Latent cache index must be a dict, got {type(index_payload)} in {index_path}."
            )

        storage_format = str(index_payload.get("storage_format", "sharded_v1"))
        shard_paths = index_payload.get("shard_paths")
        sample_to_shard = index_payload.get("sample_to_shard")
        sample_to_offset = index_payload.get("sample_to_offset")
        if not isinstance(shard_paths, list):
            raise TypeError(f"`shard_paths` must be a list in {index_path}.")
        if not isinstance(sample_to_shard, torch.Tensor) or not isinstance(sample_to_offset, torch.Tensor):
            raise TypeError(
                f"`sample_to_shard` and `sample_to_offset` must be tensors in {index_path}."
            )

        if storage_format not in {"sharded_v1", "episode_packed_v1"}:
            raise ValueError(
                f"Unsupported indexed latent cache format `{storage_format}` in {index_path}."
            )
        self._latent_cache_format = storage_format
        self._latent_cache_shard_paths = [str(path) for path in shard_paths]
        self._latent_cache_sample_to_shard = sample_to_shard.to(dtype=torch.int64, device="cpu").contiguous()
        self._latent_cache_sample_to_offset = sample_to_offset.to(dtype=torch.int64, device="cpu").contiguous()
        logger.info(
            "Loaded indexed latent cache index: format=%s shards=%d samples=%d",
            storage_format,
            len(self._latent_cache_shard_paths),
            int(self._latent_cache_sample_to_shard.shape[0]),
        )

    def _get_latent_cache_path(self, sample_idx: int) -> str:
        if self.latent_cache_dir is None:
            raise ValueError("`latent_cache_dir` is not set.")
        return os.path.join(self.latent_cache_dir, f"{int(sample_idx):08d}.pt")

    @staticmethod
    def _validate_video_latents(video_latents: torch.Tensor, source_path: str) -> torch.Tensor:
        if not isinstance(video_latents, torch.Tensor):
            raise TypeError(
                f"`video_latents` must be a torch.Tensor, got {type(video_latents)} in {source_path}."
            )
        if video_latents.ndim == 5 and video_latents.shape[0] == 1:
            video_latents = video_latents.squeeze(0)
        if video_latents.ndim != 4:
            raise ValueError(
                f"Cached `video_latents` must be 4D [C,T,H,W], got shape {tuple(video_latents.shape)} "
                f"in {source_path}"
            )
        return video_latents.contiguous()

    def _load_cached_video_latents_from_single_file(self, sample_idx: int) -> torch.Tensor:
        cache_path = self._get_latent_cache_path(sample_idx)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing latent cache for sample idx={sample_idx}: {cache_path}. "
                "Run scripts/precompute_video_latents.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict):
            if "video_latents" not in payload:
                raise KeyError(f"Latent cache payload missing `video_latents`: {cache_path}")
            video_latents = payload["video_latents"]
        elif isinstance(payload, torch.Tensor):
            video_latents = payload
        else:
            raise TypeError(
                f"Unsupported latent cache payload type {type(payload)} in {cache_path}."
            )
        return self._validate_video_latents(video_latents, cache_path)

    def _load_cached_video_latents_from_shard(self, sample_idx: int) -> torch.Tensor:
        if self.latent_cache_dir is None:
            raise ValueError("`latent_cache_dir` is not set.")
        if self._latent_cache_sample_to_shard is None or self._latent_cache_sample_to_offset is None:
            raise RuntimeError("Sharded latent cache index was not initialized.")

        if sample_idx < 0 or sample_idx >= int(self._latent_cache_sample_to_shard.shape[0]):
            raise IndexError(
                f"Latent cache sample idx {sample_idx} out of range [0, {int(self._latent_cache_sample_to_shard.shape[0])})."
            )
        shard_id = int(self._latent_cache_sample_to_shard[sample_idx].item())
        offset = int(self._latent_cache_sample_to_offset[sample_idx].item())
        if shard_id < 0 or offset < 0:
            raise FileNotFoundError(
                f"Missing sharded latent cache entry for sample idx={sample_idx}: {self.latent_cache_dir}"
            )
        if self._latent_cache_shard_paths is None or shard_id >= len(self._latent_cache_shard_paths):
            raise IndexError(
                f"Invalid shard id {shard_id} for sample idx={sample_idx} in {self.latent_cache_dir}."
            )

        shard_relpath = self._latent_cache_shard_paths[shard_id]
        if self._latent_cache_last_shard_relpath != shard_relpath:
            shard_path = os.path.join(self.latent_cache_dir, shard_relpath)
            load_kwargs = {"map_location": "cpu"}
            if "mmap" in inspect.signature(torch.load).parameters:
                load_kwargs["mmap"] = True
            self._latent_cache_last_shard_payload = torch.load(shard_path, **load_kwargs)
            self._latent_cache_last_shard_relpath = shard_relpath

        shard_payload = self._latent_cache_last_shard_payload
        if not isinstance(shard_payload, dict):
            raise TypeError(
                f"Shard payload must be a dict, got {type(shard_payload)} in {self._latent_cache_last_shard_relpath}."
            )
        if "video_latents" not in shard_payload:
            raise KeyError(f"Shard payload missing `video_latents`: {self._latent_cache_last_shard_relpath}")
        video_latents = shard_payload["video_latents"]
        if not isinstance(video_latents, torch.Tensor) or video_latents.ndim != 5:
            raise ValueError(
                f"Sharded `video_latents` must be 5D [N,C,T,H,W], got "
                f"{None if not isinstance(video_latents, torch.Tensor) else tuple(video_latents.shape)} "
                f"in {self._latent_cache_last_shard_relpath}"
            )
        if offset >= int(video_latents.shape[0]):
            raise IndexError(
                f"Offset {offset} out of range for shard {self._latent_cache_last_shard_relpath}."
            )
        sample_indices = shard_payload.get("sample_indices")
        if isinstance(sample_indices, torch.Tensor):
            if int(sample_indices[offset].item()) != int(sample_idx):
                raise ValueError(
                    f"Shard index mismatch for sample idx={sample_idx}: "
                    f"found {int(sample_indices[offset].item())} in {self._latent_cache_last_shard_relpath}"
                )
        return self._validate_video_latents(video_latents[offset], self._latent_cache_last_shard_relpath)

    def _load_cached_video_latents(self, sample_idx: int) -> torch.Tensor:
        if self._latent_cache_format in {"sharded_v1", "episode_packed_v1"}:
            return self._load_cached_video_latents_from_shard(sample_idx)
        return self._load_cached_video_latents_from_single_file(sample_idx)

    @staticmethod
    def _resolve_sample_idx(sample: dict) -> int:
        if "idx" not in sample:
            raise KeyError("Sample is missing `idx`.")
        sample_idx = sample["idx"]
        if isinstance(sample_idx, torch.Tensor):
            if sample_idx.numel() != 1:
                raise ValueError(f"`sample['idx']` tensor must contain a single value, got shape {tuple(sample_idx.shape)}")
            sample_idx = int(sample_idx.item())
        else:
            sample_idx = int(sample_idx)
        return sample_idx

    def _build_video_from_raw_images(self, sample: dict) -> torch.Tensor:
        if self.processor is None:
            raise ValueError("`processor` must be initialized for `video_only=true`.")
        if not hasattr(self.processor, "build_pixel_values_from_images"):
            raise ValueError(
                "`processor` must implement `build_pixel_values_from_images()` "
                "for `video_only=true` latent precompute."
            )
        return self.processor.build_pixel_values_from_images(sample)

    def _build_episode_pixel_values_from_raw_images(self, episode_images: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.processor is None:
            raise ValueError("`processor` must be initialized for `video_only=true`.")
        if not hasattr(self.processor, "build_pixel_values_from_episode_images"):
            raise ValueError(
                "`processor` must implement `build_pixel_values_from_episode_images()` "
                "for episode-level latent precompute."
            )
        return self.processor.build_pixel_values_from_episode_images({"images": episode_images})

    def _finalize_video_tensor(
        self,
        video: torch.Tensor,
        image_is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"`video` must be a torch.Tensor, got {type(video)}")
        if not isinstance(image_is_pad, torch.Tensor):
            image_is_pad = torch.as_tensor(image_is_pad, dtype=torch.bool)
        image_is_pad = image_is_pad.to(dtype=torch.bool)

        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, t_video, c_dim, h_dim, w_dim = video.shape
        else:
            if video.ndim != 4:
                raise ValueError(f"Expected video to have shape [T,C,H,W] or [N,T,C,H,W], got {tuple(video.shape)}")
            video = video[self.video_sample_indices, :, :, :]
            t_video, c_dim, h_dim, w_dim = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.reshape(num_cameras, t_video, c_dim, h_dim, w_dim)
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3).contiguous()
        return video, image_is_pad.contiguous()

    def _finalize_batched_video_tensor(
        self,
        video: torch.Tensor,
        image_is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"`video` must be a torch.Tensor, got {type(video)}")
        if video.ndim not in {5, 6}:
            raise ValueError(
                f"Expected batched video to have shape [B,T,C,H,W] or [B,N,T,C,H,W], got {tuple(video.shape)}"
            )
        if not isinstance(image_is_pad, torch.Tensor):
            image_is_pad = torch.as_tensor(image_is_pad, dtype=torch.bool)
        image_is_pad = image_is_pad.to(dtype=torch.bool)
        if image_is_pad.ndim != 2:
            raise ValueError(f"`image_is_pad` must be 2D [B,T], got {tuple(image_is_pad.shape)}")
        if image_is_pad.shape[0] != video.shape[0]:
            raise ValueError(
                f"`image_is_pad` batch mismatch: video batch={video.shape[0]} vs {tuple(image_is_pad.shape)}"
            )

        def _apply_framewise_transform(video_btc_hw: torch.Tensor, transform_fn) -> torch.Tensor:
            if video_btc_hw.ndim != 5:
                raise ValueError(
                    f"Expected framewise tensor with shape [B,T,C,H,W], got {tuple(video_btc_hw.shape)}"
                )
            batch_size_local, time_local, channels_local, height_local, width_local = video_btc_hw.shape
            flat_video = video_btc_hw.reshape(
                batch_size_local * time_local,
                channels_local,
                height_local,
                width_local,
            )
            flat_video = transform_fn(flat_video)
            if flat_video.ndim != 4:
                raise ValueError(
                    f"Framewise transform must return 4D [B*T,C,H,W], got {tuple(flat_video.shape)}"
                )
            out_channels, out_height, out_width = flat_video.shape[1:]
            return flat_video.reshape(
                batch_size_local,
                time_local,
                out_channels,
                out_height,
                out_width,
            )

        if video.ndim == 6:
            video = video[:, :, self.video_sample_indices, :, :, :]
            batch_size, num_cameras, t_video, c_dim, h_dim, w_dim = video.shape
        else:
            video = video[:, self.video_sample_indices, :, :, :]
            batch_size, t_video, c_dim, h_dim, w_dim = video.shape
            num_cameras = 1
        image_is_pad = image_is_pad[:, self.video_sample_indices]

        if num_cameras == 1 and video.ndim == 6:
            video = video[:, 0]
        elif video.ndim == 5:
            pass
        else:
            video = video.reshape(batch_size, num_cameras, t_video, c_dim, h_dim, w_dim)

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = _apply_framewise_transform(
                video[:, 0],
                lambda x: transforms_F.resize(
                    x,
                    size=[256, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            )
            cam_left = _apply_framewise_transform(
                video[:, 1],
                lambda x: transforms_F.resize(
                    x,
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            )
            cam_right = _apply_framewise_transform(
                video[:, 2],
                lambda x: transforms_F.resize(
                    x,
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[:, i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[:, i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        elif num_cameras == 1 and video.ndim == 6:
            video = video[:, 0]

        video = _apply_framewise_transform(video, self.resize_transform)
        video = _apply_framewise_transform(video, self.crop_transform)
        video = _apply_framewise_transform(video, self.normalize_transform)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        return video, image_is_pad.contiguous()

    def get_num_episodes(self) -> int:
        return int(self.lerobot_dataset.multi_dataset.num_episodes)

    def get_episode_sample_range(self, episode_idx: int) -> tuple[int, int]:
        num_episodes = self.get_num_episodes()
        if episode_idx < 0 or episode_idx >= num_episodes:
            raise IndexError(f"Episode index {episode_idx} out of bounds [0, {num_episodes}).")
        sample_start = int(self.lerobot_dataset.episode_data_index["from"][episode_idx].item())
        sample_end = int(self.lerobot_dataset.episode_data_index["to"][episode_idx].item())
        if sample_end <= sample_start:
            raise ValueError(
                f"Invalid episode sample range for episode {episode_idx}: start={sample_start}, end={sample_end}"
            )
        return sample_start, sample_end

    def get_source_episode_sample_ranges(self) -> dict[int, tuple[int, int]]:
        """Return flat dataset ranges keyed by immutable source episode id."""
        ranges: dict[int, tuple[int, int]] = {}
        for local_episode_idx in range(self.get_num_episodes()):
            _, _, _, source_episode_idx = self._resolve_episode_dataset(local_episode_idx)
            if source_episode_idx in ranges:
                raise ValueError(f"Duplicate source episode index: {source_episode_idx}")
            ranges[source_episode_idx] = self.get_episode_sample_range(local_episode_idx)
        return ranges

    def _resolve_episode_dataset(self, episode_idx: int):
        remaining = int(episode_idx)
        for dataset_index, dataset in enumerate(self.lerobot_dataset.multi_dataset._datasets):
            if remaining < dataset.num_episodes:
                source_episode_idx = int(dataset.episodes[remaining]) if dataset.episodes is not None else int(remaining)
                return dataset_index, dataset, int(remaining), source_episode_idx
            remaining -= dataset.num_episodes
        raise IndexError(f"Episode index {episode_idx} out of bounds.")

    @staticmethod
    def _read_episode_timestamps(parquet_path: str) -> list[float]:
        table = pq.read_table(parquet_path, columns=["timestamp"])
        column = table["timestamp"]
        try:
            timestamps = column.to_numpy(zero_copy_only=True)
        except Exception:
            timestamps = column.to_numpy()
        return np.asarray(timestamps, dtype=np.float64).tolist()

    def load_episode_raw_images(self, episode_idx: int) -> dict:
        _, dataset, _, source_episode_idx = self._resolve_episode_dataset(episode_idx)
        sample_start, sample_end = self.get_episode_sample_range(episode_idx)
        parquet_path = dataset.root / dataset.meta.get_data_file_path(source_episode_idx)
        timestamps = self._read_episode_timestamps(str(parquet_path))
        episode_length = sample_end - sample_start
        if len(timestamps) != episode_length:
            raise ValueError(
                f"Episode {episode_idx} timestamp length mismatch: expected {episode_length}, got {len(timestamps)}"
            )

        query_timestamps = {vid_key: timestamps for vid_key in dataset.meta.video_keys}
        raw_videos = dataset._query_videos(query_timestamps=query_timestamps, ep_idx=source_episode_idx)
        episode_images = {}
        for meta in self.lerobot_dataset.image_meta:
            key = meta["key"]
            lerobot_key = meta["lerobot_key"]
            if lerobot_key not in raw_videos:
                raise KeyError(
                    f"Missing video key `{lerobot_key}` when loading episode {episode_idx} from {dataset.root}."
                )
            image = raw_videos[lerobot_key]
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4:
                raise ValueError(
                    f"Episode image `{lerobot_key}` must be [T,C,H,W], got {tuple(image.shape)}"
                )
            episode_images[key] = (image * 255).to(torch.uint8).contiguous()

        return {
            "episode_index": int(episode_idx),
            "sample_start": int(sample_start),
            "sample_end": int(sample_end),
            "timestamps": timestamps,
            "images": episode_images,
        }

    def _build_episode_observation_indices(
        self,
        sample_idx: int,
        episode_sample_start: int,
        episode_sample_end: int,
    ) -> tuple[list[int], torch.Tensor]:
        if sample_idx < episode_sample_start or sample_idx >= episode_sample_end:
            raise ValueError(
                f"Sample idx {sample_idx} is outside episode range [{episode_sample_start}, {episode_sample_end})."
            )
        episode_length = episode_sample_end - episode_sample_start
        local_idx = sample_idx - episode_sample_start
        frame_indices = []
        image_is_pad = []
        for step_idx in range(self.num_frames):
            raw_idx = local_idx + step_idx * int(self.lerobot_dataset.global_sample_stride)
            is_pad = raw_idx < 0 or raw_idx >= episode_length
            clamped_idx = min(max(raw_idx, 0), episode_length - 1)
            frame_indices.append(int(clamped_idx))
            image_is_pad.append(bool(is_pad))
        return frame_indices, torch.tensor(image_is_pad, dtype=torch.bool)

    def _build_episode_observation_indices_tensor(
        self,
        sample_indices: list[int] | torch.Tensor,
        episode_sample_start: int,
        episode_sample_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.to(dtype=torch.int64).view(-1)
        else:
            sample_indices = torch.tensor(sample_indices, dtype=torch.int64)
        if sample_indices.numel() == 0:
            return (
                torch.empty((0, self.num_frames), dtype=torch.int64),
                torch.empty((0, self.num_frames), dtype=torch.bool),
            )

        if bool((sample_indices < int(episode_sample_start)).any().item()) or bool(
            (sample_indices >= int(episode_sample_end)).any().item()
        ):
            raise ValueError(
                f"Sample indices must lie within [{episode_sample_start}, {episode_sample_end})."
            )

        episode_length = int(episode_sample_end - episode_sample_start)
        local_indices = sample_indices - int(episode_sample_start)
        step_offsets = torch.arange(self.num_frames, dtype=torch.int64) * int(
            self.lerobot_dataset.global_sample_stride
        )
        raw_indices = local_indices.unsqueeze(1) + step_offsets.unsqueeze(0)
        image_is_pad = (raw_indices < 0) | (raw_indices >= episode_length)
        frame_indices = raw_indices.clamp(min=0, max=episode_length - 1)
        return frame_indices.contiguous(), image_is_pad.contiguous()

    def build_processed_episode_pixel_values(self, episode_images: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._build_episode_pixel_values_from_raw_images(episode_images).contiguous()

    def build_video_only_clip_from_episode_images(
        self,
        episode_images: dict[str, torch.Tensor],
        sample_idx: int,
        episode_sample_start: int,
        episode_sample_end: int,
    ) -> torch.Tensor:
        frame_indices, image_is_pad = self._build_episode_observation_indices(
            sample_idx=sample_idx,
            episode_sample_start=episode_sample_start,
            episode_sample_end=episode_sample_end,
        )
        sample = {
            "images": {
                key: image[frame_indices].contiguous()
                for key, image in episode_images.items()
            }
        }
        video = self._build_video_from_raw_images(sample)
        video, _ = self._finalize_video_tensor(video=video, image_is_pad=image_is_pad)
        return video

    def build_video_only_batch_from_processed_episode_pixel_values(
        self,
        processed_pixel_values: torch.Tensor,
        sample_indices: list[int] | torch.Tensor,
        episode_sample_start: int,
        episode_sample_end: int,
    ) -> torch.Tensor:
        if not isinstance(processed_pixel_values, torch.Tensor):
            raise TypeError(
                f"`processed_pixel_values` must be a torch.Tensor, got {type(processed_pixel_values)}"
            )
        if processed_pixel_values.ndim != 5:
            raise ValueError(
                f"`processed_pixel_values` must have shape [N,T,C,H,W], got {tuple(processed_pixel_values.shape)}"
            )
        frame_indices, image_is_pad = self._build_episode_observation_indices_tensor(
            sample_indices=sample_indices,
            episode_sample_start=episode_sample_start,
            episode_sample_end=episode_sample_end,
        )
        if frame_indices.numel() == 0:
            return torch.empty(
                (
                    0,
                    int(processed_pixel_values.shape[2]),
                    len(self.video_sample_indices),
                    int(self.video_size[0]),
                    int(self.video_size[1]),
                ),
                dtype=processed_pixel_values.dtype,
            )
        if int(frame_indices.max().item()) >= int(processed_pixel_values.shape[1]):
            raise ValueError(
                f"Episode frame index out of bounds: max={int(frame_indices.max().item())} "
                f"vs processed length={processed_pixel_values.shape[1]}"
            )

        gathered_video = processed_pixel_values[:, frame_indices]
        gathered_video = gathered_video.permute(1, 0, 2, 3, 4, 5).contiguous()
        video, _ = self._finalize_batched_video_tensor(
            video=gathered_video,
            image_is_pad=image_is_pad,
        )
        return video

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            image_is_pad = sample["image_is_pad"]
            has_pad = False
            if bool(image_is_pad.any().item()):
                has_pad = True
            if not self.video_only:
                action_is_pad = sample["action_is_pad"]
                proprio_is_pad = sample.get("proprio_is_pad", sample.get("state_is_pad"))
                if proprio_is_pad is None:
                    raise KeyError("Expected `proprio_is_pad` or `state_is_pad` in dataset sample.")
                if bool(action_is_pad.any().item()):
                    has_pad = True
                if bool(proprio_is_pad.any().item()):
                    has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        if self.use_latent_cache:
            # In latent-cache mode we bypass video decode/image transforms entirely and
            # keep only the original action/state/text preprocessing path.
            sample = self.processor.preprocess_without_images(sample)
            cache_sample_idx = self._resolve_sample_idx(sample)
            image_is_pad = sample["image_is_pad"][self.video_sample_indices]
            video_latents = self._load_cached_video_latents(cache_sample_idx)
        else:
            image_is_pad = sample["image_is_pad"]

            if self.video_only:
                video = self._build_video_from_raw_images(sample)
            else:
                video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
            video, image_is_pad = self._finalize_video_tensor(video=video, image_is_pad=image_is_pad)
            if self.video_only:
                return {
                    "idx": int(sample["idx"]),
                    "video": video,
                }

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if self.use_latent_cache:
            if image_is_pad.shape[0] <= 1:
                raise ValueError(
                    f"`image_is_pad` must have at least 2 frames, got shape {tuple(image_is_pad.shape)}"
                )
            transition_count = int(image_is_pad.shape[0] - 1)
        else:
            if video.shape[1] <= 1:
                raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
            transition_count = int(video.shape[1] - 1)
        if action.shape[0] % transition_count != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {transition_count}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "idx": int(sample["idx"]),
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.use_latent_cache:
            data["video_latents"] = video_latents
        else:
            data["video"] = video
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_dir,
            f"{hashed}.t5_len{self.context_len}.{self.text_embedding_cache_id}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            strict_dataset_errors = str(os.environ.get("WAM_ADAPTER_STRICT_DATASET_ERRORS", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            if strict_dataset_errors:
                raise
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
