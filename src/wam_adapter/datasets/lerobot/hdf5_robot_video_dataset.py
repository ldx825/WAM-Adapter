import bisect
import hashlib
import inspect
import io
import os
import re
import traceback
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from PIL import Image
from omegaconf import DictConfig

try:
    from hydra.utils import instantiate
except Exception:  # pragma: no cover - optional outside the training env
    instantiate = None

from .robot_video_dataset import DEFAULT_PROMPT
from .utils.normalizer import load_dataset_stats_from_json, save_dataset_stats_to_json
from ..dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from wam_adapter.utils import misc
from wam_adapter.utils.logging_config import get_logger

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5


def _decode_hdf5_image(encoded: np.ndarray) -> torch.Tensor:
    if not isinstance(encoded, np.ndarray):
        encoded = np.asarray(encoded, dtype=np.uint8)
    if encoded.dtype != np.uint8:
        encoded = encoded.astype(np.uint8, copy=False)
    image = Image.open(io.BytesIO(encoded.tobytes())).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode_(\d+)\.hdf5$", path.name)
    if match is None:
        return (10**12, path.name)
    return (int(match.group(1)), path.name)


def _to_python_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _model_id_to_cache_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


class HDF5RobotVideoDataset(torch.utils.data.Dataset):
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
        seed: int = 42,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
        use_latent_cache: bool = False,
        latent_cache_dir: Optional[str] = None,
        video_only: bool = False,
        video_backend: Optional[str] = None,
        **_unused_kwargs,
    ):
        del video_backend, skip_padding_as_possible  # Not used in the HDF5 path.
        if camera_key is not None:
            raise ValueError("`camera_key` is not supported for HDF5RobotVideoDataset.")
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

        self.dataset_dirs = [str(Path(ds).expanduser().resolve()) for ds in dataset_dirs]
        self.shape_meta = shape_meta
        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.global_sample_stride = int(global_sample_stride)
        self.context_len = int(context_len)
        self.video_size = list(video_size)
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.text_embedding_cache_id = _model_id_to_cache_id(text_embedding_model_id)
        self.max_padding_retry = int(max_padding_retry)
        self.concat_multi_camera = str(concat_multi_camera)
        self.override_instruction = override_instruction
        self.processor = None
        self._latent_cache_format = "single_file_v1"
        self._latent_cache_shard_paths = None
        self._latent_cache_sample_to_shard = None
        self._latent_cache_sample_to_offset = None
        self._latent_cache_last_shard_relpath = None
        self._latent_cache_last_shard_payload = None

        assert (self.num_frames - 1) % self.action_video_freq_ratio == 0, (
            f"num_frames-1 must be divisible by action_video_freq_ratio, got "
            f"{self.num_frames - 1} and {self.action_video_freq_ratio}"
        )
        assert ((self.num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, (
            f"video frames must be divisible by 4 for tokenization, "
            f"got {(self.num_frames - 1) // self.action_video_freq_ratio}"
        )
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))

        self.image_meta = list(shape_meta["images"])
        self.state_meta = list(shape_meta["state"])
        self.action_meta = list(shape_meta["action"])
        if len(self.state_meta) != 1 or len(self.action_meta) != 1:
            raise ValueError("HDF5RobotVideoDataset currently expects exactly one state key and one action key.")

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        self.episodes = self._build_episode_index(
            val_set_proportion=float(val_set_proportion),
            is_training_set=bool(is_training_set),
            seed=int(seed),
        )
        self._sample_ends = [int(ep["sample_end"]) for ep in self.episodes]
        if not self.episodes:
            raise ValueError(f"No valid episodes found for dataset_dirs={self.dataset_dirs}")

        if processor is not None:
            if isinstance(processor, DictConfig):
                if instantiate is None:
                    raise ImportError("hydra is required to instantiate processor from DictConfig.")
                processor = instantiate(processor)
            self.processor = processor
            if is_training_set:
                self.processor.train()
            else:
                self.processor.eval()

            if not self.video_only:
                if not pretrained_norm_stats:
                    if not is_training_set:
                        raise ValueError(
                            "pretrained_norm_stats must be provided for validation/test sets "
                            "since we don't want to calculate stats on them."
                        )
                    if PartialState().is_main_process:
                        logger.info("Calculating HDF5 dataset stats for normalization...")
                        dataset_stats = self.get_dataset_stats(processor)
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
                    logger.info("Using dataset stats: %s", pretrained_norm_stats)
                    if PartialState().is_main_process:
                        work_dir = misc.get_work_dir()
                        save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

                processor.set_normalizer_from_stats(dataset_stats)

        if self.use_latent_cache:
            if self.processor is None:
                raise ValueError("`processor` is required when `use_latent_cache=true`.")
            if not hasattr(self.processor, "preprocess_without_images"):
                raise ValueError(
                    "`processor` must implement `preprocess_without_images()` when "
                    "`use_latent_cache=true`."
                )
            self._init_latent_cache_reader()
            logger.info("Using latent cache for HDF5RobotVideoDataset: %s", self.latent_cache_dir)

    def __len__(self):
        return int(self.episodes[-1]["sample_end"])

    def _build_episode_index(
        self,
        *,
        val_set_proportion: float,
        is_training_set: bool,
        seed: int,
    ) -> list[dict[str, Any]]:
        selected_paths: list[Path] = []
        for ds_dir in self.dataset_dirs:
            root = Path(ds_dir)
            if not root.exists():
                raise FileNotFoundError(f"Dataset directory not found: {root}")
            all_paths = sorted(root.glob("episode_*.hdf5"), key=_episode_sort_key)
            if not all_paths:
                raise FileNotFoundError(f"No episode_*.hdf5 files found under: {root}")
            if val_set_proportion < 1e-6:
                chosen = all_paths
            else:
                rng = np.random.default_rng(seed)
                indices = np.arange(len(all_paths))
                rng.shuffle(indices)
                split_idx = int(len(indices) * (1.0 - val_set_proportion))
                picked = indices[:split_idx] if is_training_set else indices[split_idx:]
                chosen = [all_paths[int(i)] for i in picked.tolist()]
                chosen.sort(key=_episode_sort_key)
            selected_paths.extend(chosen)

        episodes: list[dict[str, Any]] = []
        sample_start = 0
        for episode_path in selected_paths:
            episode_meta = self._read_episode_meta(episode_path)
            sample_count = int(episode_meta["sample_count"])
            if sample_count <= 0:
                logger.warning("Skipping short episode with no valid training windows: %s", episode_path)
                continue
            episode_meta["sample_start"] = int(sample_start)
            episode_meta["sample_end"] = int(sample_start + sample_count)
            episodes.append(episode_meta)
            sample_start += sample_count
        return episodes

    def _read_episode_meta(self, episode_path: Path) -> dict[str, Any]:
        with h5py.File(str(episode_path), "r") as f:
            if "action" not in f:
                raise KeyError(f"Missing `action` dataset in {episode_path}")
            if "observation/state" not in f:
                raise KeyError(f"Missing `observation/state` dataset in {episode_path}")
            if "observation/images" not in f:
                raise KeyError(f"Missing `observation/images` group in {episode_path}")
            image_group = f["observation/images"]
            camera_keys = [meta["key"] for meta in self.image_meta]
            missing = [key for key in camera_keys if key not in image_group]
            if missing:
                raise KeyError(f"Missing camera datasets in {episode_path}: {missing}")

            state_len = int(f["observation/state"].shape[0])
            action_len = int(f["action"].shape[0])
            image_len = min(int(image_group[key].shape[0]) for key in camera_keys)
            obs_valid_starts = min(state_len, image_len) - (self.num_frames - 1) * self.global_sample_stride
            action_valid_starts = action_len - (self.num_frames - 2) * self.global_sample_stride
            sample_count = max(min(obs_valid_starts, action_valid_starts), 0)
            task = _to_python_str(f.attrs.get("task", ""))
            if task.strip() == "":
                raise ValueError(f"Missing non-empty `task` attribute in {episode_path}")

        return {
            "episode_path": str(episode_path),
            "task": task,
            "state_len": state_len,
            "action_len": action_len,
            "image_len": image_len,
            "sample_count": int(sample_count),
        }

    def _resolve_episode_for_sample(self, sample_idx: int) -> tuple[dict[str, Any], int]:
        if sample_idx < 0 or sample_idx >= len(self):
            raise IndexError(f"Index {sample_idx} out of bounds [0, {len(self)}).")
        episode_pos = bisect.bisect_right(self._sample_ends, int(sample_idx))
        episode = self.episodes[episode_pos]
        local_start = int(sample_idx - episode["sample_start"])
        return episode, local_start

    def _state_indices_for_local_start(self, local_start: int) -> np.ndarray:
        return np.arange(self.num_frames, dtype=np.int64) * self.global_sample_stride + int(local_start)

    def _action_indices_for_local_start(self, local_start: int) -> np.ndarray:
        return np.arange(self.num_frames - 1, dtype=np.int64) * self.global_sample_stride + int(local_start)

    def _load_action_state_window(self, episode: dict[str, Any], local_start: int) -> tuple[torch.Tensor, torch.Tensor]:
        state_indices = self._state_indices_for_local_start(local_start)
        action_indices = self._action_indices_for_local_start(local_start)
        with h5py.File(episode["episode_path"], "r") as f:
            state = np.asarray(f["observation/state"][state_indices], dtype=np.float32)
            action = np.asarray(f["action"][action_indices], dtype=np.float32)
        return (
            torch.from_numpy(state).contiguous(),
            torch.from_numpy(action).contiguous(),
        )

    def _load_window_raw_images(self, episode: dict[str, Any], frame_indices: np.ndarray) -> dict[str, torch.Tensor]:
        images: dict[str, torch.Tensor] = {}
        with h5py.File(episode["episode_path"], "r") as f:
            image_group = f["observation/images"]
            for meta in self.image_meta:
                key = meta["key"]
                frames = [_decode_hdf5_image(image_group[key][int(frame_idx)]) for frame_idx in frame_indices.tolist()]
                images[key] = torch.stack(frames, dim=0).contiguous()
        return images

    def _load_episode_arrays_for_stats(self, episode: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(episode["episode_path"], "r") as f:
            state = torch.from_numpy(np.asarray(f["observation/state"], dtype=np.float32)).contiguous()
            action = torch.from_numpy(np.asarray(f["action"], dtype=np.float32)).contiguous()
        return state, action

    def _build_video_from_raw_images(self, sample: dict[str, Any]) -> torch.Tensor:
        if self.processor is None:
            raise ValueError("`processor` must be initialized.")
        if not hasattr(self.processor, "build_pixel_values_from_images"):
            raise ValueError("`processor` must implement `build_pixel_values_from_images()`.")
        return self.processor.build_pixel_values_from_images(sample)

    def _build_episode_pixel_values_from_raw_images(self, episode_images: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.processor is None:
            raise ValueError("`processor` must be initialized.")
        if not hasattr(self.processor, "build_pixel_values_from_episode_images"):
            raise ValueError("`processor` must implement `build_pixel_values_from_episode_images()`.")
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
                raise ValueError(f"Expected video [T,C,H,W] or [N,T,C,H,W], got {tuple(video.shape)}")
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
            raise ValueError(f"Expected video [B,T,C,H,W] or [B,N,T,C,H,W], got {tuple(video.shape)}")
        if not isinstance(image_is_pad, torch.Tensor):
            image_is_pad = torch.as_tensor(image_is_pad, dtype=torch.bool)
        image_is_pad = image_is_pad.to(dtype=torch.bool)

        def _apply_framewise_transform(video_btc_hw: torch.Tensor, transform_fn) -> torch.Tensor:
            batch_size_local, time_local, channels_local, height_local, width_local = video_btc_hw.shape
            flat_video = video_btc_hw.reshape(batch_size_local * time_local, channels_local, height_local, width_local)
            flat_video = transform_fn(flat_video)
            out_channels, out_height, out_width = flat_video.shape[1:]
            return flat_video.reshape(batch_size_local, time_local, out_channels, out_height, out_width)

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
        elif video.ndim != 5:
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
        return len(self.episodes)

    def get_episode_sample_range(self, episode_idx: int) -> tuple[int, int]:
        if episode_idx < 0 or episode_idx >= len(self.episodes):
            raise IndexError(f"Episode index {episode_idx} out of bounds [0, {len(self.episodes)}).")
        episode = self.episodes[episode_idx]
        return int(episode["sample_start"]), int(episode["sample_end"])

    def load_episode_raw_images(self, episode_idx: int) -> dict[str, Any]:
        if episode_idx < 0 or episode_idx >= len(self.episodes):
            raise IndexError(f"Episode index {episode_idx} out of bounds [0, {len(self.episodes)}).")
        episode = self.episodes[episode_idx]
        with h5py.File(episode["episode_path"], "r") as f:
            image_group = f["observation/images"]
            episode_images = {}
            for meta in self.image_meta:
                key = meta["key"]
                frames = [_decode_hdf5_image(image_group[key][frame_idx]) for frame_idx in range(int(episode["image_len"]))]
                episode_images[key] = torch.stack(frames, dim=0).contiguous()
        return {
            "episode_index": int(episode_idx),
            "sample_start": int(episode["sample_start"]),
            "sample_end": int(episode["sample_end"]),
            "images": episode_images,
        }

    def build_processed_episode_pixel_values(self, episode_images: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._build_episode_pixel_values_from_raw_images(episode_images).contiguous()

    def build_video_only_batch_from_processed_episode_pixel_values(
        self,
        processed_pixel_values: torch.Tensor,
        sample_indices: list[int] | torch.Tensor,
        episode_sample_start: int,
        episode_sample_end: int,
    ) -> torch.Tensor:
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.to(dtype=torch.int64).view(-1)
        else:
            sample_indices = torch.tensor(sample_indices, dtype=torch.int64)
        if sample_indices.numel() == 0:
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
        local_indices = sample_indices - int(episode_sample_start)
        if bool((local_indices < 0).any().item()) or bool(
            (sample_indices >= int(episode_sample_end)).any().item()
        ):
            raise ValueError(f"Sample indices must lie within [{episode_sample_start}, {episode_sample_end}).")

        step_offsets = torch.arange(self.num_frames, dtype=torch.int64) * self.global_sample_stride
        frame_indices = local_indices.unsqueeze(1) + step_offsets.unsqueeze(0)
        image_is_pad = torch.zeros_like(frame_indices, dtype=torch.bool)
        if int(frame_indices.max().item()) >= int(processed_pixel_values.shape[1]):
            raise ValueError(
                f"Episode frame index out of bounds: max={int(frame_indices.max().item())} "
                f"vs processed length={processed_pixel_values.shape[1]}"
            )

        gathered_video = processed_pixel_values[:, frame_indices]
        gathered_video = gathered_video.permute(1, 0, 2, 3, 4, 5).contiguous()
        video, _ = self._finalize_batched_video_tensor(video=gathered_video, image_is_pad=image_is_pad)
        return video

    def _build_raw_window_sample(self, sample_idx: int) -> dict[str, Any]:
        episode, local_start = self._resolve_episode_for_sample(sample_idx)
        frame_indices = self._state_indices_for_local_start(local_start)
        state, action = self._load_action_state_window(episode, local_start)
        sample = {
            "idx": int(sample_idx),
            "task": episode["task"],
            "action": {self.action_meta[0]["key"]: action},
            "state": {self.state_meta[0]["key"]: state},
            "image_is_pad": torch.zeros(self.num_frames, dtype=torch.bool),
            "state_is_pad": torch.zeros(self.num_frames, dtype=torch.bool),
            "action_is_pad": torch.zeros(self.num_frames - 1, dtype=torch.bool),
        }
        if not self.use_latent_cache:
            sample["images"] = self._load_window_raw_images(episode, frame_indices)
        return sample

    def _episode_windows_for_stats(self, episode: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
        state, action = self._load_episode_arrays_for_stats(episode)
        sample_count = int(episode["sample_count"])
        state_windows = []
        action_windows = []
        for local_start in range(sample_count):
            state_windows.append(state[self._state_indices_for_local_start(local_start)])
            action_windows.append(action[self._action_indices_for_local_start(local_start)])
        return {
            "state": {self.state_meta[0]["key"]: torch.stack(state_windows, dim=0).float()},
            "action": {self.action_meta[0]["key"]: torch.stack(action_windows, dim=0).float()},
        }

    def get_dataset_stats(self, preprocessor) -> dict[str, Any]:
        from collections import defaultdict

        state_min = defaultdict(list)
        state_max = defaultdict(list)
        state_mean = defaultdict(list)
        state_var = defaultdict(list)
        state_q01 = defaultdict(list)
        state_q99 = defaultdict(list)

        action_min = defaultdict(list)
        action_max = defaultdict(list)
        action_mean = defaultdict(list)
        action_var = defaultdict(list)
        action_q01 = defaultdict(list)
        action_q99 = defaultdict(list)

        for episode_idx, episode in enumerate(self.episodes):
            batch = self._episode_windows_for_stats(episode)
            batch = preprocessor.action_state_transform(batch)

            for meta in self.state_meta:
                key = meta["key"]
                cur_state: torch.Tensor = batch["state"][key]
                state_min[key].append(cur_state.amin(0))
                state_max[key].append(cur_state.amax(0))
                state_mean[key].append(cur_state.mean(0))
                state_var[key].append(cur_state.var(0))
                state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))

            for meta in self.action_meta:
                key = meta["key"]
                cur_action: torch.Tensor = batch["action"][key]
                action_min[key].append(cur_action.amin(0))
                action_max[key].append(cur_action.amax(0))
                action_mean[key].append(cur_action.mean(0))
                action_var[key].append(cur_action.var(0))
                action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))

            if episode_idx % 10 == 0:
                logger.info(
                    "Stats progress: episode=%d/%d task=%s windows=%d",
                    episode_idx + 1,
                    len(self.episodes),
                    episode["task"],
                    int(episode["sample_count"]),
                )

        def get_mean_std(means, vars):
            means = torch.stack(means)
            vars = torch.stack(vars)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {"state": defaultdict(dict), "action": defaultdict(dict), "num_episodes": len(self.episodes), "num_transition": len(self)}
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])

        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            (
                stats["action"][key]["stepwise_mean"],
                stats["action"][key]["stepwise_std"],
                stats["action"][key]["global_mean"],
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])
        return stats

    def _init_latent_cache_reader(self):
        if self.latent_cache_dir is None:
            raise ValueError("`latent_cache_dir` is not set.")
        index_path = os.path.join(self.latent_cache_dir, "index.pt")
        if not os.path.exists(index_path):
            self._latent_cache_format = "single_file_v1"
            return

        index_payload = torch.load(index_path, map_location="cpu")
        if not isinstance(index_payload, dict):
            raise TypeError(f"Latent cache index must be a dict, got {type(index_payload)} in {index_path}.")
        storage_format = str(index_payload.get("storage_format", "sharded_v1"))
        shard_paths = index_payload.get("shard_paths")
        sample_to_shard = index_payload.get("sample_to_shard")
        sample_to_offset = index_payload.get("sample_to_offset")
        if not isinstance(shard_paths, list):
            raise TypeError(f"`shard_paths` must be a list in {index_path}.")
        if not isinstance(sample_to_shard, torch.Tensor) or not isinstance(sample_to_offset, torch.Tensor):
            raise TypeError(f"`sample_to_shard` and `sample_to_offset` must be tensors in {index_path}.")

        if storage_format not in {"sharded_v1", "episode_packed_v1"}:
            raise ValueError(f"Unsupported indexed latent cache format `{storage_format}` in {index_path}.")
        self._latent_cache_format = storage_format
        self._latent_cache_shard_paths = [str(path) for path in shard_paths]
        self._latent_cache_sample_to_shard = sample_to_shard.to(dtype=torch.int64, device="cpu").contiguous()
        self._latent_cache_sample_to_offset = sample_to_offset.to(dtype=torch.int64, device="cpu").contiguous()
        expected_num_samples = int(len(self))
        cached_num_samples = int(self._latent_cache_sample_to_shard.shape[0])
        if cached_num_samples != expected_num_samples:
            raise ValueError(
                "Latent cache sample count does not match dataset length: "
                f"cache={cached_num_samples} dataset={expected_num_samples}. "
                "This usually means the cache was precomputed with a different split "
                "(e.g. old val_set_proportion) or different dataset_dirs/num_frames settings. "
                "Re-run experiments/real_robot/prepare_real_robot_hdf5.sh with matching training config."
            )

    def _get_latent_cache_path(self, sample_idx: int) -> str:
        if self.latent_cache_dir is None:
            raise ValueError("`latent_cache_dir` is not set.")
        return os.path.join(self.latent_cache_dir, f"{int(sample_idx):08d}.pt")

    @staticmethod
    def _validate_video_latents(video_latents: torch.Tensor, source_path: str) -> torch.Tensor:
        if not isinstance(video_latents, torch.Tensor):
            raise TypeError(f"`video_latents` must be a torch.Tensor, got {type(video_latents)} in {source_path}.")
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
            raise TypeError(f"Unsupported latent cache payload type {type(payload)} in {cache_path}.")
        return self._validate_video_latents(video_latents, cache_path)

    def _load_cached_video_latents_from_shard(self, sample_idx: int) -> torch.Tensor:
        if self._latent_cache_sample_to_shard is None or self._latent_cache_sample_to_offset is None:
            raise RuntimeError("Sharded latent cache index was not initialized.")
        shard_id = int(self._latent_cache_sample_to_shard[sample_idx].item())
        offset = int(self._latent_cache_sample_to_offset[sample_idx].item())
        if shard_id < 0 or offset < 0:
            raise FileNotFoundError(
                f"Missing sharded latent cache entry for sample idx={sample_idx}: {self.latent_cache_dir}"
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
        video_latents = shard_payload["video_latents"]
        return self._validate_video_latents(video_latents[offset], self._latent_cache_last_shard_relpath)

    def _load_cached_video_latents(self, sample_idx: int) -> torch.Tensor:
        if self._latent_cache_format in {"sharded_v1", "episode_packed_v1"}:
            return self._load_cached_video_latents_from_shard(sample_idx)
        return self._load_cached_video_latents_from_single_file(sample_idx)

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
        return context, context_mask

    def _get(self, idx: int):
        sample_idx = int(idx)
        sample = self._build_raw_window_sample(sample_idx)
        if self.use_latent_cache:
            sample = self.processor.preprocess_without_images(sample)
            image_is_pad = sample["image_is_pad"][self.video_sample_indices]
            video_latents = self._load_cached_video_latents(sample_idx)
        else:
            image_is_pad = sample["image_is_pad"]
            if self.video_only:
                video = self._build_video_from_raw_images(sample)
            else:
                sample = self.processor.preprocess(sample)
                video = sample["pixel_values"]
            video, image_is_pad = self._finalize_video_tensor(video=video, image_is_pad=image_is_pad)
            if self.video_only:
                return {"idx": int(sample_idx), "video": video}

        if not self.use_latent_cache:
            action = sample["action"]
            proprio = sample["proprio"][:-1, :]
        else:
            action = sample["action"]
            proprio = sample["proprio"][:-1, :]

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        data = {
            "idx": int(sample_idx),
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

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
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
            data = self._get(int(random_idx))
        return data
