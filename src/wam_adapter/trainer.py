import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler, TaskEpisodeWindowSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        warmup_steps = cfg.get("warmup_steps")
        self.warmup_steps = int(warmup_steps) if warmup_steps is not None else None
        self.warmup_ratio = float(cfg.get("warmup_ratio", 0.05))
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError(f"`warmup_steps` must be >= 0, got {self.warmup_steps}.")
        if self.warmup_ratio < 0:
            raise ValueError(f"`warmup_ratio` must be >= 0, got {self.warmup_ratio}.")
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        scheduler_total_steps = cfg.get("lr_scheduler_total_steps")
        self.lr_scheduler_total_steps = (
            None if scheduler_total_steps is None else int(scheduler_total_steps)
        )
        self.retarget_lr_scheduler_on_resume = bool(
            cfg.get("retarget_lr_scheduler_on_resume", False)
        )
        if self.lr_scheduler_total_steps is not None and self.lr_scheduler_total_steps <= 0:
            raise ValueError(
                "`lr_scheduler_total_steps` must be positive when set, got "
                f"{self.lr_scheduler_total_steps}."
            )
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.resume_path = self._resolve_resume_path()
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        timing_cfg = cfg.get("timing_breakdown", {})
        self.timing_breakdown_enabled = bool(timing_cfg.get("enabled", False))
        self.timing_breakdown_sync_cuda = bool(timing_cfg.get("sync_cuda", True))
        self._timing_accumulator: dict[str, float] = {}
        train_vis_cfg = cfg.get("train_visualization", {})
        self.train_visualization_enabled = bool(train_vis_cfg.get("enabled", False))
        self.train_visualization_every = int(train_vis_cfg.get("every", 0))
        self.train_visualization_fps = int(train_vis_cfg.get("fps", 8))
        self.train_visualization_tiled = bool(train_vis_cfg.get("tiled", False))
        self.train_action_fit_enabled = bool(train_vis_cfg.get("action_fit_enabled", False))
        action_fit_num_steps = train_vis_cfg.get("action_fit_num_steps", None)
        self.train_action_fit_num_steps = (
            None
            if action_fit_num_steps in (None, "", "null")
            else int(action_fit_num_steps)
        )
        if self.train_action_fit_num_steps is not None and self.train_action_fit_num_steps <= 0:
            raise ValueError(
                "`train_visualization.action_fit_num_steps` must be positive when provided, "
                f"got {self.train_action_fit_num_steps}."
            )
        parameter_report_cfg = cfg.get("parameter_report", {})
        self.parameter_report_enabled = bool(parameter_report_cfg.get("enabled", False))
        self.parameter_report_filename = (
            str(parameter_report_cfg.get("filename", "parameter_report.json")).strip()
            or "parameter_report.json"
        )
        benchmark_cfg = cfg.get("benchmark", {})
        self.benchmark_enabled = bool(benchmark_cfg.get("enabled", False))
        self.benchmark_warmup_steps = int(benchmark_cfg.get("warmup_steps", 10))
        self.benchmark_measure_steps = int(benchmark_cfg.get("measure_steps", 50))
        self.benchmark_output_filename = (
            str(benchmark_cfg.get("output_filename", "training_speed_benchmark.json")).strip()
            or "training_speed_benchmark.json"
        )
        benchmark_description = benchmark_cfg.get("description")
        self.benchmark_description = None if benchmark_description in (None, "", "null") else str(benchmark_description)
        self._benchmark_start_time: float | None = None
        self.wandb_enabled = bool(cfg.wandb.enabled)
        if self.benchmark_enabled:
            if self.benchmark_warmup_steps < 0:
                raise ValueError(
                    f"`benchmark.warmup_steps` must be >= 0, got {self.benchmark_warmup_steps}."
                )
            if self.benchmark_measure_steps <= 0:
                raise ValueError(
                    f"`benchmark.measure_steps` must be > 0, got {self.benchmark_measure_steps}."
                )
            self.benchmark_total_steps = self.benchmark_warmup_steps + self.benchmark_measure_steps
            if self.max_steps is None or int(self.max_steps) != int(self.benchmark_total_steps):
                logger.info(
                    "Benchmark mode enabled: overriding max_steps from %s to %d "
                    "(warmup=%d measure=%d).",
                    self.max_steps,
                    self.benchmark_total_steps,
                    self.benchmark_warmup_steps,
                    self.benchmark_measure_steps,
                )
                self.max_steps = int(self.benchmark_total_steps)
        else:
            self.benchmark_total_steps = None

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        logger.info(
            "Timing breakdown: enabled=%s sync_cuda=%s",
            self.timing_breakdown_enabled,
            self.timing_breakdown_sync_cuda,
        )
        logger.info(
            "Train visualization: enabled=%s every=%d fps=%d tiled=%s action_fit_enabled=%s action_fit_num_steps=%s",
            self.train_visualization_enabled,
            self.train_visualization_every,
            self.train_visualization_fps,
            self.train_visualization_tiled,
            self.train_action_fit_enabled,
            self.train_action_fit_num_steps,
        )
        logger.info(
            "Parameter report: enabled=%s filename=%s",
            self.parameter_report_enabled,
            self.parameter_report_filename,
        )
        logger.info(
            "Benchmark mode: enabled=%s warmup_steps=%d measure_steps=%d output_filename=%s description=%s",
            self.benchmark_enabled,
            self.benchmark_warmup_steps,
            self.benchmark_measure_steps,
            self.benchmark_output_filename,
            self.benchmark_description,
        )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        if hasattr(self.model, "set_timing_breakdown"):
            self.model.set_timing_breakdown(
                enabled=self.timing_breakdown_enabled,
                sync_cuda=self.timing_breakdown_sync_cuda,
            )

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        self._maybe_load_weight_checkpoint_before_prepare()
        if hasattr(self.model, "log_parameter_summary"):
            self.model.log_parameter_summary()
        self.total_params = int(sum(param.numel() for param in self.model.parameters()))
        self.trainable_params = int(
            sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        )
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters were left after applying the model freeze policy.")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        stop_train_steps = self._estimate_total_train_steps()
        self.max_steps = stop_train_steps
        scheduler_total_steps = (
            stop_train_steps
            if self.lr_scheduler_total_steps is None
            else self.lr_scheduler_total_steps
        )
        if scheduler_total_steps < stop_train_steps:
            raise ValueError(
                "`lr_scheduler_total_steps` cannot be smaller than `max_steps`: "
                f"{scheduler_total_steps} < {stop_train_steps}."
            )
        warmup_steps = (
            self.warmup_steps
            if self.warmup_steps is not None
            else int(scheduler_total_steps * self.warmup_ratio)
        )
        logger.info(
            "Scheduler: type=%s invocation_stop_step=%d total_train_steps=%d "
            "warmup_steps=%d warmup_ratio=%.4f",
            cfg.lr_scheduler_type,
            stop_train_steps,
            scheduler_total_steps,
            int(warmup_steps),
            self.warmup_ratio,
        )
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=scheduler_total_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        self.train_vis_dir = os.path.join(self.output_dir, "train_vis")
        self.action_fit_dir = os.path.join(self.output_dir, "action_fit")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)
        ensure_dir(self.train_vis_dir)
        ensure_dir(self.action_fit_dir)
        self._maybe_save_parameter_report(self.model)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        prepared_model = self.accelerator.unwrap_model(self.model)
        if hasattr(prepared_model, "set_timing_breakdown"):
            prepared_model.set_timing_breakdown(
                enabled=self.timing_breakdown_enabled,
                sync_cuda=self.timing_breakdown_sync_cuda,
            )
        self.wandb_run = None
        self._init_wandb()
        self._resume_after_prepare()
        self.optimizer.zero_grad(set_to_none=True)

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    @staticmethod
    def _slice_sample_batch(sample, batch_size: int = 1):
        if isinstance(sample, torch.Tensor):
            if sample.ndim == 0:
                return sample
            return sample[:batch_size]
        if isinstance(sample, dict):
            return {key: Wan22Trainer._slice_sample_batch(value, batch_size=batch_size) for key, value in sample.items()}
        if isinstance(sample, list):
            return sample[:batch_size]
        if isinstance(sample, tuple):
            return sample[:batch_size]
        return sample

    def _maybe_save_train_visualization(self, sample) -> Optional[str]:
        if not self.train_visualization_enabled:
            return None
        if self.train_visualization_every <= 0:
            return None
        if self.global_step % self.train_visualization_every != 0:
            return None
        if not self.accelerator.is_main_process:
            return None

        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "render_training_visualization"):
            return None

        vis_sample = self._slice_sample_batch(sample, batch_size=1)
        vis_payload = model.render_training_visualization(
            vis_sample,
            tiled=self.train_visualization_tiled,
        )
        frames = vis_payload.get("frames") if isinstance(vis_payload, dict) else None
        if not frames:
            return None

        output_path = os.path.join(self.train_vis_dir, f"step_{self.global_step:06d}.mp4")
        save_mp4(frames, output_path, fps=self.train_visualization_fps)
        return output_path

    def _save_action_fit_plot(
        self,
        *,
        pred_action: torch.Tensor,
        gt_action: torch.Tensor,
    ) -> str:
        if pred_action.ndim != 2 or gt_action.ndim != 2:
            raise ValueError(
                f"`pred_action` and `gt_action` must be [T, D], got {tuple(pred_action.shape)} and {tuple(gt_action.shape)}"
            )
        if pred_action.shape != gt_action.shape:
            raise ValueError(
                f"Action plot shape mismatch: pred={tuple(pred_action.shape)} vs gt={tuple(gt_action.shape)}"
            )

        pred = pred_action.detach().to(device="cpu", dtype=torch.float32).numpy()
        gt = gt_action.detach().to(device="cpu", dtype=torch.float32).numpy()
        horizon, action_dim = pred.shape
        num_cols = 4
        num_rows = int(ceil(action_dim / num_cols))
        margin = 24
        gap = 18
        tile_w = 240
        tile_h = 150
        title_h = 44
        canvas_w = margin * 2 + num_cols * tile_w + (num_cols - 1) * gap
        canvas_h = margin * 2 + title_h + num_rows * tile_h + (num_rows - 1) * gap
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, margin), f"Action Fit | step={self.global_step}", fill=(0, 0, 0))
        draw.text((margin, margin + 18), f"Blue: GT   Red: Pred   Horizon: {horizon}", fill=(60, 60, 60))

        plot_y0 = margin + title_h
        for dim_idx in range(action_dim):
            row = dim_idx // num_cols
            col = dim_idx % num_cols
            x0 = margin + col * (tile_w + gap)
            y0 = plot_y0 + row * (tile_h + gap)
            x1 = x0 + tile_w
            y1 = y0 + tile_h
            draw.rectangle((x0, y0, x1, y1), outline=(180, 180, 180), width=1)
            draw.text((x0 + 8, y0 + 6), f"dim {dim_idx}", fill=(0, 0, 0))

            gt_series = gt[:, dim_idx]
            pred_series = pred[:, dim_idx]
            y_min = float(min(gt_series.min(), pred_series.min()))
            y_max = float(max(gt_series.max(), pred_series.max()))
            if abs(y_max - y_min) < 1e-6:
                y_min -= 1.0
                y_max += 1.0
            pad = 0.08 * (y_max - y_min)
            y_min -= pad
            y_max += pad

            left = x0 + 12
            right = x1 - 12
            top = y0 + 28
            bottom = y1 - 12
            draw.line((left, bottom, right, bottom), fill=(160, 160, 160), width=1)
            draw.line((left, top, left, bottom), fill=(160, 160, 160), width=1)

            def _to_points(series: np.ndarray):
                pts = []
                for t_idx, value in enumerate(series.tolist()):
                    x = left if horizon <= 1 else left + (right - left) * (t_idx / (horizon - 1))
                    y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
                    pts.append((float(x), float(y)))
                return pts

            gt_points = _to_points(gt_series)
            pred_points = _to_points(pred_series)
            if len(gt_points) >= 2:
                draw.line(gt_points, fill=(50, 110, 240), width=2)
            if len(pred_points) >= 2:
                draw.line(pred_points, fill=(230, 60, 60), width=2)

        output_path = os.path.join(self.action_fit_dir, f"step_{self.global_step:06d}.png")
        canvas.save(output_path)
        return output_path

    @staticmethod
    def _resolve_dataset_processor(dataset):
        processor = getattr(dataset, "processor", None)
        if processor is not None:
            return processor
        inner_dataset = getattr(dataset, "lerobot_dataset", None)
        if inner_dataset is not None:
            inner_processor = getattr(inner_dataset, "processor", None)
            if inner_processor is not None:
                return inner_processor
        raise AttributeError(
            f"Failed to resolve processor from dataset type {type(dataset)}. "
            "Expected `.processor` or `.lerobot_dataset.processor`."
        )

    @staticmethod
    def _resize_video_frames(frames, *, width: int, height: int) -> list[Image.Image]:
        resized = []
        for frame in frames:
            image = frame.convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), resample=Image.BILINEAR)
            resized.append(image)
        return resized

    @torch.no_grad()
    def _render_readout_lowres_eval_video(
        self,
        *,
        model,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        num_inference_steps: int,
    ) -> list[Image.Image]:
        if not getattr(model, "uses_readout")():
            raise ValueError("Low-resolution eval video rendering only supports readout mode.")
        if not getattr(model, "_use_lowres_video_training_objective")():
            raise ValueError("Low-resolution eval video rendering requires low-res video training objective.")

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )

        input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
        first_frame_latents = model._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=False,
        )
        video_first_frame_latents, _ = model._maybe_downsample_video_latents_for_backbone(
            first_frame_latents
        )
        fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

        latent_t = (int(num_frames) - 1) // model.vae.temporal_downsample_factor + 1
        latent_h = int(video_first_frame_latents.shape[-2])
        latent_w = int(video_first_frame_latents.shape[-1])
        generator = torch.Generator(device="cpu").manual_seed(42)
        latents_video = torch.randn(
            (1, model.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)
        latents_video[:, :, 0:1] = video_first_frame_latents.clone()

        if prompt is not None:
            if context is not None or context_mask is not None:
                raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
            context, context_mask = model.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=model.device, dtype=torch.bool, non_blocking=True)

        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
            context, context_mask = model._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            action = action.to(device=model.device, dtype=model.torch_dtype)

        infer_timesteps_video, infer_deltas_video = model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=model.device,
            dtype=latents_video.dtype,
            shift_override=None,
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=model.device)
            pred_video = model._predict_video_only(
                latents_video=latents_video,
                timestep_video=timestep_video,
                context=context,
                context_mask=context_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_flag,
                apply_spatial_downsample=False,
                restore_spatial_resolution=False,
            )
            latents_video = model.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_video[:, :, 0:1] = video_first_frame_latents.clone()

        return model._decode_latents(latents_video, tiled=False)

    @staticmethod
    def _stitch_eval_rows(
        row_frames: list[list[Image.Image]],
        row_labels: list[str],
    ) -> list[Image.Image]:
        if len(row_frames) != len(row_labels):
            raise ValueError(
                f"Expected same number of row frames and labels, got {len(row_frames)} and {len(row_labels)}."
            )
        if not row_frames or not row_frames[0]:
            raise ValueError("`row_frames` must be non-empty.")

        num_frames = len(row_frames[0])
        if any(len(frames) != num_frames for frames in row_frames):
            raise ValueError("All rows must contain the same number of frames.")

        sample_frame = row_frames[0][0].convert("RGB")
        frame_w, frame_h = sample_frame.size
        label_w = 110
        stitched_frames: list[Image.Image] = []
        for frame_idx in range(num_frames):
            canvas = Image.new("RGB", (label_w + frame_w, frame_h * len(row_frames)), color=(255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            for row_idx, (frames, label) in enumerate(zip(row_frames, row_labels)):
                y0 = row_idx * frame_h
                frame = frames[frame_idx].convert("RGB")
                if frame.size != (frame_w, frame_h):
                    frame = frame.resize((frame_w, frame_h), resample=Image.BILINEAR)
                canvas.paste(frame, (label_w, y0))
                draw.text((12, y0 + 12), label, fill=(0, 0, 0))
                if row_idx > 0:
                    draw.line((0, y0, canvas.size[0], y0), fill=(210, 210, 210), width=1)
            stitched_frames.append(canvas)
        return stitched_frames

    @staticmethod
    def _make_param_count_bucket() -> dict[str, int]:
        return {
            "parameter_tensors": 0,
            "parameter_tensors_trainable": 0,
            "parameter_tensors_frozen": 0,
            "numel_total": 0,
            "numel_trainable": 0,
            "numel_frozen": 0,
        }

    @staticmethod
    def _accumulate_param_count(bucket: dict[str, int], *, numel: int, requires_grad: bool) -> None:
        bucket["parameter_tensors"] += 1
        bucket["numel_total"] += int(numel)
        if requires_grad:
            bucket["parameter_tensors_trainable"] += 1
            bucket["numel_trainable"] += int(numel)
        else:
            bucket["parameter_tensors_frozen"] += 1
            bucket["numel_frozen"] += int(numel)

    @staticmethod
    def _module_key(module_name: str) -> str:
        return "<root>" if module_name == "" else module_name

    def _build_parameter_report(self, model) -> dict:
        named_modules = {
            self._module_key(module_name): module.__class__.__name__
            for module_name, module in model.named_modules()
        }
        if "<root>" not in named_modules:
            named_modules["<root>"] = model.__class__.__name__

        module_stats: dict[str, dict[str, object]] = {}
        for module_key, module_type in named_modules.items():
            module_stats[module_key] = {
                "module_type": module_type,
                "direct": self._make_param_count_bucket(),
                "inclusive": self._make_param_count_bucket(),
            }

        total_counts = self._make_param_count_bucket()
        parameter_entries = []
        named_parameters = list(model.named_parameters())

        for name, param in named_parameters:
            numel = int(param.numel())
            requires_grad = bool(param.requires_grad)
            module_path, _, leaf_name = name.rpartition(".")
            module_key = self._module_key(module_path)
            if module_key not in module_stats:
                module_stats[module_key] = {
                    "module_type": "<unknown>",
                    "direct": self._make_param_count_bucket(),
                    "inclusive": self._make_param_count_bucket(),
                }

            self._accumulate_param_count(total_counts, numel=numel, requires_grad=requires_grad)
            self._accumulate_param_count(
                module_stats[module_key]["direct"],
                numel=numel,
                requires_grad=requires_grad,
            )

            ancestor_paths = [""]
            if module_path:
                prefix = ""
                for part in module_path.split("."):
                    prefix = part if prefix == "" else f"{prefix}.{part}"
                    ancestor_paths.append(prefix)
            for ancestor_path in ancestor_paths:
                ancestor_key = self._module_key(ancestor_path)
                if ancestor_key not in module_stats:
                    module_stats[ancestor_key] = {
                        "module_type": "<unknown>",
                        "direct": self._make_param_count_bucket(),
                        "inclusive": self._make_param_count_bucket(),
                    }
                self._accumulate_param_count(
                    module_stats[ancestor_key]["inclusive"],
                    numel=numel,
                    requires_grad=requires_grad,
                )

            parameter_entries.append(
                {
                    "name": name,
                    "module": module_key,
                    "leaf_name": leaf_name if leaf_name else name,
                    "shape": list(param.shape),
                    "numel": numel,
                    "dtype": str(param.dtype).replace("torch.", ""),
                    "requires_grad": requires_grad,
                    "status": "trainable" if requires_grad else "frozen",
                }
            )

        sorted_module_stats = {
            module_key: module_stats[module_key]
            for module_key in sorted(module_stats.keys())
        }

        return {
            "generated_at": datetime.now().isoformat(),
            "output_dir": self.output_dir,
            "model_type": model.__class__.__name__,
            "summary": total_counts,
            "modules": sorted_module_stats,
            "parameters": parameter_entries,
        }

    def _maybe_save_parameter_report(self, model) -> None:
        if not self.parameter_report_enabled:
            return
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        report_path = os.path.join(self.output_dir, self.parameter_report_filename)
        report_dir = os.path.dirname(report_path)
        if report_dir:
            ensure_dir(report_dir)
        report = self._build_parameter_report(model)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)
        logger.info("Saved parameter report to %s", report_path)
        self.accelerator.wait_for_everyone()

    def _build_loader(self, dataset, worker_init_fn=None):
        sampler_cfg = self.cfg.get("train_sampler")
        sampler_kind = "resumable_epoch" if sampler_cfg is None else str(
            sampler_cfg.get("kind", "resumable_epoch")
        )
        if sampler_kind == "task_episode_window":
            self.train_sampler = TaskEpisodeWindowSampler(
                dataset=dataset,
                manifest_path=str(sampler_cfg.manifest_path),
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
                expected_world_size=int(sampler_cfg.expected_world_size),
                samples_per_epoch=int(sampler_cfg.samples_per_epoch),
                expected_task_count=int(sampler_cfg.expected_task_count),
                episodes_per_task=int(sampler_cfg.episodes_per_task),
            )
            logger.info(
                "Using task-balanced sampler: tasks=%s samples_per_epoch=%d",
                self.train_sampler.tasks,
                len(self.train_sampler),
            )
        elif sampler_kind == "resumable_epoch":
            self.train_sampler = ResumableEpochSampler(
                dataset=dataset,
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
            )
        else:
            raise ValueError(f"Unsupported train_sampler.kind: {sampler_kind}")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _timing_sync(self):
        if not self.timing_breakdown_enabled:
            return
        if (
            self.timing_breakdown_sync_cuda
            and self.accelerator.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.accelerator.device)

    def _timing_start(self):
        if not self.timing_breakdown_enabled:
            return None
        self._timing_sync()
        return time.perf_counter()

    def _timing_end_ms(self, start_time):
        if start_time is None:
            return 0.0
        self._timing_sync()
        return float((time.perf_counter() - start_time) * 1000.0)

    def _accumulate_timing(self, metrics: dict[str, float]):
        if not self.timing_breakdown_enabled:
            return
        for key, value in metrics.items():
            self._timing_accumulator[key] = self._timing_accumulator.get(key, 0.0) + float(value)

    @staticmethod
    def _split_timing_metrics(metrics: dict) -> tuple[dict[str, float], dict[str, float]]:
        timing_metrics = {}
        other_metrics = {}
        for key, value in metrics.items():
            if str(key).startswith("timing/"):
                timing_metrics[str(key)] = float(value)
            else:
                other_metrics[key] = value
        return timing_metrics, other_metrics

    def _consume_timing_metrics(self) -> dict[str, float]:
        metrics = dict(self._timing_accumulator)
        self._timing_accumulator.clear()
        return metrics

    @staticmethod
    def _format_timing_log_line(metrics: dict[str, float]) -> str:
        ordered_keys = [
            "timing/trainer/data_wait_ms",
            "timing/trainer/forward_ms",
            "timing/trainer/backward_ms",
            "timing/trainer/optimizer_ms",
            "timing/trainer/step_total_ms",
            "timing/model/vae_encode_ms",
            "timing/model/build_inputs_ms",
            "timing/model/future_backbone_ms",
            "timing/model/future_decode_ms",
            "timing/model/action_backbone_ms",
            "timing/model/readout_ms",
            "timing/model/training_loss_total_ms",
        ]
        parts = []
        for key in ordered_keys:
            if key in metrics:
                parts.append(f"{key.split('/')[-1]}={metrics[key]:.1f}ms")
        for key in sorted(metrics.keys()):
            if key not in ordered_keys:
                parts.append(f"{key.split('/')[-1]}={metrics[key]:.1f}ms")
        return " ".join(parts)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _benchmark_sync(self) -> None:
        if self.accelerator.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.accelerator.device)
        self.accelerator.wait_for_everyone()
        if self.accelerator.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.accelerator.device)

    def _maybe_start_benchmark_window(self) -> None:
        if not self.benchmark_enabled or self._benchmark_start_time is not None:
            return
        if self.global_step != self.benchmark_warmup_steps:
            return
        self._benchmark_sync()
        if self.accelerator.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)
        self._benchmark_start_time = time.perf_counter()
        if self.accelerator.is_main_process:
            logger.info(
                "[benchmark] warmup complete at step=%d; starting measurement window for %d optimizer steps.",
                self.global_step,
                self.benchmark_measure_steps,
            )

    def _benchmark_peak_gpu_memory_bytes(self) -> tuple[int, int]:
        if self.accelerator.device.type != "cuda" or not torch.cuda.is_available():
            return 0, 0

        local_allocated = int(torch.cuda.max_memory_allocated(self.accelerator.device))
        local_reserved = int(torch.cuda.max_memory_reserved(self.accelerator.device))
        local_values = torch.tensor(
            [local_allocated, local_reserved],
            device=self.accelerator.device,
            dtype=torch.int64,
        )
        gathered_values = self.accelerator.gather(local_values)
        gathered_values = gathered_values.view(-1, 2)
        return (
            int(gathered_values[:, 0].max().item()),
            int(gathered_values[:, 1].max().item()),
        )

    def _finalize_benchmark(self) -> dict[str, object]:
        if self._benchmark_start_time is None:
            raise RuntimeError("Benchmark window was never started.")
        self._benchmark_sync()
        elapsed_sec = max(time.perf_counter() - self._benchmark_start_time, 1e-12)
        steps_per_sec = float(self.benchmark_measure_steps / elapsed_sec)
        micro_batch_global = int(self.batch_size * self.accelerator.num_processes)
        effective_global_batch = int(micro_batch_global * self.gradient_accumulation_steps)
        samples_per_sec = float(steps_per_sec * effective_global_batch)
        peak_memory_allocated_bytes, peak_memory_reserved_bytes = self._benchmark_peak_gpu_memory_bytes()
        payload: dict[str, object] = {
            "description": self.benchmark_description,
            "step_unit": "optimizer_step",
            "warmup_steps": int(self.benchmark_warmup_steps),
            "measure_steps": int(self.benchmark_measure_steps),
            "elapsed_sec": float(elapsed_sec),
            "steps_per_sec": steps_per_sec,
            "samples_per_sec": samples_per_sec,
            "batch_size_per_process": int(self.batch_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "num_processes": int(self.accelerator.num_processes),
            "micro_batch_global": micro_batch_global,
            "effective_global_batch_size": effective_global_batch,
            "total_params": int(self.total_params),
            "trainable_params": int(self.trainable_params),
            "total_params_billion": float(self.total_params / 1_000_000_000),
            "trainable_params_million": float(self.trainable_params / 1_000_000),
            "peak_gpu_mem_allocated_gb": float(peak_memory_allocated_bytes / (1024**3)),
            "peak_gpu_mem_reserved_gb": float(peak_memory_reserved_bytes / (1024**3)),
            "peak_gpu_mem_scope": "max_across_processes_per_gpu",
            "mixed_precision": str(self.accelerator.mixed_precision),
            "device": str(self.accelerator.device),
            "global_step": int(self.global_step),
            "output_dir": self.output_dir,
            "generated_at": datetime.now().isoformat(),
        }

        result_path = os.path.join(self.output_dir, self.benchmark_output_filename)
        if self.accelerator.is_main_process:
            ensure_dir(os.path.dirname(result_path) or self.output_dir)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
            logger.info(
                "[benchmark] saved=%s steps_per_sec=%.4f samples_per_sec=%.4f peak_mem=%.3fGiB elapsed=%.3fs",
                result_path,
                steps_per_sec,
                samples_per_sec,
                payload["peak_gpu_mem_allocated_gb"],
                elapsed_sec,
            )
        self.accelerator.wait_for_everyone()
        return payload

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resolve_resume_path(self) -> Optional[Path]:
        if not self.resume:
            return None
        resume_path = Path(str(self.resume))
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {self.resume}")
        return resume_path

    def _maybe_load_weight_checkpoint_before_prepare(self):
        if self.resume_path is None or self.resume_path.is_dir():
            return
        logger.info(
            "Loading weight checkpoint before optimizer/accelerator init: %s",
            self.resume,
        )
        self.model.load_checkpoint(str(self.resume_path), optimizer=None)
        logger.warning(
            "Loaded .pt weights before optimizer/prepare; optimizer/scheduler/step were not restored."
        )

    def _resume_after_prepare(self):
        if self.resume_path is None:
            return
        if self.resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", self.resume)
            self.load_training_state(str(self.resume_path))

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        if hasattr(model, "configure_trainable_modules"):
            model.configure_trainable_modules()
            return
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            if hasattr(model, "set_training_step"):
                model.set_training_step(self.global_step)
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio_step0 = (
            sample["proprio"][0, 0]
            if "proprio" in sample and sample["proprio"] is not None
            else None
        )  # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio_step0,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio_seq = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self._resolve_dataset_processor(self.val_dataset)

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio_seq,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

            if self.train_action_fit_enabled and self.accelerator.is_main_process:
                if self.train_action_fit_num_steps is not None:
                    fit_horizon = min(int(self.train_action_fit_num_steps), int(pred_action_denorm.shape[1]))
                    pred_action_plot = pred_action_denorm[:, :fit_horizon]
                    gt_action_plot = gt_action_denorm[:, :fit_horizon]
                else:
                    pred_action_plot = pred_action_denorm
                    gt_action_plot = gt_action_denorm
                action_fit_path = self._save_action_fit_plot(
                    pred_action=pred_action_plot.squeeze(0),
                    gt_action=gt_action_plot.squeeze(0),
                )
            else:
                action_fit_path = None
        else:
            action_fit_path = None

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        pred_row_frames = pred["video"]
        pred_row_label = "Pred"
        if (
            getattr(model, "uses_readout")()
            and getattr(model, "_use_lowres_video_training_objective")()
        ):
            pred_row_frames = self._render_readout_lowres_eval_video(
                model=model,
                prompt=infer_kwargs["prompt"],
                input_image=input_image,
                num_frames=num_frames,
                action=action,
                proprio=proprio_step0,
                context=(sample["context"][0] if sample["context"] is not None else None),
                context_mask=(sample["context_mask"][0] if sample["context_mask"] is not None else None),
                num_inference_steps=self.eval_num_inference_steps,
            )
            pred_row_label = "Pred (lowres)"
        pred_row_frames = self._resize_video_frames(
            pred_row_frames,
            width=int(video0.shape[3]),
            height=int(video0.shape[2]),
        )
        vae_row_frames = self._resize_video_frames(
            vae_recon_video,
            width=int(video0.shape[3]),
            height=int(video0.shape[2]),
        )
        gt_row_frames = []
        for t in range(gt_video_tensor.shape[1]):
            frame = (gt_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            gt_row_frames.append(Image.fromarray(frame))
        stitched_frames = self._stitch_eval_rows(
            [pred_row_frames, vae_row_frames, gt_row_frames],
            [pred_row_label, "VAE Recon", "GT"],
        )

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_fit_path is not None:
            result["action_fit_path"] = action_fit_path
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        if hasattr(self.train_sampler, "state_dict"):
            payload["sampler_state"] = self.train_sampler.state_dict()
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def _maybe_retarget_resumed_lr_scheduler(self):
        """Extend a resumed cosine schedule without introducing an LR jump.

        Accelerate restores the scheduler state verbatim, including the old
        ``CosineAnnealingLR.T_max``.  Merely launching a resumed segment with a
        larger ``lr_scheduler_total_steps`` therefore has no effect.  Updating
        only ``T_max`` after restore keeps the optimizer's current learning rate
        and the restored scheduler phase, then decays smoothly to ``eta_min`` at
        the newly requested global horizon.
        """
        if not self.retarget_lr_scheduler_on_resume:
            return
        if self.lr_scheduler_total_steps is None:
            raise ValueError(
                "`retarget_lr_scheduler_on_resume=true` requires "
                "`lr_scheduler_total_steps`."
            )

        scheduler = self.scheduler
        # Accelerate wraps the torch scheduler but keeps it on `.scheduler`.
        visited = set()
        while not isinstance(scheduler, (SequentialLR, CosineAnnealingLR)):
            scheduler_id = id(scheduler)
            if scheduler_id in visited or not hasattr(scheduler, "scheduler"):
                raise TypeError(
                    "Cannot retarget unsupported resumed scheduler wrapper: "
                    f"{type(scheduler).__name__}."
                )
            visited.add(scheduler_id)
            scheduler = scheduler.scheduler

        if isinstance(scheduler, SequentialLR):
            if not scheduler._schedulers:
                raise RuntimeError("Resumed SequentialLR contains no child schedulers.")
            cosine_scheduler = scheduler._schedulers[-1]
        else:
            cosine_scheduler = scheduler
        if not isinstance(cosine_scheduler, CosineAnnealingLR):
            raise TypeError(
                "Scheduler retargeting currently supports a cosine main scheduler, got "
                f"{type(cosine_scheduler).__name__}."
            )

        warmup_steps = 0 if self.warmup_steps is None else int(self.warmup_steps)
        new_t_max = max(int(self.lr_scheduler_total_steps) - warmup_steps, 1)
        old_t_max = int(cosine_scheduler.T_max)
        if new_t_max < int(cosine_scheduler.last_epoch):
            raise ValueError(
                "Retargeted cosine horizon is behind the restored scheduler phase: "
                f"T_max={new_t_max}, last_epoch={cosine_scheduler.last_epoch}."
            )
        if new_t_max == old_t_max:
            logger.info(
                "Resumed cosine scheduler already targets T_max=%d; current_lr=%s.",
                new_t_max,
                [group["lr"] for group in self.optimizer.param_groups],
            )
            return

        cosine_scheduler.T_max = new_t_max
        logger.warning(
            "Retargeted resumed cosine scheduler T_max=%d -> %d while preserving "
            "last_epoch=%d and current_lr=%s.",
            old_t_max,
            new_t_max,
            cosine_scheduler.last_epoch,
            [group["lr"] for group in self.optimizer.param_groups],
        )

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        self._maybe_retarget_resumed_lr_scheduler()
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                sampler_state = payload.get("sampler_state")
                if sampler_state is not None and hasattr(self.train_sampler, "load_state_dict"):
                    self.train_sampler.load_state_dict(sampler_state)
                else:
                    self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        if self.benchmark_enabled and self.benchmark_warmup_steps == 0:
            self._benchmark_sync()
            if self.accelerator.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.accelerator.device)
            self._benchmark_start_time = time.perf_counter()
            if self.accelerator.is_main_process:
                logger.info(
                    "[benchmark] starting measurement immediately for %d optimizer steps.",
                    self.benchmark_measure_steps,
                )

        while self.global_step < self.max_steps:
            try:
                data_wait_start = self._timing_start()
                sample = next(data_iter)
                self._accumulate_timing(
                    {"timing/trainer/data_wait_ms": self._timing_end_ms(data_wait_start)}
                )
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)
                schedule_model = self.accelerator.unwrap_model(self.model)
                if hasattr(schedule_model, "set_training_step"):
                    schedule_model.set_training_step(self.global_step)

                forward_start = self._timing_start()
                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self._accumulate_timing(
                    {"timing/trainer/forward_ms": self._timing_end_ms(forward_start)}
                )
                timing_loss_metrics, loss_dict = self._split_timing_metrics(loss_dict)
                self._accumulate_timing(timing_loss_metrics)

                backward_start = self._timing_start()
                self.accelerator.backward(loss)
                self._accumulate_timing(
                    {"timing/trainer/backward_ms": self._timing_end_ms(backward_start)}
                )

                if self.accelerator.sync_gradients:
                    optimizer_start = self._timing_start()
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_ms = self._timing_end_ms(optimizer_start)
                    self._accumulate_timing({"timing/trainer/optimizer_ms": optimizer_ms})
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                    timing_metrics = self._consume_timing_metrics()
                    if timing_metrics:
                        trainer_total_ms = sum(
                            value
                            for key, value in timing_metrics.items()
                            if key.startswith("timing/trainer/") and key != "timing/trainer/step_total_ms"
                        )
                        timing_metrics["timing/trainer/step_total_ms"] = trainer_total_ms
                    global_timing_metrics = {}
                    for key, value in timing_metrics.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_timing_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if (
                        self.benchmark_enabled
                        and self.global_step == self.benchmark_total_steps
                    ):
                        self._finalize_benchmark()
                        return

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)
                        if global_timing_metrics:
                            logger.info(
                                "[timing] step=%d %s",
                                self.global_step,
                                self._format_timing_log_line(global_timing_metrics),
                            )

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        for key, value in global_timing_metrics.items():
                            wandb_payload[key] = value
                        self._wandb_log(wandb_payload)

                    self._maybe_start_benchmark_window()

                    train_vis_path = self._maybe_save_train_visualization(sample)
                    if train_vis_path is not None and self.accelerator.is_main_process:
                        logger.info(
                            "[train_vis] step=%d saved=%s",
                            self.global_step,
                            train_vis_path,
                        )

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            if metrics.get("action_fit_path") is not None:
                                logger.info(
                                    "[action_fit] step=%d saved=%s",
                                    self.global_step,
                                    metrics["action_fit_path"],
                                )
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
