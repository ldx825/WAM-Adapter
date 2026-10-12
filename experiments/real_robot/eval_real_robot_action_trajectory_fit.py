import argparse
import json
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
from PIL import Image, ImageDraw

from experiments.real_robot.serve_real_robot_policy import (
    RealRobotPolicyService,
    _resolve_checkpoint,
    _resolve_config_path,
    _resolve_dataset_stats_path,
)


def _parse_csv_paths(value: Optional[str]) -> list[Path]:
    if value is None or str(value).strip() == "":
        return []
    parts = [part.strip() for part in str(value).split(",")]
    out: list[Path] = []
    for part in parts:
        if not part:
            continue
        out.append(Path(part).expanduser().resolve())
    return out


def _resolve_episode_paths(
    *,
    episode_hdf5s: Optional[str],
    dataset_dirs: Optional[str],
) -> list[Path]:
    episode_paths = _parse_csv_paths(episode_hdf5s)
    if episode_paths:
        return episode_paths

    roots = _parse_csv_paths(dataset_dirs)
    if not roots:
        raise ValueError("Either `--episode-hdf5s` or `--dataset-dirs` must be provided.")

    resolved: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        candidates = sorted(root.glob("episode_*.hdf5"))
        if not candidates:
            raise FileNotFoundError(f"No episode_*.hdf5 found under: {root}")
        resolved.append(candidates[0].resolve())
    return resolved


def _load_hdf5_image(encoded: np.ndarray) -> np.ndarray:
    from experiments.real_robot.query_real_robot_policy_example import _decode_hdf5_image

    return _decode_hdf5_image(encoded)


def _swap_rb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {tuple(image.shape)}")
    return np.ascontiguousarray(image[..., [2, 1, 0]])


def _apply_color_fix(
    images: dict[str, np.ndarray],
    *,
    color_fix: str,
) -> dict[str, np.ndarray]:
    mode = str(color_fix).strip().lower()
    if mode == "none":
        return {key: np.ascontiguousarray(value.copy()) for key, value in images.items()}
    if mode == "swap_rb":
        return {key: _swap_rb(value) for key, value in images.items()}
    raise ValueError(f"Unsupported color_fix={color_fix}. Expected one of ['none', 'swap_rb'].")


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path)


def _build_robotwin_preview(images: dict[str, np.ndarray]) -> np.ndarray:
    from experiments.real_robot.query_real_robot_policy_example import _build_robotwin_preview

    return _build_robotwin_preview(images)


def _draw_trajectory_plot(
    *,
    pred_action: np.ndarray,
    gt_action: np.ndarray,
    title: str,
) -> Image.Image:
    if pred_action.shape != gt_action.shape:
        raise ValueError(
            f"Pred/GT action trajectory shape mismatch: pred={pred_action.shape} gt={gt_action.shape}"
        )
    if pred_action.ndim != 2:
        raise ValueError(f"Expected [T, D] action trajectories, got {pred_action.shape}")

    horizon, action_dim = pred_action.shape
    num_cols = 4
    num_rows = int(np.ceil(action_dim / num_cols))
    margin = 24
    gap = 18
    tile_w = 300
    tile_h = 160
    title_h = 58
    canvas_w = margin * 2 + num_cols * tile_w + (num_cols - 1) * gap
    canvas_h = margin * 2 + title_h + num_rows * tile_h + (num_rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill=(0, 0, 0))
    draw.text((margin, margin + 20), f"Trajectory length: {horizon} | Blue: GT   Red: Pred", fill=(60, 60, 60))

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

        gt_series = gt_action[:, dim_idx]
        pred_series = pred_action[:, dim_idx]
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
    return canvas


def _episode_name(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _evaluate_episode(
    *,
    service: RealRobotPolicyService,
    episode_path: Path,
    output_dir: Path,
    action_horizon: Optional[int],
    num_inference_steps: Optional[int],
    max_steps: Optional[int],
    color_fix: str,
    apply_gripper_postprocess: bool,
) -> dict[str, Any]:
    with h5py.File(str(episode_path), "r") as f:
        task_attr = f.attrs.get("task", None)
        if task_attr is None:
            raise KeyError(f"Missing `task` attribute in {episode_path}")
        task_description = task_attr.decode("utf-8") if isinstance(task_attr, bytes) else str(task_attr)

        action = np.asarray(f["action"], dtype=np.float32)
        state = np.asarray(f["observation/state"], dtype=np.float32)
        image_group = f["observation/images"]
        image_len = min(
            int(image_group["cam_high"].shape[0]),
            int(image_group["cam_left_wrist"].shape[0]),
            int(image_group["cam_right_wrist"].shape[0]),
        )

        total_steps = min(int(action.shape[0]), int(state.shape[0]), int(image_len))
        if max_steps is not None:
            total_steps = min(total_steps, int(max_steps))

        pred_first_actions = []
        step_latencies = []
        preview_saved = False
        for step_idx in range(total_steps):
            raw_images = {
                "cam_high": _load_hdf5_image(image_group["cam_high"][step_idx]),
                "cam_left_wrist": _load_hdf5_image(image_group["cam_left_wrist"][step_idx]),
                "cam_right_wrist": _load_hdf5_image(image_group["cam_right_wrist"][step_idx]),
            }
            images = _apply_color_fix(raw_images, color_fix=color_fix)
            if not preview_saved:
                _save_rgb(output_dir / "current_obs_step0_raw.png", _build_robotwin_preview(raw_images))
                _save_rgb(output_dir / "current_obs_step0_corrected.png", _build_robotwin_preview(images))
                preview_saved = True

            payload = {
                "task_description": task_description,
                "observation": {
                    "state": state[step_idx],
                    "images": images,
                },
            }
            if action_horizon is not None:
                payload["action_horizon"] = int(action_horizon)
            if num_inference_steps is not None:
                payload["num_inference_steps"] = int(num_inference_steps)
            payload["apply_gripper_postprocess"] = bool(apply_gripper_postprocess)

            result = service.infer_action_chunk(payload)
            action_chunk = np.asarray(result["action_chunk"], dtype=np.float32)
            pred_first_actions.append(action_chunk[0].copy())
            step_latencies.append(float(result["latency_s"]))

    pred_action = np.stack(pred_first_actions, axis=0).astype(np.float32)
    gt_action = action[:total_steps].astype(np.float32)
    if pred_action.shape != gt_action.shape:
        raise ValueError(
            f"Episode prediction/GT shape mismatch: pred={pred_action.shape} gt={gt_action.shape}"
        )

    diff = pred_action - gt_action
    action_l1 = float(np.abs(diff).mean())
    action_l2 = float((diff ** 2).mean())
    per_dim_l1 = np.abs(diff).mean(axis=0).tolist()
    per_dim_l2 = (diff ** 2).mean(axis=0).tolist()

    np.save(output_dir / "pred_action_trajectory.npy", pred_action)
    np.save(output_dir / "gt_action_trajectory.npy", gt_action)
    plot = _draw_trajectory_plot(
        pred_action=pred_action,
        gt_action=gt_action,
        title=f"Real-Robot Trajectory Fit | {_episode_name(episode_path)}",
    )
    plot.save(output_dir / "trajectory_fit.png")

    summary = {
        "episode_hdf5": str(episode_path),
        "task_description": task_description,
        "num_steps": int(total_steps),
        "action_horizon": int(action_horizon if action_horizon is not None else service.action_horizon),
        "num_inference_steps": int(
            num_inference_steps if num_inference_steps is not None else service.num_inference_steps
        ),
        "color_fix": str(color_fix),
        "apply_gripper_postprocess": bool(apply_gripper_postprocess),
        "action_l1": action_l1,
        "action_l2": action_l2,
        "per_dim_l1": per_dim_l1,
        "per_dim_l2": per_dim_l2,
        "mean_latency_s": float(np.mean(step_latencies)) if step_latencies else None,
        "median_latency_s": float(np.median(step_latencies)) if step_latencies else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline real-robot trajectory-fit evaluation on full HDF5 demonstrations."
    )
    parser.add_argument("--ckpt", required=True, help="Path to model weights checkpoint.")
    parser.add_argument("--config-path", default=None, help="Optional resolved config.yaml.")
    parser.add_argument("--dataset-stats-path", default=None, help="Optional dataset_stats.json path.")
    parser.add_argument(
        "--episode-hdf5s",
        default=None,
        help="Comma-separated list of exact episode_*.hdf5 files to evaluate.",
    )
    parser.add_argument(
        "--dataset-dirs",
        default=None,
        help="Comma-separated list of task directories. Uses the first episode_*.hdf5 from each.",
    )
    parser.add_argument("--device", default=None, help="Device, e.g. cuda or cpu.")
    parser.add_argument("--mixed-precision", default=None, help="Optional mixed precision override.")
    parser.add_argument("--action-horizon", type=int, default=None, help="Optional action horizon override.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Optional inference steps override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on evaluated timesteps per demo.")
    parser.add_argument(
        "--input-color-fix",
        default="none",
        choices=["none", "swap_rb"],
        help="Optional external color fix applied before feeding images into the model.",
    )
    parser.add_argument(
        "--output-dir",
        default="./real_robot_action_trajectory_eval",
        help="Directory to save plots and arrays.",
    )
    parser.add_argument(
        "--apply-gripper-postprocess",
        action="store_true",
        help="Apply the service's legacy gripper postprocess before computing trajectory-fit metrics. "
        "Disabled by default so raw denormalized actions are compared against GT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = _resolve_checkpoint(args.ckpt)
    config_path = _resolve_config_path(ckpt_path, args.config_path)
    dataset_stats_path = _resolve_dataset_stats_path(ckpt_path, args.dataset_stats_path)
    episode_paths = _resolve_episode_paths(
        episode_hdf5s=args.episode_hdf5s,
        dataset_dirs=args.dataset_dirs,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    service = RealRobotPolicyService(
        ckpt_path=ckpt_path,
        config_path=config_path,
        dataset_stats_path=dataset_stats_path,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=None,
        text_cfg_scale=1.0,
        negative_prompt="",
        rand_device="cpu",
        tiled=False,
        binarize_gripper=False,
        use_prompt_cache=True,
    )

    all_summaries = []
    for episode_idx, episode_path in enumerate(episode_paths):
        episode_output_dir = output_dir / f"demo_{episode_idx:02d}_{episode_path.parent.name}"
        episode_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[trajectory-fit] evaluating {episode_path}")
        summary = _evaluate_episode(
            service=service,
            episode_path=episode_path,
            output_dir=episode_output_dir,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            max_steps=args.max_steps,
            color_fix=args.input_color_fix,
            apply_gripper_postprocess=bool(args.apply_gripper_postprocess),
        )
        all_summaries.append(summary)
        print(
            f"[trajectory-fit] saved={episode_output_dir} "
            f"steps={summary['num_steps']} action_l1={summary['action_l1']:.6f} "
            f"action_l2={summary['action_l2']:.6f}"
        )

    aggregate = {
        "ckpt": str(ckpt_path),
        "config_path": str(config_path),
        "dataset_stats_path": str(dataset_stats_path),
        "episodes": all_summaries,
        "mean_action_l1": float(np.mean([item["action_l1"] for item in all_summaries])),
        "mean_action_l2": float(np.mean([item["action_l2"] for item in all_summaries])),
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"[trajectory-fit] summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
