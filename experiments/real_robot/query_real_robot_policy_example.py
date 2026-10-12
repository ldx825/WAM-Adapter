import argparse
import base64
import io
import json
import socket
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
from PIL import Image, ImageDraw


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_json_dumps(data: Any) -> bytes:
    return json.dumps(data, cls=NumpyJSONEncoder).encode("utf-8")


def numpy_json_loads(payload: bytes) -> Any:
    def object_hook(obj: dict[str, Any]) -> Any:
        if obj.get("__numpy_array__") is True:
            raw = base64.b64decode(obj["data"])
            return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
        return obj

    return json.loads(payload.decode("utf-8"), object_hook=object_hook)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < size:
        chunk = sock.recv(min(size - received, 4096))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _send_request(host: str, port: int, cmd: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    request = {"cmd": cmd, "payload": payload}
    encoded = numpy_json_dumps(request)
    with socket.create_connection((host, port), timeout=30.0) as sock:
        sock.sendall(len(encoded).to_bytes(4, "big"))
        sock.sendall(encoded)
        reply_len = int.from_bytes(_recv_exact(sock, 4), "big")
        return numpy_json_loads(_recv_exact(sock, reply_len))


def _decode_hdf5_image(encoded: np.ndarray) -> np.ndarray:
    if encoded.dtype != np.uint8:
        encoded = encoded.astype(np.uint8, copy=False)
    image = Image.open(io.BytesIO(encoded.tobytes())).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _build_robotwin_preview(images: dict[str, np.ndarray]) -> np.ndarray:
    top = Image.fromarray(images["cam_high"]).resize((320, 256), resample=Image.BILINEAR)
    left = Image.fromarray(images["cam_left_wrist"]).resize((160, 128), resample=Image.BILINEAR)
    right = Image.fromarray(images["cam_right_wrist"]).resize((160, 128), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (320, 384), color=(255, 255, 255))
    canvas.paste(top, (0, 0))
    canvas.paste(left, (0, 256))
    canvas.paste(right, (160, 256))
    return np.asarray(canvas, dtype=np.uint8)


def _swap_rb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {tuple(image.shape)}")
    return np.ascontiguousarray(image[..., [2, 1, 0]])


def _apply_color_fix(images: dict[str, np.ndarray], color_fix: str) -> dict[str, np.ndarray]:
    mode = str(color_fix).strip().lower()
    if mode == "none":
        return {key: np.ascontiguousarray(value.copy()) for key, value in images.items()}
    if mode == "swap_rb":
        return {key: _swap_rb(value) for key, value in images.items()}
    raise ValueError(f"Unsupported color_fix={color_fix}. Expected one of ['none', 'swap_rb'].")


def _load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(str(path)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _load_request_bundle(request_dir: Path) -> dict[str, Any]:
    meta_path = request_dir / "server_request" / "request_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Saved request meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    request_root = meta_path.parent
    raw_images = {
        "cam_high": _load_rgb_image(request_root / "cam_high_raw.png"),
        "cam_left_wrist": _load_rgb_image(request_root / "cam_left_wrist_raw.png"),
        "cam_right_wrist": _load_rgb_image(request_root / "cam_right_wrist_raw.png"),
    }
    gt_action_chunk_path = request_root / "gt_action_chunk.npy"
    gt_action_chunk = None
    if gt_action_chunk_path.exists():
        gt_action_chunk = np.load(gt_action_chunk_path)
    return {
        "meta": meta,
        "raw_images": raw_images,
        "gt_action_chunk": gt_action_chunk,
    }


def _draw_action_chunk_fit_plot(
    *,
    pred_action: np.ndarray,
    gt_action: np.ndarray,
    title: str,
) -> Image.Image:
    if pred_action.shape != gt_action.shape:
        raise ValueError(f"Pred/GT action shape mismatch: pred={pred_action.shape} gt={gt_action.shape}")
    if pred_action.ndim != 2:
        raise ValueError(f"Expected [T, D] action chunks, got {pred_action.shape}")

    horizon, action_dim = pred_action.shape
    num_cols = 4
    num_rows = int(np.ceil(action_dim / num_cols))
    margin = 24
    gap = 18
    tile_w = 240
    tile_h = 150
    title_h = 48
    canvas_w = margin * 2 + num_cols * tile_w + (num_cols - 1) * gap
    canvas_h = margin * 2 + title_h + num_rows * tile_h + (num_rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill=(0, 0, 0))
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
            points = []
            for t_idx, value in enumerate(series.tolist()):
                x = left if horizon <= 1 else left + (right - left) * (t_idx / (horizon - 1))
                y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
                points.append((float(x), float(y)))
            return points

        gt_points = _to_points(gt_series)
        pred_points = _to_points(pred_series)
        if len(gt_points) >= 2:
            draw.line(gt_points, fill=(50, 110, 240), width=2)
        if len(pred_points) >= 2:
            draw.line(pred_points, fill=(230, 60, 60), width=2)

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the real-robot WAM-Adapter service with one HDF5 sample.")
    parser.add_argument("--episode-hdf5", default=None, help="Path to one episode_*.hdf5 file.")
    parser.add_argument(
        "--request-dir",
        default=None,
        help="Optional saved request directory produced by this script. When provided, the script loads the "
        "saved three-view images, task, state, and GT action chunk from disk instead of reading HDF5.",
    )
    parser.add_argument("--sample-step", type=int, default=0, help="Timestep index to query.")
    parser.add_argument(
        "--current-obs-image",
        default=None,
        help="Optional precomposed current observation image (e.g. current_obs_step0_corrected.png). "
        "When provided, the script sends this single canvas image instead of three raw camera images.",
    )
    parser.add_argument(
        "--input-color-fix",
        default="swap_rb",
        choices=["none", "swap_rb"],
        help="Optional external color fix applied to the three raw camera images before sending them to the service.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Policy service host.")
    parser.add_argument("--port", type=int, default=5566, help="Policy service port.")
    parser.add_argument("--task-description", default=None, help="Optional task override.")
    parser.add_argument("--output-dir", default="./real_robot_query_example", help="Where to save the request/response.")
    parser.add_argument("--action-horizon", type=int, default=None, help="Optional override.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Optional override.")
    parser.add_argument(
        "--apply-gripper-postprocess",
        action="store_true",
        help="Apply the service's legacy gripper postprocess on the returned action chunk. "
        "Disabled by default for evaluation so raw denormalized actions are compared against GT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episode_hdf5 is None and args.request_dir is None:
        raise ValueError("Either --episode-hdf5 or --request-dir must be provided.")
    episode_path = None
    if args.episode_hdf5 is not None:
        episode_path = Path(args.episode_hdf5).expanduser().resolve()
        if not episode_path.exists():
            raise FileNotFoundError(f"Episode HDF5 not found: {episode_path}")
    request_dir = None
    if args.request_dir is not None:
        request_dir = Path(args.request_dir).expanduser().resolve()
        if not request_dir.exists():
            raise FileNotFoundError(f"Request dir not found: {request_dir}")
    current_obs_image_path: Optional[Path] = None
    if args.current_obs_image is not None:
        current_obs_image_path = Path(args.current_obs_image).expanduser().resolve()
        if not current_obs_image_path.exists():
            raise FileNotFoundError(f"Current observation image not found: {current_obs_image_path}")

    task_description = args.task_description
    step_idx = int(args.sample_step)
    action = None
    state = None
    images = None
    raw_images = None
    raw_preview = None
    corrected_preview = None
    current_obs_image = None
    gt_action_chunk_saved = None

    if request_dir is not None:
        bundle = _load_request_bundle(request_dir)
        meta = bundle["meta"]
        if task_description is None:
            task_description = str(meta["task_description"])
        state = np.asarray(meta["state"], dtype=np.float32)
        step_idx = int(meta.get("sample_step", step_idx))
        raw_images = bundle["raw_images"]
        raw_preview = _build_robotwin_preview(raw_images)
        images = _apply_color_fix(raw_images, args.input_color_fix)
        corrected_preview = _build_robotwin_preview(images)
        gt_action_chunk_saved = bundle["gt_action_chunk"]
    else:
        with h5py.File(str(episode_path), "r") as f:
            if task_description is None:
                task_attr = f.attrs.get("task", None)
                if task_attr is None:
                    raise KeyError("Missing `task` attribute in episode HDF5 and no --task-description was given.")
                task_description = task_attr.decode("utf-8") if isinstance(task_attr, bytes) else str(task_attr)

            total_steps = int(f["action"].shape[0])
            if step_idx < 0 or step_idx >= total_steps:
                raise IndexError(f"sample-step {step_idx} out of bounds [0, {total_steps})")

            action = np.asarray(f["action"], dtype=np.float32)
            state = np.asarray(f["observation/state"][step_idx], dtype=np.float32)
            if current_obs_image_path is None:
                raw_images = {
                    "cam_high": _decode_hdf5_image(f["observation/images/cam_high"][step_idx]),
                    "cam_left_wrist": _decode_hdf5_image(f["observation/images/cam_left_wrist"][step_idx]),
                    "cam_right_wrist": _decode_hdf5_image(f["observation/images/cam_right_wrist"][step_idx]),
                }
                raw_preview = _build_robotwin_preview(raw_images)
                images = _apply_color_fix(raw_images, args.input_color_fix)
                corrected_preview = _build_robotwin_preview(images)
            else:
                current_obs_image = _load_rgb_image(current_obs_image_path)

    payload = {"task_description": task_description, "observation": {"state": state}}
    infer_cmd = "infer_action_chunk"
    if images is not None:
        payload["observation"]["images"] = images
    else:
        infer_cmd = "infer_action_chunk_from_canvas"
        payload["current_obs_image"] = current_obs_image
    if args.action_horizon is not None:
        payload["action_horizon"] = int(args.action_horizon)
    if args.num_inference_steps is not None:
        payload["num_inference_steps"] = int(args.num_inference_steps)
    payload["apply_gripper_postprocess"] = bool(args.apply_gripper_postprocess)

    health_reply = _send_request(args.host, args.port, "health", None)
    infer_reply = _send_request(args.host, args.port, infer_cmd, payload)
    if not bool(health_reply.get("ok", False)):
        raise RuntimeError(f"Health request failed: {health_reply}")
    if not bool(infer_reply.get("ok", False)):
        raise RuntimeError(f"Inference request failed: {infer_reply}")

    result = infer_reply["result"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if images is not None:
        Image.fromarray(raw_preview).save(output_dir / "current_obs_step0_raw.png")
        Image.fromarray(corrected_preview).save(output_dir / "current_obs_step0_corrected.png")
        Image.fromarray(corrected_preview).save(output_dir / "request_current_obs.png")
    else:
        Image.fromarray(current_obs_image).save(output_dir / "request_current_obs.png")

    pred_action_chunk = np.asarray(result["action_chunk"], dtype=np.float32)
    np.save(output_dir / "response_action_chunk.npy", pred_action_chunk)
    np.save(output_dir / "pred_action_chunk.npy", pred_action_chunk)
    (output_dir / "request_meta.json").write_text(
        json.dumps(
            {
                "episode_hdf5": None if episode_path is None else str(episode_path),
                "request_dir": None if request_dir is None else str(request_dir),
                "sample_step": int(step_idx),
                "task_description": task_description,
                "state": state.tolist(),
                "mode": "canvas" if current_obs_image_path is not None else "three_camera",
                "current_obs_image": None if current_obs_image_path is None else str(current_obs_image_path),
                "input_color_fix": str(args.input_color_fix),
                "apply_gripper_postprocess": bool(args.apply_gripper_postprocess),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if raw_images is not None:
        request_root = output_dir / "server_request"
        request_root.mkdir(parents=True, exist_ok=True)
        for key, image in raw_images.items():
            Image.fromarray(image).save(request_root / f"{key}_raw.png")
        for key, image in images.items():
            Image.fromarray(image).save(request_root / f"{key}_corrected.png")
        (request_root / "task.txt").write_text(str(task_description), encoding="utf-8")
        (request_root / "state.json").write_text(json.dumps(state.tolist(), indent=2), encoding="utf-8")
        np.save(request_root / "state.npy", state.astype(np.float32))
        gt_request_meta = {
            "task_description": str(task_description),
            "state": state.tolist(),
            "sample_step": int(step_idx),
            "input_color_fix": str(args.input_color_fix),
            "action_horizon": int(result["action_horizon"]),
            "num_inference_steps": int(args.num_inference_steps) if args.num_inference_steps is not None else None,
            "apply_gripper_postprocess": bool(args.apply_gripper_postprocess),
            "image_files": {
                "cam_high_raw": "cam_high_raw.png",
                "cam_left_wrist_raw": "cam_left_wrist_raw.png",
                "cam_right_wrist_raw": "cam_right_wrist_raw.png",
                "cam_high_corrected": "cam_high_corrected.png",
                "cam_left_wrist_corrected": "cam_left_wrist_corrected.png",
                "cam_right_wrist_corrected": "cam_right_wrist_corrected.png",
            },
        }
        (request_root / "request_meta.json").write_text(json.dumps(gt_request_meta, indent=2), encoding="utf-8")

    response_summary: dict[str, Any] = {
        "health": health_reply["result"],
        "latency_s": float(result["latency_s"]),
        "action_horizon": int(result["action_horizon"]),
        "apply_gripper_postprocess": bool(result.get("apply_gripper_postprocess", args.apply_gripper_postprocess)),
        "first_action": pred_action_chunk[0].tolist(),
    }

    metric_horizon = int(result["action_horizon"])
    if action is not None:
        gt_stop = min(int(action.shape[0]), step_idx + metric_horizon)
        gt_action_chunk = np.asarray(action[step_idx:gt_stop], dtype=np.float32)
    elif gt_action_chunk_saved is not None:
        gt_action_chunk = np.asarray(gt_action_chunk_saved, dtype=np.float32)
    else:
        raise ValueError("GT action chunk is unavailable. Provide --episode-hdf5 or a saved request bundle with GT.")
    np.save(output_dir / "gt_action_chunk.npy", gt_action_chunk)
    if int(pred_action_chunk.shape[0]) < metric_horizon:
        raise ValueError(
            f"Predicted action chunk is shorter than ACTION_HORIZON: pred={pred_action_chunk.shape[0]} horizon={metric_horizon}"
        )
    if int(gt_action_chunk.shape[0]) < metric_horizon:
        raise ValueError(
            f"GT action chunk is shorter than ACTION_HORIZON: gt={gt_action_chunk.shape[0]} horizon={metric_horizon}"
        )
    pred_compare = pred_action_chunk[:metric_horizon]
    gt_compare = gt_action_chunk[:metric_horizon]
    diff = pred_compare - gt_compare
    action_l1 = float(np.abs(diff).mean())
    action_l2 = float((diff ** 2).mean())
    per_dim_l1 = np.abs(diff).mean(axis=0).tolist()
    per_dim_l2 = (diff ** 2).mean(axis=0).tolist()

    fit_plot = _draw_action_chunk_fit_plot(
        pred_action=pred_compare,
        gt_action=gt_compare,
        title=f"Action Chunk Fit | step={step_idx}",
    )
    fit_plot.save(output_dir / "action_chunk_fit.png")
    (output_dir / "pred_vs_gt.json").write_text(
        json.dumps(
            {
                "compare_horizon": metric_horizon,
                "pred_action_chunk": pred_compare.tolist(),
                "gt_action_chunk": gt_compare.tolist(),
                "diff_action_chunk": diff.tolist(),
                "action_l1": action_l1,
                "action_l2": action_l2,
                "per_dim_l1": per_dim_l1,
                "per_dim_l2": per_dim_l2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    response_summary.update(
        {
            "compare_horizon": metric_horizon,
            "action_l1": action_l1,
            "action_l2": action_l2,
            "per_dim_l1": per_dim_l1,
            "per_dim_l2": per_dim_l2,
            "gt_first_action": gt_compare[0].tolist(),
        }
    )
    (output_dir / "response.json").write_text(json.dumps(response_summary, indent=2), encoding="utf-8")

    print(f"[real-robot-query] saved={output_dir}")
    print(f"[real-robot-query] latency_s={float(result['latency_s']):.4f}")
    print(f"[real-robot-query] action_chunk_shape={pred_action_chunk.shape}")
    print(f"[real-robot-query] compare_horizon={response_summary['compare_horizon']}")
    print(f"[real-robot-query] action_l1={response_summary['action_l1']:.6f}")
    print(f"[real-robot-query] action_l2={response_summary['action_l2']:.6f}")


if __name__ == "__main__":
    main()
