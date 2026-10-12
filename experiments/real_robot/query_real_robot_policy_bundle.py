import argparse
import base64
import json
import socket
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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


def _load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.uint8)


def _build_robotwin_preview(images: dict[str, np.ndarray]) -> np.ndarray:
    top = Image.fromarray(images["cam_high"]).resize((320, 256), resample=Image.BILINEAR)
    left = Image.fromarray(images["cam_left_wrist"]).resize((160, 128), resample=Image.BILINEAR)
    right = Image.fromarray(images["cam_right_wrist"]).resize((160, 128), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (320, 384), color=(255, 255, 255))
    canvas.paste(top, (0, 0))
    canvas.paste(left, (0, 256))
    canvas.paste(right, (160, 256))
    return np.asarray(canvas, dtype=np.uint8)


def _load_request_meta(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"`request_meta.json` must be a JSON object, got {type(data)}")
    return data


def _load_state(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "state" in data:
            data = data["state"]
        elif "observation" in data and isinstance(data["observation"], dict) and "state" in data["observation"]:
            data = data["observation"]["state"]
    state = np.asarray(data, dtype=np.float32).reshape(-1)
    if state.shape != (14,):
        raise ValueError(f"`state.json` must contain 14 values, got shape {tuple(state.shape)}")
    return state


def _load_task(task_txt_path: Path, request_meta: dict[str, Any]) -> str:
    text = task_txt_path.read_text(encoding="utf-8").strip()
    if text:
        return text
    task = request_meta.get("task_description")
    if isinstance(task, str) and task.strip():
        return task.strip()
    raise ValueError("Failed to resolve task description from `task.txt` or `request_meta.json`.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the real-robot policy server from a minimal corrected-image bundle."
    )
    parser.add_argument("--bundle-dir", required=True, help="Directory containing corrected images and metadata.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy service host.")
    parser.add_argument("--port", type=int, default=5566, help="Policy service port.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save the response. Defaults to <bundle-dir>/server_query_response.",
    )
    parser.add_argument("--action-horizon", type=int, default=None, help="Optional override.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Optional override.")
    parser.add_argument(
        "--apply-gripper-postprocess",
        action="store_true",
        help="Apply the server's legacy gripper postprocess before returning actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    required_paths = {
        "cam_high": bundle_dir / "cam_high_corrected.png",
        "cam_left_wrist": bundle_dir / "cam_left_wrist_corrected.png",
        "cam_right_wrist": bundle_dir / "cam_right_wrist_corrected.png",
        "request_meta": bundle_dir / "request_meta.json",
        "state": bundle_dir / "state.json",
        "task": bundle_dir / "task.txt",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required bundle files: {missing}")

    request_meta = _load_request_meta(required_paths["request_meta"])
    state = _load_state(required_paths["state"])
    task_description = _load_task(required_paths["task"], request_meta)
    images = {
        "cam_high": _load_rgb_image(required_paths["cam_high"]),
        "cam_left_wrist": _load_rgb_image(required_paths["cam_left_wrist"]),
        "cam_right_wrist": _load_rgb_image(required_paths["cam_right_wrist"]),
    }

    payload: dict[str, Any] = {
        "task_description": task_description,
        "observation": {
            "state": state,
            "images": images,
        },
        "apply_gripper_postprocess": bool(args.apply_gripper_postprocess),
    }

    meta_action_horizon = request_meta.get("action_horizon")
    meta_num_inference_steps = request_meta.get("num_inference_steps")
    if args.action_horizon is not None:
        payload["action_horizon"] = int(args.action_horizon)
    elif meta_action_horizon is not None:
        payload["action_horizon"] = int(meta_action_horizon)
    if args.num_inference_steps is not None:
        payload["num_inference_steps"] = int(args.num_inference_steps)
    elif meta_num_inference_steps is not None:
        payload["num_inference_steps"] = int(meta_num_inference_steps)

    health_reply = _send_request(args.host, args.port, "health", None)
    infer_reply = _send_request(args.host, args.port, "infer_action_chunk", payload)
    if not bool(health_reply.get("ok", False)):
        raise RuntimeError(f"Health request failed: {health_reply}")
    if not bool(infer_reply.get("ok", False)):
        raise RuntimeError(f"Inference request failed: {infer_reply}")

    result = infer_reply["result"]
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (bundle_dir / "server_query_response").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    preview = _build_robotwin_preview(images)
    Image.fromarray(preview).save(output_dir / "request_current_obs.png")

    action_chunk = np.asarray(result["action_chunk"], dtype=np.float32)
    np.save(output_dir / "response_action_chunk.npy", action_chunk)

    response_payload = {
        "bundle_dir": str(bundle_dir),
        "task_description": task_description,
        "state": state.tolist(),
        "health": health_reply["result"],
        "latency_s": float(result["latency_s"]),
        "action_horizon": int(result["action_horizon"]),
        "apply_gripper_postprocess": bool(result.get("apply_gripper_postprocess", args.apply_gripper_postprocess)),
        "action_chunk_shape": list(action_chunk.shape),
        "first_action": action_chunk[0].tolist(),
        "action_chunk": action_chunk.tolist(),
    }
    (output_dir / "response.json").write_text(json.dumps(response_payload, indent=2), encoding="utf-8")

    print(f"[real-robot-bundle-query] bundle_dir={bundle_dir}")
    print(f"[real-robot-bundle-query] saved={output_dir}")
    print(f"[real-robot-bundle-query] latency_s={float(result['latency_s']):.4f}")
    print(f"[real-robot-bundle-query] action_chunk_shape={tuple(action_chunk.shape)}")


if __name__ == "__main__":
    main()
