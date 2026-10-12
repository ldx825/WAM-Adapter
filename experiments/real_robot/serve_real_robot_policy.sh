#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
if [[ -n "${WAM_ADAPTER_ROOT:-}" ]]; then
  export PYTHONPATH="${WAM_ADAPTER_ROOT}:${PYTHONPATH:-}"
fi

CKPT="${CKPT:-}"
if [[ -z "${CKPT}" ]]; then
  echo "Error: CKPT is required." >&2
  echo "Example:" >&2
  echo "  CKPT=/path/to/checkpoints/weights/step_050000.pt GPU_ID=0 bash serve_real_robot_policy.sh" >&2
  exit 1
fi

GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5566}"
PROTOCOL="${PROTOCOL:-openpi}"
CONFIG_PATH="${CONFIG_PATH:-}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-}"
MIXED_PRECISION="${MIXED_PRECISION:-}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
SIGMA_SHIFT="${SIGMA_SHIFT:-}"
TEXT_CFG_SCALE="${TEXT_CFG_SCALE:-1.0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
RAND_DEVICE="${RAND_DEVICE:-cpu}"
TILED="${TILED:-false}"
BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-true}"
ENABLE_REAL_GRIPPER_SNAP="${ENABLE_REAL_GRIPPER_SNAP:-true}"
REAL_GRIPPER_DIM_INDICES="${REAL_GRIPPER_DIM_INDICES:-6,13}"
REAL_GRIPPER_CLOSE_SNAP_THRESHOLD="${REAL_GRIPPER_CLOSE_SNAP_THRESHOLD:-5.0}"
USE_PROMPT_CACHE="${USE_PROMPT_CACHE:-true}"
QUERY_SAVE_DIR="${QUERY_SAVE_DIR:-${SCRIPT_DIR}/server_query_logs}"
SAVE_QUERIES="${SAVE_QUERIES:-true}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=("$@")

echo "[serve-real-robot] ckpt=${CKPT}"
echo "[serve-real-robot] host=${HOST} port=${PORT} gpu_id=${GPU_ID}"
echo "[serve-real-robot] protocol=${PROTOCOL}"
echo "[serve-real-robot] config_path=${CONFIG_PATH:-<auto>}"
echo "[serve-real-robot] dataset_stats_path=${DATASET_STATS_PATH:-<auto>}"
echo "[serve-real-robot] enable_real_gripper_snap=${ENABLE_REAL_GRIPPER_SNAP} real_gripper_dim_indices=${REAL_GRIPPER_DIM_INDICES} close_snap_threshold=${REAL_GRIPPER_CLOSE_SNAP_THRESHOLD}"
echo "[serve-real-robot] query_save_dir=${QUERY_SAVE_DIR}"
echo "[serve-real-robot] save_queries=${SAVE_QUERIES}"

cmd=(
  python "${SCRIPT_DIR}/serve_real_robot_policy.py"
  --ckpt "${CKPT}"
  --host "${HOST}"
  --port "${PORT}"
  --protocol "${PROTOCOL}"
  --device "cuda:0"
  --text-cfg-scale "${TEXT_CFG_SCALE}"
  --negative-prompt "${NEGATIVE_PROMPT}"
  --rand-device "${RAND_DEVICE}"
)

if [[ -n "${CONFIG_PATH}" ]]; then
  cmd+=(--config-path "${CONFIG_PATH}")
fi
if [[ -n "${DATASET_STATS_PATH}" ]]; then
  cmd+=(--dataset-stats-path "${DATASET_STATS_PATH}")
fi
if [[ -n "${MIXED_PRECISION}" ]]; then
  cmd+=(--mixed-precision "${MIXED_PRECISION}")
fi
if [[ -n "${ACTION_HORIZON}" ]]; then
  cmd+=(--action-horizon "${ACTION_HORIZON}")
fi
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  cmd+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi
if [[ -n "${SIGMA_SHIFT}" ]]; then
  cmd+=(--sigma-shift "${SIGMA_SHIFT}")
fi
if [[ "${TILED}" == "true" ]]; then
  cmd+=(--tiled)
fi
if [[ "${BINARIZE_GRIPPER}" != "true" ]]; then
  cmd+=(--no-binarize-gripper)
fi
if [[ "${ENABLE_REAL_GRIPPER_SNAP}" != "true" ]]; then
  cmd+=(--disable-real-gripper-snap)
fi
if [[ -n "${REAL_GRIPPER_DIM_INDICES}" ]]; then
  cmd+=(--real-gripper-dim-indices "${REAL_GRIPPER_DIM_INDICES}")
fi
if [[ -n "${REAL_GRIPPER_CLOSE_SNAP_THRESHOLD}" ]]; then
  cmd+=(--real-gripper-close-snap-threshold "${REAL_GRIPPER_CLOSE_SNAP_THRESHOLD}")
fi
if [[ "${USE_PROMPT_CACHE}" != "true" ]]; then
  cmd+=(--disable-prompt-cache)
fi
if [[ -n "${QUERY_SAVE_DIR}" ]]; then
  cmd+=(--query-save-dir "${QUERY_SAVE_DIR}")
fi
if [[ "${SAVE_QUERIES}" != "true" ]]; then
  cmd+=(--disable-query-save)
fi
cmd+=("${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run] command:'
  printf ' %q' CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"

# HOST=0.0.0.0 \
# PORT=5566 \
# PROTOCOL=openpi \
# bash serve_real_robot_policy.sh
