#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

EPISODE_HDF5="${EPISODE_HDF5:-}"
REQUEST_DIR="${REQUEST_DIR:-}"
SAMPLE_STEP="${SAMPLE_STEP:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5566}"
OUTPUT_DIR="${OUTPUT_DIR:-./real_robot_query_example}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-}"
ACTION_HORIZON="${ACTION_HORIZON:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
CURRENT_OBS_IMAGE="${CURRENT_OBS_IMAGE:-}"
INPUT_COLOR_FIX="${INPUT_COLOR_FIX:-swap_rb}"
APPLY_GRIPPER_POSTPROCESS="${APPLY_GRIPPER_POSTPROCESS:-false}"
EXTRA_ARGS=("$@")

echo "[query-real-robot] episode_hdf5=${EPISODE_HDF5}"
echo "[query-real-robot] request_dir=${REQUEST_DIR:-<none>}"
echo "[query-real-robot] sample_step=${SAMPLE_STEP}"
echo "[query-real-robot] host=${HOST} port=${PORT}"
echo "[query-real-robot] output_dir=${OUTPUT_DIR}"
echo "[query-real-robot] current_obs_image=${CURRENT_OBS_IMAGE:-<none>}"
echo "[query-real-robot] input_color_fix=${INPUT_COLOR_FIX}"
echo "[query-real-robot] apply_gripper_postprocess=${APPLY_GRIPPER_POSTPROCESS}"

cmd=(
  python "${SCRIPT_DIR}/query_real_robot_policy_example.py"
  --sample-step "${SAMPLE_STEP}"
  --host "${HOST}"
  --port "${PORT}"
  --output-dir "${OUTPUT_DIR}"
  --input-color-fix "${INPUT_COLOR_FIX}"
)

if [[ -n "${REQUEST_DIR}" ]]; then
  cmd+=(--request-dir "${REQUEST_DIR}")
else
  cmd+=(--episode-hdf5 "${EPISODE_HDF5}")
fi

if [[ -n "${TASK_DESCRIPTION}" ]]; then
  cmd+=(--task-description "${TASK_DESCRIPTION}")
fi
if [[ -n "${ACTION_HORIZON}" ]]; then
  cmd+=(--action-horizon "${ACTION_HORIZON}")
fi
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  cmd+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi
if [[ -n "${CURRENT_OBS_IMAGE}" ]]; then
  cmd+=(--current-obs-image "${CURRENT_OBS_IMAGE}")
fi
if [[ "${APPLY_GRIPPER_POSTPROCESS}" == "true" ]]; then
  cmd+=(--apply-gripper-postprocess)
fi
cmd+=("${EXTRA_ARGS[@]}")

"${cmd[@]}"
