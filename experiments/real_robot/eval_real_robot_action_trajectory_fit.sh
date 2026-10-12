#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

CKPT="${CKPT:-}"
if [[ -z "${CKPT}" ]]; then
  echo "Error: CKPT is required."
  echo "Example:"
  echo "  CKPT=/path/to/checkpoints/weights/step_050000.pt DATASET_DIRS=/path/task_a,/path/task_b,/path/task_c bash experiments/real_robot/eval_real_robot_action_trajectory_fit.sh"
  exit 1
fi

EPISODE_HDF5S="${EPISODE_HDF5S:-}"
DATASET_DIRS="${DATASET_DIRS:-}"
if [[ -z "${EPISODE_HDF5S}" && -z "${DATASET_DIRS}" ]]; then
  echo "Error: either EPISODE_HDF5S or DATASET_DIRS must be provided."
  exit 1
fi

GPU_ID="${GPU_ID:-0}"
CONFIG_PATH="${CONFIG_PATH:-}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./real_robot_action_trajectory_eval/$(date +%Y%m%d_%H%M%S)}"
ACTION_HORIZON="${ACTION_HORIZON:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
MAX_STEPS="${MAX_STEPS:-}"
MIXED_PRECISION="${MIXED_PRECISION:-}"
INPUT_COLOR_FIX="${INPUT_COLOR_FIX:-swap_rb}"
APPLY_GRIPPER_POSTPROCESS="${APPLY_GRIPPER_POSTPROCESS:-false}"

EXTRA_ARGS=("$@")

echo "[real-robot-trajectory-fit] ckpt=${CKPT}"
echo "[real-robot-trajectory-fit] output_dir=${OUTPUT_DIR}"
echo "[real-robot-trajectory-fit] gpu_id=${GPU_ID}"
echo "[real-robot-trajectory-fit] episode_hdf5s=${EPISODE_HDF5S:-<none>}"
echo "[real-robot-trajectory-fit] dataset_dirs=${DATASET_DIRS:-<none>}"
echo "[real-robot-trajectory-fit] input_color_fix=${INPUT_COLOR_FIX}"
echo "[real-robot-trajectory-fit] apply_gripper_postprocess=${APPLY_GRIPPER_POSTPROCESS}"

cmd=(
  python "${SCRIPT_DIR}/eval_real_robot_action_trajectory_fit.py"
  "--ckpt" "${CKPT}"
  "--output-dir" "${OUTPUT_DIR}"
  "--device" "cuda"
  "--input-color-fix" "${INPUT_COLOR_FIX}"
)

if [[ -n "${CONFIG_PATH}" ]]; then
  cmd+=("--config-path" "${CONFIG_PATH}")
fi
if [[ -n "${DATASET_STATS_PATH}" ]]; then
  cmd+=("--dataset-stats-path" "${DATASET_STATS_PATH}")
fi
if [[ -n "${EPISODE_HDF5S}" ]]; then
  cmd+=("--episode-hdf5s" "${EPISODE_HDF5S}")
fi
if [[ -n "${DATASET_DIRS}" ]]; then
  cmd+=("--dataset-dirs" "${DATASET_DIRS}")
fi
if [[ -n "${ACTION_HORIZON}" ]]; then
  cmd+=("--action-horizon" "${ACTION_HORIZON}")
fi
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  cmd+=("--num-inference-steps" "${NUM_INFERENCE_STEPS}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  cmd+=("--max-steps" "${MAX_STEPS}")
fi
if [[ -n "${MIXED_PRECISION}" ]]; then
  cmd+=("--mixed-precision" "${MIXED_PRECISION}")
fi
if [[ "${APPLY_GRIPPER_POSTPROCESS}" == "true" ]]; then
  cmd+=("--apply-gripper-postprocess")
fi

cmd+=("${EXTRA_ARGS[@]}")

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
