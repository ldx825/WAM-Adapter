#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

CKPT="${CKPT:-}"
if [[ -z "${CKPT}" ]]; then
  echo "Error: CKPT is required."
  echo "Example:"
  echo "  CKPT=/path/to/checkpoints/weights/step_080000.pt bash experiments/real_robot/visualize_future_video_real_robot_rollout.sh"
  exit 1
fi

ROLLOUT_DIRS="${ROLLOUT_DIRS:-./result/result_1,./result/result_2}"
OUTPUT_DIR="${OUTPUT_DIR:-./qualitative_results/real_robot_rollout/$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
SIGMA_SHIFT="${SIGMA_SHIFT:-}"
PANEL_OFFSETS="${PANEL_OFFSETS:-8,16,24,32}"
GT_OFFSET="${GT_OFFSET:-24}"
SAVE_LATENTS="${SAVE_LATENTS:-false}"
ALLOW_MISSING_GT="${ALLOW_MISSING_GT:-false}"
TILED="${TILED:-false}"
SEED="${SEED:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-}"

EXTRA_ARGS=("$@")

IFS=',' read -r -a ROLLOUT_DIR_ARRAY <<< "${ROLLOUT_DIRS}"
if [[ "${#ROLLOUT_DIR_ARRAY[@]}" -eq 0 ]]; then
  echo "Error: ROLLOUT_DIRS is empty."
  exit 1
fi

echo "[qualitative-real-robot] ckpt=${CKPT}"
echo "[qualitative-real-robot] rollout_dirs=${ROLLOUT_DIRS}"
echo "[qualitative-real-robot] output_dir=${OUTPUT_DIR}"
echo "[qualitative-real-robot] gpu_id=${GPU_ID}"
echo "[qualitative-real-robot] device=${DEVICE}"
echo "[qualitative-real-robot] panel_offsets=${PANEL_OFFSETS}"
echo "[qualitative-real-robot] gt_offset=${GT_OFFSET}"

cmd=(
  python "${SCRIPT_DIR}/qualitative_future_video_real_robot_rollout.py"
  --ckpt "${CKPT}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --panel-offsets "${PANEL_OFFSETS}"
  --gt-offset "${GT_OFFSET}"
  --rollout-dirs
)

for rollout_dir in "${ROLLOUT_DIR_ARRAY[@]}"; do
  trimmed="$(echo "${rollout_dir}" | sed 's/^ *//;s/ *$//')"
  if [[ -n "${trimmed}" ]]; then
    cmd+=("${trimmed}")
  fi
done

if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  cmd+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi

if [[ -n "${SIGMA_SHIFT}" ]]; then
  cmd+=(--sigma-shift "${SIGMA_SHIFT}")
fi

if [[ -n "${SEED}" ]]; then
  cmd+=(--seed "${SEED}")
fi

if [[ -n "${CONFIG_PATH}" ]]; then
  cmd+=(--config-path "${CONFIG_PATH}")
fi

if [[ -n "${DATASET_STATS_PATH}" ]]; then
  cmd+=(--dataset-stats-path "${DATASET_STATS_PATH}")
fi

if [[ "${SAVE_LATENTS}" == "true" ]]; then
  cmd+=(--save-latents)
fi

if [[ "${ALLOW_MISSING_GT}" == "true" ]]; then
  cmd+=(--allow-missing-gt)
fi

if [[ "${TILED}" == "true" ]]; then
  cmd+=(--tiled)
fi

cmd+=("${EXTRA_ARGS[@]}")

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
