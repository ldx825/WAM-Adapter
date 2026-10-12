#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/data/models}"
export DIFFSYNTH_SKIP_DOWNLOAD=true

CONFIG_TASK="${CONFIG_TASK:-robotwin_uncond_3cam_384_1e-4}"
EVAL_MODE="${EVAL_MODE:-manager}"  # manager | single
CKPT="${CKPT:-}"
if [[ -z "${CKPT}" ]]; then
  echo "CKPT must point to a weights checkpoint, e.g. runs/.../checkpoints/weights/step_150000.pt" >&2
  exit 1
fi

CKPT_ABS="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")"
RUN_DIR="$(cd "$(dirname "${CKPT_ABS}")/../.." && pwd)"
RUN_TAG="$(basename "${RUN_DIR}")"
CKPT_FILE="$(basename "${CKPT_ABS}")"
CKPT_TAG="${CKPT_FILE%.*}"
OUTPUT_DIR_BASE="${OUTPUT_DIR:-${RUN_DIR}/eval_robotwin_${RUN_TAG}}"
if [[ "${OUTPUT_DIR_BASE}" == *"${CKPT_TAG}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR_BASE}"
else
  OUTPUT_DIR="${OUTPUT_DIR_BASE}_${CKPT_TAG}"
fi

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/third_party/RoboTwin}"
USE_TRAINING_RUN_CONFIG="${USE_TRAINING_RUN_CONFIG:-true}"
TRAINING_CONFIG_PATH="${TRAINING_CONFIG_PATH:-}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-}"
GPU_ID="${GPU_ID:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"

ROBOTWIN_TASK_NAME="${ROBOTWIN_TASK_NAME:-}"
TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-100}"
EVAL_TOTAL_EPISODES="${EVAL_TOTAL_EPISODES:-${EVAL_NUM_EPISODES}}"
EVAL_SEED_LIST_PATH="${EVAL_SEED_LIST_PATH:-null}"
EVAL_EPISODE_OFFSET="${EVAL_EPISODE_OFFSET:-0}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
ACTION_HORIZON="${ACTION_HORIZON:-null}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-null}"
SIGMA_SHIFT="${SIGMA_SHIFT:-null}"
TEXT_CFG_SCALE="${TEXT_CFG_SCALE:-1.0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
RAND_DEVICE="${RAND_DEVICE:-cpu}"
TILED="${TILED:-false}"
TIMING_ENABLED="${TIMING_ENABLED:-false}"
SKIP_GET_OBS_WITHIN_REPLAN="${SKIP_GET_OBS_WITHIN_REPLAN:-true}"
EVAL_VIDEO_FPS="${EVAL_VIDEO_FPS:-10}"
EVAL_VIDEO_RECORD_MODE="${EVAL_VIDEO_RECORD_MODE:-action_step}"
EVAL_VIDEO_FRAME_STRIDE="${EVAL_VIDEO_FRAME_STRIDE:-1}"
DEVICE="${DEVICE:-cuda}"

EXTRA_ARGS=("$@")

COMMON_ARGS=(
  "task=${CONFIG_TASK}"
  "ckpt=${CKPT_ABS}"
  "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}"
  "EVALUATION.output_dir=${OUTPUT_DIR}"
  "EVALUATION.use_training_run_config=${USE_TRAINING_RUN_CONFIG}"
  "EVALUATION.task_config=${TASK_CONFIG}"
  "EVALUATION.eval_num_episodes=${EVAL_NUM_EPISODES}"
  "EVALUATION.eval_total_episodes=${EVAL_TOTAL_EPISODES}"
  "EVALUATION.eval_seed_list_path=${EVAL_SEED_LIST_PATH}"
  "EVALUATION.eval_episode_offset=${EVAL_EPISODE_OFFSET}"
  "EVALUATION.instruction_type=${INSTRUCTION_TYPE}"
  "EVALUATION.action_horizon=${ACTION_HORIZON}"
  "EVALUATION.replan_steps=${REPLAN_STEPS}"
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "EVALUATION.sigma_shift=${SIGMA_SHIFT}"
  "EVALUATION.text_cfg_scale=${TEXT_CFG_SCALE}"
  "EVALUATION.negative_prompt=${NEGATIVE_PROMPT}"
  "EVALUATION.rand_device=${RAND_DEVICE}"
  "EVALUATION.tiled=${TILED}"
  "EVALUATION.timing_enabled=${TIMING_ENABLED}"
  "EVALUATION.skip_get_obs_within_replan=${SKIP_GET_OBS_WITHIN_REPLAN}"
  "EVALUATION.eval_video_fps=${EVAL_VIDEO_FPS}"
  "EVALUATION.eval_video_record_mode=${EVAL_VIDEO_RECORD_MODE}"
  "EVALUATION.eval_video_frame_stride=${EVAL_VIDEO_FRAME_STRIDE}"
  "EVALUATION.device=${DEVICE}"
)

if [[ -n "${TRAINING_CONFIG_PATH}" ]]; then
  COMMON_ARGS+=("EVALUATION.training_config_path=${TRAINING_CONFIG_PATH}")
fi
if [[ -n "${DATASET_STATS_PATH}" ]]; then
  COMMON_ARGS+=("EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}")
fi

# A caller may override the RoboTwin starting seed (e.g. to give every trial
# shard a disjoint seed interval when sharding without a prior seed manifest).
# This maps 1:1 onto eval_policy's ``st_seed = 100000 * (1 + seed)``.
SEED_OVERRIDE="${SEED_OVERRIDE:-}"

if [[ "${EVAL_MODE}" == "single" ]]; then
  if [[ -z "${ROBOTWIN_TASK_NAME}" ]]; then
    echo "ROBOTWIN_TASK_NAME is required when EVAL_MODE=single" >&2
    exit 1
  fi
  mkdir -p "${OUTPUT_DIR}"
  WRAPPER_LOG="${OUTPUT_DIR}/eval_robotwin_single_${ROBOTWIN_TASK_NAME}_${TASK_CONFIG}.log"
  echo "[eval] mode=single task=${ROBOTWIN_TASK_NAME} gpu=${GPU_ID} log=${WRAPPER_LOG}"
  EXTRA_SEED_ARGS=()
  if [[ -n "${SEED_OVERRIDE}" ]]; then
    EXTRA_SEED_ARGS+=("seed=${SEED_OVERRIDE}")
  fi
  python experiments/robotwin/eval_robotwin_single.py     "${COMMON_ARGS[@]}"     "gpu_id=${GPU_ID}"     "EVALUATION.task_name=${ROBOTWIN_TASK_NAME}"     "${EXTRA_SEED_ARGS[@]}"     "${EXTRA_ARGS[@]}" 2>&1 | tee "${WRAPPER_LOG}"
elif [[ "${EVAL_MODE}" == "manager" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  WRAPPER_LOG="${OUTPUT_DIR}/eval_robotwin_manager.log"
  echo "[eval] mode=manager num_gpus=${NUM_GPUS} max_tasks_per_gpu=${MAX_TASKS_PER_GPU} log=${WRAPPER_LOG}"
  python experiments/robotwin/run_robotwin_manager.py     "${COMMON_ARGS[@]}"     "MULTIRUN.num_gpus=${NUM_GPUS}"     "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"     "${EXTRA_ARGS[@]}" 2>&1 | tee "${WRAPPER_LOG}"
else
  echo "Unsupported EVAL_MODE=${EVAL_MODE}. Expected manager or single." >&2
  exit 1
fi
