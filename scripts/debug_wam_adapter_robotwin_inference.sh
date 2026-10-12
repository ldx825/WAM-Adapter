#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${ROOT:-${REPO_ROOT}}"
PYTHON="${PYTHON:-python}"
HOST="${DEBUGPY_HOST:-0.0.0.0}"
PORT="${DEBUGPY_PORT:-2694}"
GPU_ID="${GPU_ID:-7}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/wam_adapter_robotwin_dev5_240k}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/weights/step_060000.pt}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${RUN_DIR}/config.yaml}"
DATASET_STATS="${DATASET_STATS:-${REPO_ROOT}/data/robotwin2.0/dataset_stats.json}"
TASK_NAME="${TASK_NAME:-press_stapler}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/wam_adapter_robotwin_debug}"

for required in "${PYTHON}" "${CKPT}" "${TRAIN_CONFIG}" "${DATASET_STATS}"; do
    if [[ ! -e "${required}" ]]; then
        echo "[debug-launch] missing required path: ${required}" >&2
        exit 2
    fi
done

echo "[debug-launch] WAMAdapter RoboTwin inference"
echo "[debug-launch] task=${TASK_NAME} config=${TASK_CONFIG} gpu=${GPU_ID} ckpt=${CKPT}"
echo "[debug-launch] command uses: -m debugpy --listen ${HOST}:${PORT} --wait-for-client"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

"${PYTHON}" -m debugpy --listen "${HOST}:${PORT}" --wait-for-client \
    experiments/robotwin/eval_robotwin_single.py \
    ckpt="${CKPT}" \
    gpu_id="${GPU_ID}" \
    EVALUATION.robotwin_root="${ROOT}/third_party/RoboTwin" \
    EVALUATION.output_dir="${OUTPUT_DIR}" \
    EVALUATION.use_training_run_config=true \
    EVALUATION.training_config_path="${TRAIN_CONFIG}" \
    EVALUATION.dataset_stats_path="${DATASET_STATS}" \
    EVALUATION.task_name="${TASK_NAME}" \
    EVALUATION.task_config="${TASK_CONFIG}" \
    EVALUATION.instruction_type=unseen \
    EVALUATION.eval_num_episodes=1 \
    EVALUATION.eval_total_episodes=1 \
    EVALUATION.eval_episode_offset=0 \
    EVALUATION.replan_steps=24 \
    EVALUATION.text_cfg_scale=1.0 \
    EVALUATION.rand_device=cpu \
    EVALUATION.tiled=false \
    EVALUATION.timing_enabled=true \
    EVALUATION.skip_get_obs_within_replan=true \
    EVALUATION.eval_video_fps=10 \
    EVALUATION.eval_video_record_mode=action_step \
    EVALUATION.eval_video_frame_stride=1 \
    EVALUATION.device=cuda &

debug_pid=$!
cleanup() {
    if kill -0 "${debug_pid}" 2>/dev/null; then
        kill "${debug_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 200); do
    if ! kill -0 "${debug_pid}" 2>/dev/null; then
        wait "${debug_pid}"
        exit $?
    fi
    if ss -ltnH "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
        echo "[debugpy] listening on ${HOST}:${PORT}; waiting for VS Code attach"
        wait "${debug_pid}"
        status=$?
        trap - EXIT INT TERM
        exit "${status}"
    fi
    sleep 0.05
done

echo "[debug-launch] debugpy did not bind ${HOST}:${PORT} within 10 seconds" >&2
exit 3
