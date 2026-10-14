#!/usr/bin/env bash
# WAMAdapter -- RoboTwin full-task (50-task) rollout evaluation.
#
# Why this shape:
#   * The queue splits every task/condition into small trial shards and every GPU
#     worker pulls the next free shard, so a slow task can never leave the other
#     cards idle (no "N cards waiting for 1 card").
#   * Resumable: re-running with the same OUTPUT_DIR skips shards whose results
#     are already published.
#   * Cold-start seeds are deterministic per shard (seed_base 0, stride 1), which
#     keeps the accepted-seed layout identical across checkpoints.
#
# Task order: scripts/robotwin50_eval_tasks.txt
#   (the 10-task comparison subset first, then the remaining 40 RoboTwin tasks)
#
# Usage:
#   CKPT=/path/to/step_080000.pt OUTPUT_DIR=/path/to/eval_dir \
#     bash scripts/run_robotwin50_eval_trials50.sh
#
# Defaults: GPUs 0-7, 50 trials/task/condition, 10 trials per shard,
#           2 evaluator processes per GPU, official 50-task dataset stats.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ROOT="${ROOT:-${REPO_ROOT}/..}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CKPT="${CKPT:?CKPT must point to a step_XXXXXX.pt checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must point to the evaluation output directory}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_TRIALS="${NUM_TRIALS:-50}"
TRIALS_PER_SHARD="${TRIALS_PER_SHARD:-10}"
MAX_PROCESSES_PER_GPU="${MAX_PROCESSES_PER_GPU:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-14400}"
TASKS="${TASKS:-$(cat "${SCRIPT_DIR}/robotwin50_eval_tasks.txt")}"
DATASET_STATS="${DATASET_STATS:-${REPO_ROOT}/data/robotwin2.0/dataset_stats.json}"

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/data/models}"
export DIFFSYNTH_SKIP_DOWNLOAD=true

mkdir -p "${OUTPUT_DIR}"

echo "[eval-launch] ckpt=${CKPT}"
echo "[eval-launch] output_dir=${OUTPUT_DIR}"
echo "[eval-launch] gpus=${GPU_IDS} procs_per_gpu=${MAX_PROCESSES_PER_GPU}"
echo "[eval-launch] trials/task/condition=${NUM_TRIALS} trials_per_shard=${TRIALS_PER_SHARD}"
echo "[eval-launch] n_tasks=$(awk -F, 'NF>0{print NF}' <<< "${TASKS}")"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_robotwin_eval_queue.py" \
  --checkpoint "${CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --tasks "${TASKS}" \
  --gpu-ids "${GPU_IDS}" \
  --max-processes-per-gpu "${MAX_PROCESSES_PER_GPU}" \
  --trials-per-shard "${TRIALS_PER_SHARD}" \
  --cold-seed-base 0 --cold-seed-stride 1 \
  --num-trials "${NUM_TRIALS}" \
  --dataset-stats "${DATASET_STATS}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --timeout-seconds "${TIMEOUT_SECONDS}"
