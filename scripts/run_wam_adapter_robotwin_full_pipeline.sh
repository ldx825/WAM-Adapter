#!/usr/bin/env bash
# Baige entrypoint: build caches and then run the resumable 500k recipe.
# Set STAGE=cache or STAGE=train to run only one half.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-all}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
WAM_ADAPTER_ENV_BIN="${WAM_ADAPTER_ENV_BIN:-$(dirname "$(command -v python3)")}"
export PATH="${WAM_ADAPTER_ENV_BIN}:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate || true)}"

DATASET_DIR="${DATASET_DIR:-}"
NORM_STATS="${NORM_STATS:-}"
VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:-${MODEL_PATH:-}}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-}"
TEXT_TOKENIZER_MODEL_PATH="${TEXT_TOKENIZER_MODEL_PATH:-${TEXT_MODEL_PATH}}"
TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID:-${TEXT_MODEL_PATH}}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-}"
RUN_NAME="${RUN_NAME:-wam_adapter_robotwin50_3cam384_factor2_r64_gbs128_500k}"
RUN_DIR="${RUN_DIR:-}"
STATUS_DIR="${STATUS_DIR:-${RUN_DIR:-${REPO_ROOT}/runs/${RUN_NAME}}/job_logs}"

fail() { echo "Error: $*" >&2; exit 2; }
case "${STAGE}" in all|cache|train) ;; *) fail "STAGE must be all, cache, or train" ;; esac
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
[[ "${#GPU_ARRAY[@]}" -eq 8 ]] || fail "this job requires exactly 8 GPUs; GPU_IDS=${GPU_IDS}"
for executable in "${PYTHON_BIN}" "${TORCHRUN_BIN}" "${ACCELERATE_BIN}"; do
  [[ -n "${executable}" && -x "${executable}" ]] || fail "missing executable: '${executable}'"
done
for required in "${DATASET_DIR}" "${NORM_STATS}" "${VIDEO_MODEL_PATH}" \
  "${TEXT_MODEL_PATH}" "${TEXT_TOKENIZER_MODEL_PATH}"; do
  [[ -n "${required}" && -e "${required}" ]] || fail "missing mounted input: '${required}'"
done
[[ -n "${LATENT_CACHE_DIR}" ]] || fail "LATENT_CACHE_DIR is required and must use persistent storage"
[[ -n "${TEXT_EMBED_CACHE_DIR}" ]] || fail "TEXT_EMBED_CACHE_DIR is required and must use persistent storage"
[[ -n "${RUN_DIR}" ]] || fail "RUN_DIR is required and must use persistent storage"

mkdir -p "${STATUS_DIR}" "${RUN_DIR}" "$(dirname "${LATENT_CACHE_DIR}")" "${TEXT_EMBED_CACHE_DIR}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$(dirname "${VIDEO_MODEL_PATH}")}"
export WAM_ADAPTER_HF_HOME="${WAM_ADAPTER_HF_HOME:-${RUN_DIR}/.cache/huggingface}"

status() {
  printf '%s status=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" \
    | tee -a "${STATUS_DIR}/baige_job_status.log"
}
on_exit() {
  local rc=$?
  if (( rc != 0 )); then status "JOB_FAILED stage=${STAGE} exit_code=${rc}"; fi
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

status "PREFLIGHT_OK stage=${STAGE} host=$(hostname) gpus=${GPU_IDS} python=${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY' | tee "${STATUS_DIR}/runtime_versions.txt"
import platform, torch
print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit("Expected exactly 8 visible CUDA GPUs")
PY
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
  | tee "${STATUS_DIR}/gpu_inventory.txt"

if [[ "${STAGE}" == all || "${STAGE}" == cache ]]; then
  status "CACHE_STAGE_START"
  GPU_IDS="${GPU_IDS}" PRECOMPUTE_NUM_PROCESSES=8 \
    WAM_ADAPTER_ENV_BIN="${WAM_ADAPTER_ENV_BIN}" PYTHON_BIN="${PYTHON_BIN}" TORCHRUN_BIN="${TORCHRUN_BIN}" \
    DATASET_DIR="${DATASET_DIR}" NORM_STATS="${NORM_STATS}" \
    VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH}" TEXT_MODEL_PATH="${TEXT_MODEL_PATH}" \
    TEXT_TOKENIZER_MODEL_PATH="${TEXT_TOKENIZER_MODEL_PATH}" TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID}" \
    LATENT_CACHE_DIR="${LATENT_CACHE_DIR}" TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR}" \
    CACHE_LOG_DIR="${STATUS_DIR}/cache" \
    bash "${SCRIPT_DIR}/precompute_wam_adapter_robotwin_full.sh"
  status "CACHE_STAGE_COMPLETE"
fi

if [[ "${STAGE}" == all || "${STAGE}" == train ]]; then
  status "TRAIN_STAGE_START"
  GPU_IDS="${GPU_IDS}" MICRO_BATCH=16 GRAD_ACC=1 TOTAL_STEPS=500000 SEGMENT_STEPS=20000 \
    WAM_ADAPTER_ENV_BIN="${WAM_ADAPTER_ENV_BIN}" PYTHON_BIN="${PYTHON_BIN}" ACCELERATE_BIN="${ACCELERATE_BIN}" \
    DATASET_DIR="${DATASET_DIR}" NORM_STATS="${NORM_STATS}" MODEL_PATH="${VIDEO_MODEL_PATH}" \
    TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID}" LATENT_CACHE_DIR="${LATENT_CACHE_DIR}" \
    TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR}" RUN_NAME="${RUN_NAME}" RUN_DIR="${RUN_DIR}" \
    bash "${SCRIPT_DIR}/run_wam_adapter_robotwin_50task_500k_8gpu.sh"
  status "TRAIN_STAGE_COMPLETE"
fi

touch "${STATUS_DIR}/_SUCCESS"
status "JOB_COMPLETE stage=${STAGE}"
trap - EXIT TERM INT
