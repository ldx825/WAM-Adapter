#!/usr/bin/env bash
# Build every cache required by WAMAdapter on one 8-GPU node.
# Rerunning this script resumes the episode-packed video cache.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${WAM_ADAPTER_ENV_BIN:-}" ]]; then
  export PATH="${WAM_ADAPTER_ENV_BIN}:${PATH}"
fi
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES:-${NUM_GPUS}}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-8}"
PRECOMPUTE_NUM_WORKERS="${PRECOMPUTE_NUM_WORKERS:-0}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"

DATASET_DIR="${DATASET_DIR:-}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-}"
VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:-${MODEL_PATH:-}}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-}"
TEXT_TOKENIZER_MODEL_PATH="${TEXT_TOKENIZER_MODEL_PATH:-${TEXT_MODEL_PATH}}"
TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID:-${TEXT_MODEL_PATH}}"
NORM_STATS="${NORM_STATS:-}"
LOG_DIR="${CACHE_LOG_DIR:-$(dirname "${LATENT_CACHE_DIR:-/tmp/cache}")/logs}"
# 6,075,103 = sum of episode lengths of the RoboTwin2.0 snapshot in use
# (27,500 episodes; verified against meta/episodes.jsonl and the dataset
# validator). The previous 6,011,575 came from an older snapshot and made the
# post-build cache validation fail after hours of encoding.
EXPECTED_NUM_SAMPLES="${EXPECTED_NUM_SAMPLES:-6075103}"
MIN_FREE_GIB="${MIN_FREE_GIB:-1400}"

fail() { echo "Error: $*" >&2; exit 2; }
[[ "${NUM_GPUS}" -eq 8 ]] || fail "cache recipe requires 8 GPUs; GPU_IDS=${GPU_IDS}"
[[ "${PRECOMPUTE_NUM_PROCESSES}" -eq 8 ]] || fail "PRECOMPUTE_NUM_PROCESSES must be 8"
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || fail "python3 not found; set PYTHON_BIN/WAM_ADAPTER_ENV_BIN"
[[ -n "${TORCHRUN_BIN}" && -x "${TORCHRUN_BIN}" ]] || fail "torchrun not found; set TORCHRUN_BIN/WAM_ADAPTER_ENV_BIN"
for required in "${DATASET_DIR}" "${DATASET_DIR}/meta/info.json" \
  "${DATASET_DIR}/meta/tasks.jsonl" "${NORM_STATS}" "${VIDEO_MODEL_PATH}" \
  "${TEXT_MODEL_PATH}" "${TEXT_TOKENIZER_MODEL_PATH}"; do
  [[ -n "${required}" && -e "${required}" ]] || fail "missing required input: '${required}'"
done
[[ -n "${LATENT_CACHE_DIR}" ]] || fail "LATENT_CACHE_DIR is required"
[[ -n "${TEXT_EMBED_CACHE_DIR}" ]] || fail "TEXT_EMBED_CACHE_DIR is required"

mkdir -p "${LATENT_CACHE_DIR}" "${TEXT_EMBED_CACHE_DIR}" "${LOG_DIR}"
available_gib="$(df --output=avail -BG "${LATENT_CACHE_DIR}" | tail -n 1 | tr -dc '0-9')"
(( available_gib >= MIN_FREE_GIB )) || fail \
  "Wan2.2 text plus factor-2 video caches need at least ${MIN_FREE_GIB} GiB free; found ${available_gib} GiB"

status() {
  printf '%s status=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" \
    | tee -a "${LOG_DIR}/cache_pipeline_status.log"
}
on_exit() {
  local rc=$?
  if (( rc != 0 )); then status "CACHE_PIPELINE_FAILED exit_code=${rc}"; fi
}
trap on_exit EXIT

status "CACHE_PREFLIGHT_OK gpus=${GPU_IDS} dataset=${DATASET_DIR} free_gib=${available_gib}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_robotwin_full_dataset.py" "${DATASET_DIR}" \
  | tee "${LOG_DIR}/dataset_validation.json"
TEXT_SUCCESS_MARKER="${TEXT_EMBED_CACHE_DIR}/_SUCCESS"

if [[ -f "${LATENT_CACHE_DIR}/_SUCCESS" ]] \
  && "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_text_embedding_cache.py" \
       --dataset-dir "${DATASET_DIR}" --cache-dir "${TEXT_EMBED_CACHE_DIR}" \
       --model-id "${TEXT_EMBED_MODEL_ID}" > "${LOG_DIR}/text_cache_validation.json" \
  && "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_video_latent_cache.py" \
       "${LATENT_CACHE_DIR}" --expected-factor 2 --expected-shape 16 3 24 20 \
       --expected-num-samples "${EXPECTED_NUM_SAMPLES}" > "${LOG_DIR}/video_cache_validation.json"; then
  status "CACHE_REUSE_COMPLETE samples=${EXPECTED_NUM_SAMPLES}"
  trap - EXIT
  exit 0
fi

# Wan2.2 UMT5-XXL context cache, one entry per text task record.
if [[ -f "${TEXT_SUCCESS_MARKER}" ]] \
  && "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_text_embedding_cache.py" \
       --dataset-dir "${DATASET_DIR}" --cache-dir "${TEXT_EMBED_CACHE_DIR}" \
       --model-id "${TEXT_EMBED_MODEL_ID}" > "${LOG_DIR}/text_cache_validation.json"; then
  status "TEXT_CACHE_REUSE_COMPLETE"
else
  rm -f "${TEXT_SUCCESS_MARKER}"
  status "TEXT_CACHE_START model=${TEXT_EMBED_MODEL_ID}"
  TARGET=robotwin RUN_TEXT=true RUN_VIDEO=false OVERWRITE=false \
    PRECOMPUTE_GPU_IDS="${GPU_IDS}" PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES}" \
    DATASET_DIR="${DATASET_DIR}" TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR}" \
    TEXT_MODEL_PATH="${TEXT_MODEL_PATH}" TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID}" \
    TEXT_TOKENIZER_MODEL_PATH="${TEXT_TOKENIZER_MODEL_PATH}" MODEL_PATH="${VIDEO_MODEL_PATH}" \
    TORCHRUN_BIN="${TORCHRUN_BIN}" \
    bash "${SCRIPT_DIR}/precompute.sh" 2>&1 | tee -a "${LOG_DIR}/precompute_text.log"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_text_embedding_cache.py" \
    --dataset-dir "${DATASET_DIR}" --cache-dir "${TEXT_EMBED_CACHE_DIR}" \
    --model-id "${TEXT_EMBED_MODEL_ID}" \
    | tee "${LOG_DIR}/text_cache_validation.json"
  touch "${TEXT_SUCCESS_MARKER}"
  status "TEXT_CACHE_COMPLETE"
fi

# Wan2.1 VAE cache: BF16, spatial factor=2, one resumable file per episode.
rm -f "${LATENT_CACHE_DIR}/_SUCCESS"
status "VIDEO_CACHE_START model=${VIDEO_MODEL_PATH} format=episode_packed_v1 factor=2"
TARGET=robotwin RUN_TEXT=false RUN_VIDEO=true OVERWRITE=false \
  PRECOMPUTE_GPU_IDS="${GPU_IDS}" PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES}" \
  PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE}" PRECOMPUTE_NUM_WORKERS="${PRECOMPUTE_NUM_WORKERS}" \
  PRECOMPUTE_STORAGE_FORMAT=episode_packed_v1 \
  PRECOMPUTE_LATENT_SPATIAL_DOWNSAMPLE_FACTOR=2 PRECOMPUTE_LOG_EVERY=20 \
  DATASET_DIR="${DATASET_DIR}" LATENT_CACHE_DIR="${LATENT_CACHE_DIR}" \
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR}" MODEL_PATH="${VIDEO_MODEL_PATH}" \
  VIDEO_BACKBONE_TYPE=wan2_1_t2v SAMPLE_START_STRIDE=1 \
  VIDEO_SIZE='[384,320]' CAMERA_SIZE='[240,320]' CAMERA_SHAPE='[3,240,320]' \
  TORCHRUN_BIN="${TORCHRUN_BIN}" \
  bash "${SCRIPT_DIR}/precompute.sh" 2>&1 | tee -a "${LOG_DIR}/precompute_video.log"

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_video_latent_cache.py" \
  "${LATENT_CACHE_DIR}" --expected-factor 2 --expected-shape 16 3 24 20 \
  --expected-num-samples "${EXPECTED_NUM_SAMPLES}" \
  | tee "${LOG_DIR}/video_cache_validation.json"
touch "${LATENT_CACHE_DIR}/_SUCCESS"
status "VIDEO_CACHE_COMPLETE samples=${EXPECTED_NUM_SAMPLES}"
trap - EXIT
