#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"

TASK_NAME="${TASK_NAME:-real_robot_hdf5_3cam_384_1e-4}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/real_robot_hdf5_raw}"
DATASET_DIRS="${DATASET_DIRS:-}"
PREP_ROOT="${PREP_ROOT:-${REPO_ROOT}/data/real_robot_hdf5_prepared/default}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-${PREP_ROOT}/dataset_stats.json}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${PREP_ROOT}/text_embeds_cache}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-${PREP_ROOT}/latent_cache_Wan2.1-T2V-1.3B}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-8}"
PRECOMPUTE_NUM_WORKERS="${PRECOMPUTE_NUM_WORKERS:-8}"
PRECOMPUTE_GPU_IDS="${PRECOMPUTE_GPU_IDS:-}"
PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PRECOMPUTE_STORAGE_FORMAT="${PRECOMPUTE_STORAGE_FORMAT:-episode_packed_v1}"
OVERWRITE="${OVERWRITE:-false}"
SKIP_STATS="${SKIP_STATS:-false}"
SKIP_TEXT="${SKIP_TEXT:-false}"
SKIP_LATENTS="${SKIP_LATENTS:-false}"
WAM_ADAPTER_VIDEO_BACKBONE_TYPE="${WAM_ADAPTER_VIDEO_BACKBONE_TYPE:-wan2_1_t2v}"
WAM_ADAPTER_VIDEO_BACKBONE_NAME="${WAM_ADAPTER_VIDEO_BACKBONE_NAME:-Wan-AI/Wan2.1-T2V-1.3B}"
WAM_ADAPTER_TEXT_EMBEDDING_MODEL_ID="${WAM_ADAPTER_TEXT_EMBEDDING_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
WAM_ADAPTER_VIDEO_LATENT_SPATIAL_DOWNSAMPLE_FACTOR="${WAM_ADAPTER_VIDEO_LATENT_SPATIAL_DOWNSAMPLE_FACTOR:-2}"
WAM_ADAPTER_APPLY_VIDEO_LATENT_DOWNSAMPLE_TO_ACTION_BRANCH="${WAM_ADAPTER_APPLY_VIDEO_LATENT_DOWNSAMPLE_TO_ACTION_BRANCH:-false}"
EXTRA_ARGS=("$@")

trim_spaces() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  echo "$value"
}

collect_dataset_dirs() {
  local -a dirs=()
  if [[ -n "${DATASET_DIRS}" ]]; then
    local raw="${DATASET_DIRS#[}"
    raw="${raw%]}"
    IFS=',' read -r -a parts <<< "${raw}"
    for part in "${parts[@]}"; do
      local dir
      dir="$(trim_spaces "${part}")"
      dir="${dir#\'}"
      dir="${dir%\'}"
      dir="${dir#\"}"
      dir="${dir%\"}"
      [[ -z "${dir}" ]] && continue
      dirs+=("$(cd "${dir}" && pwd)")
    done
  else
    if compgen -G "${DATASET_ROOT}/episode_*.hdf5" > /dev/null; then
      dirs+=("$(cd "${DATASET_ROOT}" && pwd)")
    fi
    while IFS= read -r candidate; do
      if compgen -G "${candidate}/episode_*.hdf5" > /dev/null; then
        dirs+=("$(cd "${candidate}" && pwd)")
      fi
    done < <(find "${DATASET_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
  if [[ "${#dirs[@]}" -eq 0 ]]; then
    echo "Error: no HDF5 dataset directories found." >&2
    exit 1
  fi
  DATASET_DIR_LIST=("${dirs[@]}")
  DATASET_DIRS_HYDRA="["
  local first=1
  for dir in "${DATASET_DIR_LIST[@]}"; do
    if [[ "${first}" -eq 0 ]]; then
      DATASET_DIRS_HYDRA+=","
    fi
    DATASET_DIRS_HYDRA+="'${dir}'"
    first=0
  done
  DATASET_DIRS_HYDRA+="]"
}

collect_dataset_dirs
mkdir -p "${PREP_ROOT}" "${TEXT_EMBED_CACHE_DIR}" "${LATENT_CACHE_DIR}"

echo "[prepare-real-robot] python=${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import traceback
try:
    from wam_adapter.datasets.lerobot.hdf5_robot_video_dataset import HDF5RobotVideoDataset
    print("[prepare-real-robot] dataset_target_import=ok")
except Exception:
    print("[prepare-real-robot] dataset_target_import=failed")
    traceback.print_exc()
    raise
PY

echo "[prepare-real-robot] task=${TASK_NAME}"
echo "[prepare-real-robot] dataset_dirs=${DATASET_DIRS_HYDRA}"
echo "[prepare-real-robot] prep_root=${PREP_ROOT}"
echo "[prepare-real-robot] dataset_stats_path=${DATASET_STATS_PATH}"
echo "[prepare-real-robot] text_embed_cache_dir=${TEXT_EMBED_CACHE_DIR}"
echo "[prepare-real-robot] latent_cache_dir=${LATENT_CACHE_DIR}"
echo "[prepare-real-robot] mixed_precision=${MIXED_PRECISION}"
echo "[prepare-real-robot] precompute_batch_size=${PRECOMPUTE_BATCH_SIZE}"
echo "[prepare-real-robot] precompute_num_workers=${PRECOMPUTE_NUM_WORKERS}"
echo "[prepare-real-robot] precompute_gpu_ids=${PRECOMPUTE_GPU_IDS:-<default>}"
echo "[prepare-real-robot] precompute_num_processes=${PRECOMPUTE_NUM_PROCESSES}"
echo "[prepare-real-robot] overwrite=${OVERWRITE}"
echo "[prepare-real-robot] model.video_backbone_type=${WAM_ADAPTER_VIDEO_BACKBONE_TYPE}"
echo "[prepare-real-robot] model.video_backbone_name=${WAM_ADAPTER_VIDEO_BACKBONE_NAME}"
echo "[prepare-real-robot] model.text_embedding_model_id=${WAM_ADAPTER_TEXT_EMBEDDING_MODEL_ID}"

run_precompute_py() {
  local script_path="$1"
  shift
  if [[ "${PRECOMPUTE_NUM_PROCESSES}" -gt 1 ]]; then
    if [[ -z "${TORCHRUN_BIN}" ]]; then
      echo "Error: torchrun not found but PRECOMPUTE_NUM_PROCESSES=${PRECOMPUTE_NUM_PROCESSES}" >&2
      exit 1
    fi
    if [[ -n "${PRECOMPUTE_GPU_IDS}" ]]; then
      CUDA_VISIBLE_DEVICES="${PRECOMPUTE_GPU_IDS}" \
        "${TORCHRUN_BIN}" --standalone --nproc_per_node="${PRECOMPUTE_NUM_PROCESSES}" \
        "${script_path}" "$@"
    else
      "${TORCHRUN_BIN}" --standalone --nproc_per_node="${PRECOMPUTE_NUM_PROCESSES}" \
        "${script_path}" "$@"
    fi
  else
    if [[ -n "${PRECOMPUTE_GPU_IDS}" ]]; then
      CUDA_VISIBLE_DEVICES="${PRECOMPUTE_GPU_IDS}" \
        "${PYTHON_BIN}" "${script_path}" "$@"
    else
      "${PYTHON_BIN}" "${script_path}" "$@"
    fi
  fi
}

if [[ "${SKIP_STATS}" != "true" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_real_robot_hdf5_stats.py" \
    "task=${TASK_NAME}" \
    "data.train.dataset_dirs=${DATASET_DIRS_HYDRA}" \
    "data.train.val_set_proportion=0.0" \
    "data.val.dataset_dirs=${DATASET_DIRS_HYDRA}" \
    "data.val.is_training_set=true" \
    "data.val.val_set_proportion=0.0" \
    "+output_stats_path=${DATASET_STATS_PATH}" \
    "${EXTRA_ARGS[@]}"
fi

if [[ "${SKIP_TEXT}" != "true" ]]; then
  run_precompute_py scripts/precompute_text_embeds.py \
    "task=${TASK_NAME}" \
    "mixed_precision=${MIXED_PRECISION}" \
    "overwrite=${OVERWRITE}" \
    "data.train.dataset_dirs=${DATASET_DIRS_HYDRA}" \
    "data.train.val_set_proportion=0.0" \
    "data.val.dataset_dirs=${DATASET_DIRS_HYDRA}" \
    "data.val.is_training_set=true" \
    "data.val.val_set_proportion=0.0" \
    "data.train.text_embedding_cache_dir='${TEXT_EMBED_CACHE_DIR}'" \
    "data.val.text_embedding_cache_dir='${TEXT_EMBED_CACHE_DIR}'" \
    "data.train.text_embedding_model_id='${WAM_ADAPTER_TEXT_EMBEDDING_MODEL_ID}'" \
    "data.val.text_embedding_model_id='${WAM_ADAPTER_TEXT_EMBEDDING_MODEL_ID}'" \
    "${EXTRA_ARGS[@]}"
fi

if [[ "${SKIP_LATENTS}" != "true" ]]; then
  run_precompute_py scripts/precompute_video_latents.py \
    "task=${TASK_NAME}" \
    "mixed_precision=${MIXED_PRECISION}" \
    "overwrite=${OVERWRITE}" \
    "precompute_video_only=true" \
    "precompute_storage_format=${PRECOMPUTE_STORAGE_FORMAT}" \
    "precompute_batch_size=${PRECOMPUTE_BATCH_SIZE}" \
    "precompute_num_workers=${PRECOMPUTE_NUM_WORKERS}" \
    "model.video_backbone_type=${WAM_ADAPTER_VIDEO_BACKBONE_TYPE}" \
    "model.video_backbone_name=${WAM_ADAPTER_VIDEO_BACKBONE_NAME}" \
    "model.video_latent_spatial_downsample_factor=${WAM_ADAPTER_VIDEO_LATENT_SPATIAL_DOWNSAMPLE_FACTOR}" \
    "model.apply_video_latent_downsample_to_action_branch=${WAM_ADAPTER_APPLY_VIDEO_LATENT_DOWNSAMPLE_TO_ACTION_BRANCH}" \
    "data.train.dataset_dirs=${DATASET_DIRS_HYDRA}" \
    "data.train.val_set_proportion=0.0" \
    "data.train.pretrained_norm_stats='${DATASET_STATS_PATH}'" \
    "data.train.text_embedding_cache_dir='${TEXT_EMBED_CACHE_DIR}'" \
    "data.train.use_latent_cache=true" \
    "data.train.latent_cache_dir='${LATENT_CACHE_DIR}'" \
    "${EXTRA_ARGS[@]}"
fi

echo "[prepare-real-robot] done"
