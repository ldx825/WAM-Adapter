#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${WAM_ADAPTER_HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${WAM_ADAPTER_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${WAM_ADAPTER_HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

TARGET="${TARGET:-${1:-libero_spatial}}"
PRECOMPUTE_GPU_IDS="${PRECOMPUTE_GPU_IDS:-0,1,2,3}"
PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES:-4}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-32}"
PRECOMPUTE_NUM_WORKERS="${PRECOMPUTE_NUM_WORKERS:-8}"
PRECOMPUTE_SHARD_SIZE="${PRECOMPUTE_SHARD_SIZE:-1024}"
PRECOMPUTE_STORAGE_FORMAT="${PRECOMPUTE_STORAGE_FORMAT:-sharded_v1}"
PRECOMPUTE_LATENT_SPATIAL_DOWNSAMPLE_FACTOR="${PRECOMPUTE_LATENT_SPATIAL_DOWNSAMPLE_FACTOR:-1}"
PRECOMPUTE_LOG_EVERY="${PRECOMPUTE_LOG_EVERY:-10}"
PRECOMPUTE_VIDEO_SIZE="${VIDEO_SIZE:-[224,448]}"
PRECOMPUTE_CAMERA_SIZE="${CAMERA_SIZE:-[224,224]}"
PRECOMPUTE_CAMERA_SHAPE="${CAMERA_SHAPE:-[3,224,224]}"
SAMPLE_START_STRIDE="${SAMPLE_START_STRIDE:-1}"
RUN_TEXT="${RUN_TEXT:-true}"
RUN_VIDEO="${RUN_VIDEO:-true}"
OVERWRITE="${OVERWRITE:-false}"
EPISODE_INDICES_FILE="${EPISODE_INDICES_FILE:-}"
REQUESTED_DATASET_DIR="${DATASET_DIR:-}"
REQUESTED_LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-}"
PRECOMPUTE_MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/data/models/Wan2.1-T2V-1.3B}"
PRECOMPUTE_VIDEO_BACKBONE_TYPE="${VIDEO_BACKBONE_TYPE:-wan2_1_t2v}"
PRECOMPUTE_TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-${TEXT_EMBED_MODEL_ID:-${PRECOMPUTE_MODEL_PATH}}}"
PRECOMPUTE_TEXT_TOKENIZER_PATH="${TEXT_TOKENIZER_MODEL_PATH:-${PRECOMPUTE_MODEL_PATH}}"
TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID:-${PRECOMPUTE_TEXT_MODEL_PATH}}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"

if [[ -z "${TORCHRUN_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Error: torchrun executable not found; set TORCHRUN_BIN or WAM_ADAPTER_ENV_BIN." >&2
  exit 2
fi

case "${TARGET}" in
  libero_spatial)
    TASK_NAME="libero_uncond_2cam224_1e-4"
    DATASET_DIR="./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"
    LATENT_CACHE_DIR="./data/latent_cache_Wan2.1-T2V-1.3B/libero_spatial_2cam224"
    TEXT_CACHE_DIR="./data/text_embeds_cache/libero"
    ;;
  libero_object)
    TASK_NAME="libero_uncond_2cam224_1e-4"
    DATASET_DIR="./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot"
    LATENT_CACHE_DIR="./data/latent_cache_Wan2.1-T2V-1.3B/libero_object_2cam224"
    TEXT_CACHE_DIR="./data/text_embeds_cache/libero"
    ;;
  libero_goal)
    TASK_NAME="libero_uncond_2cam224_1e-4"
    DATASET_DIR="./data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot"
    LATENT_CACHE_DIR="./data/latent_cache_Wan2.1-T2V-1.3B/libero_goal_2cam224"
    TEXT_CACHE_DIR="./data/text_embeds_cache/libero"
    ;;
  libero_10)
    TASK_NAME="libero_uncond_2cam224_1e-4"
    DATASET_DIR="./data/libero_mujoco3.3.2/libero_10_no_noops_lerobot"
    LATENT_CACHE_DIR="./data/latent_cache_Wan2.1-T2V-1.3B/libero_10_2cam224"
    TEXT_CACHE_DIR="./data/text_embeds_cache/libero"
    ;;
  robotwin)
    TASK_NAME="robotwin_uncond_3cam_384_1e-4"
    DATASET_DIR="./data/robotwin2.0/robotwin2.0"
    LATENT_CACHE_DIR="./data/latent_cache_Wan2.1-T2V-1.3B/robotwin_3cam384_sharded"
    TEXT_CACHE_DIR="./data/text_embeds_cache/robotwin"
    ;;
  *)
    echo "Unsupported TARGET=${TARGET}. Expected one of: libero_spatial, libero_object, libero_goal, libero_10, robotwin" >&2
    exit 1
    ;;
esac

DATASET_DIR="${REQUESTED_DATASET_DIR:-${ROBOTWIN_TRAIN_DIR:-${DATASET_DIR}}}"
LATENT_CACHE_DIR="${REQUESTED_LATENT_CACHE_DIR:-${LATENT_CACHE_DIR}}"
TEXT_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${TEXT_CACHE_DIR}}"

echo "[precompute] target=${TARGET}"
echo "[precompute] task=${TASK_NAME}"
echo "[precompute] dataset_dir=${DATASET_DIR}"
echo "[precompute] text_cache_dir=${TEXT_CACHE_DIR}"
echo "[precompute] latent_cache_dir=${LATENT_CACHE_DIR}"
echo "[precompute] episode_indices_file=${EPISODE_INDICES_FILE:-<all>}"
echo "[precompute] text_model=${PRECOMPUTE_TEXT_MODEL_PATH} text_cache_id=${TEXT_EMBED_MODEL_ID}"
echo "[precompute] video_model=${PRECOMPUTE_MODEL_PATH} video_backbone_type=${PRECOMPUTE_VIDEO_BACKBONE_TYPE}"

if [[ "${RUN_TEXT}" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="${PRECOMPUTE_GPU_IDS}" "${TORCHRUN_BIN}" --standalone --nproc_per_node="${PRECOMPUTE_NUM_PROCESSES}" \
    scripts/precompute_text_embeds.py \
    "task=${TASK_NAME}" \
    "overwrite=${OVERWRITE}" \
    "model.model_id=${PRECOMPUTE_TEXT_MODEL_PATH}" \
    "model.tokenizer_model_id=${PRECOMPUTE_TEXT_TOKENIZER_PATH}" \
    "+precompute_episode_indices_file=${EPISODE_INDICES_FILE:-null}" \
    "data.train.dataset_dirs=['${DATASET_DIR}']" \
    "++data.train.episode_indices_file=${EPISODE_INDICES_FILE:-null}" \
    "++data.train.sample_start_stride=${SAMPLE_START_STRIDE}" \
    "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}" \
    "data.train.text_embedding_model_id=${TEXT_EMBED_MODEL_ID}" \
    "++data.val.dataset_dirs=['${DATASET_DIR}']" \
    "++data.val.episode_indices_file=${EPISODE_INDICES_FILE:-null}" \
    "++data.val.sample_start_stride=${SAMPLE_START_STRIDE}" \
    "++data.val.text_embedding_cache_dir=${TEXT_CACHE_DIR}" \
    "++data.val.text_embedding_model_id=${TEXT_EMBED_MODEL_ID}"
fi

if [[ "${RUN_VIDEO}" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="${PRECOMPUTE_GPU_IDS}" "${TORCHRUN_BIN}" --standalone --nproc_per_node="${PRECOMPUTE_NUM_PROCESSES}" \
    scripts/precompute_video_latents.py \
    "task=${TASK_NAME}" \
    "overwrite=${OVERWRITE}" \
    "model.video_backbone_type=${PRECOMPUTE_VIDEO_BACKBONE_TYPE}" \
    "model.model_id=${PRECOMPUTE_MODEL_PATH}" \
    "model.tokenizer_model_id=${PRECOMPUTE_MODEL_PATH}" \
    "model.video_backbone_name=${PRECOMPUTE_MODEL_PATH}" \
    "precompute_storage_format=${PRECOMPUTE_STORAGE_FORMAT}" \
    "precompute_video_only=true" \
    "precompute_shard_size=${PRECOMPUTE_SHARD_SIZE}" \
    "precompute_batch_size=${PRECOMPUTE_BATCH_SIZE}" \
    "precompute_num_workers=${PRECOMPUTE_NUM_WORKERS}" \
    "precompute_cache_dtype=model" \
    "precompute_latent_spatial_downsample_factor=${PRECOMPUTE_LATENT_SPATIAL_DOWNSAMPLE_FACTOR}" \
    "precompute_resume=true" \
    "precompute_timing.enabled=true" \
    "precompute_timing.sync_cuda=false" \
    "precompute_timing.log_every=${PRECOMPUTE_LOG_EVERY}" \
    "data.train.dataset_dirs=['${DATASET_DIR}']" \
    "++data.train.episode_indices_file=${EPISODE_INDICES_FILE:-null}" \
    "++data.train.sample_start_stride=${SAMPLE_START_STRIDE}" \
    "data.train.video_size=${PRECOMPUTE_VIDEO_SIZE}" \
    "data.train.shape_meta.images.0.shape=${PRECOMPUTE_CAMERA_SHAPE}" \
    "data.train.shape_meta.images.1.shape=${PRECOMPUTE_CAMERA_SHAPE}" \
    "data.train.processor.train_transforms.1.size=${PRECOMPUTE_CAMERA_SIZE}" \
    "data.train.latent_cache_dir=${LATENT_CACHE_DIR}" \
    "++data.val.dataset_dirs=['${DATASET_DIR}']" \
    "++data.val.episode_indices_file=${EPISODE_INDICES_FILE:-null}" \
    "++data.val.sample_start_stride=${SAMPLE_START_STRIDE}" \
    "++data.val.video_size=${PRECOMPUTE_VIDEO_SIZE}" \
    "++data.val.shape_meta.images.0.shape=${PRECOMPUTE_CAMERA_SHAPE}" \
    "++data.val.shape_meta.images.1.shape=${PRECOMPUTE_CAMERA_SHAPE}" \
    "++data.val.processor.train_transforms.1.size=${PRECOMPUTE_CAMERA_SIZE}" \
    "++data.val.latent_cache_dir=${LATENT_CACHE_DIR}"
fi
