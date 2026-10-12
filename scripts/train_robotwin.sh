#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
# Keep Hugging Face's generated parquet metadata in a writable per-project
# location.  The large, reusable RoboTwin latent/text caches are configured
# separately and remain on the public disk.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${WAM_ADAPTER_HF_DATASETS_CACHE:-${REPO_ROOT}/.cache/huggingface-datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}"

TASK_NAME="${TASK_NAME:-robotwin_uncond_3cam_384_1e-4}"
EXTRA_ARGS=("$@")

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29578}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_$$}"
RUN_TAG="${RUN_TAG:-wam_adapter_robotwin_3cam384_1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/${RUN_TAG}/${RUN_ID}_wam_adapter}"
WANDB_PROJECT="${WANDB_PROJECT:-wam-adapter}"
WANDB_NAME="${WANDB_NAME:-${RUN_TAG}_${RUN_ID}}"
RESUME="${RESUME:-null}"

ROBOTWIN_TRAIN_DIR="${ROBOTWIN_TRAIN_DIR:-${REPO_ROOT}/data/robotwin2.0/robotwin2.0}"
ROBOTWIN_VAL_DIR="${ROBOTWIN_VAL_DIR:-${REPO_ROOT}/data/robotwin2.0/robotwin2.0}"
ROBOTWIN_NORM_STATS="${ROBOTWIN_NORM_STATS:-${REPO_ROOT}/data/robotwin2.0/dataset_stats.json}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${REPO_ROOT}/data/text_embeds_cache/robotwin}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-./data/latent_cache_Wan2.1-T2V-1.3B/robotwin_3cam384_sharded}"
EPISODE_INDICES_FILE="${EPISODE_INDICES_FILE:-}"

NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_VIDEO_FREQ_RATIO="${ACTION_VIDEO_FREQ_RATIO:-4}"
VIDEO_SIZE="${VIDEO_SIZE:-[384,320]}"
CONCAT_MULTI_CAMERA="${CONCAT_MULTI_CAMERA:-robotwin}"
NUM_OUTPUT_CAMERAS="${NUM_OUTPUT_CAMERAS:-3}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-2}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EVAL_EVERY="${EVAL_EVERY:-0}"
MAX_STEPS="${MAX_STEPS:-500000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"

USE_WAM_ADAPTER="${USE_WAM_ADAPTER:-true}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
REMOVE_ORIGINAL_ACTION_EXPERT="${REMOVE_ORIGINAL_ACTION_EXPERT:-true}"
ADAPTER_LAYERS="${ADAPTER_LAYERS:-[8,16,24]}"
ADAPTER_DIM="${ADAPTER_DIM:-256}"
ADAPTER_SCALE="${ADAPTER_SCALE:-1.0}"

USE_BACKBONE_LORA="${USE_BACKBONE_LORA:-true}"
LORA_LAYER_INDICES="${LORA_LAYER_INDICES:-[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-['self_attn.q','self_attn.k','self_attn.v','self_attn.o','cross_attn.q','cross_attn.k','cross_attn.v','cross_attn.o','ffn.0','ffn.2']}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

TEMPORAL_WEIGHTING_ENABLED="${TEMPORAL_WEIGHTING_ENABLED:-true}"
TEMPORAL_PREFIX_STEPS="${TEMPORAL_PREFIX_STEPS:-24}"
TEMPORAL_PREFIX_WEIGHT="${TEMPORAL_PREFIX_WEIGHT:-1.0}"
TEMPORAL_TAIL_WEIGHT="${TEMPORAL_TAIL_WEIGHT:-0.0}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
MOT_CHECKPOINT_MIXED_ATTN="${MOT_CHECKPOINT_MIXED_ATTN:-False}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

echo "[launch] task=${TASK_NAME}"
echo "[launch] mode=wam_adapter_robotwin gpus=${GPU_IDS} num_processes=${NUM_PROCESSES} port=${MAIN_PROCESS_PORT}"
echo "[launch] train_dir=${ROBOTWIN_TRAIN_DIR}"
echo "[launch] val_dir=${ROBOTWIN_VAL_DIR}"
echo "[launch] latent_cache_dir=${LATENT_CACHE_DIR}"
echo "[launch] text_embed_cache_dir=${TEXT_EMBED_CACHE_DIR}"
echo "[launch] episode_indices_file=${EPISODE_INDICES_FILE:-<all>}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] resume=${RESUME}"
echo "[launch] wandb.project=${WANDB_PROJECT} wandb.name=${WANDB_NAME} wandb.mode=offline"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --gpu_ids "${GPU_IDS}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  scripts/train.py \
  "task=${TASK_NAME}" \
  "output_dir=${OUTPUT_DIR}" \
  "resume=${RESUME}" \
  "wandb.enabled=true" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${WANDB_NAME}" \
  "wandb.mode=offline" \
  "batch_size=${BATCH_SIZE}" \
  "gradient_accumulation_steps=${GRAD_ACC}" \
  "num_workers=${NUM_WORKERS}" \
  "eval_every=${EVAL_EVERY}" \
  "learning_rate=${LEARNING_RATE}" \
  "lr_scheduler_type=${LR_SCHEDULER_TYPE}" \
  "max_steps=${MAX_STEPS}" \
  "save_every=${SAVE_EVERY}" \
  "warmup_steps=${WARMUP_STEPS}" \
  "num_epochs=${NUM_EPOCHS}" \
  "model.mot_checkpoint_mixed_attn=${MOT_CHECKPOINT_MIXED_ATTN}" \
  "model.loss.use_first_frame_residual_video_target=false" \
  "model.loss.action_temporal_weighting.enabled=${TEMPORAL_WEIGHTING_ENABLED}" \
  "model.loss.action_temporal_weighting.num_prefix_steps=${TEMPORAL_PREFIX_STEPS}" \
  "model.loss.action_temporal_weighting.prefix_weight=${TEMPORAL_PREFIX_WEIGHT}" \
  "model.loss.action_temporal_weighting.tail_weight=${TEMPORAL_TAIL_WEIGHT}" \
  "model.wam_adapter.use_wam_adapter=${USE_WAM_ADAPTER}" \
  "model.wam_adapter.freeze_backbone=${FREEZE_BACKBONE}" \
  "model.wam_adapter.remove_original_action_expert=${REMOVE_ORIGINAL_ACTION_EXPERT}" \
  "model.wam_adapter.adapter_layer_indices=${ADAPTER_LAYERS}" \
  "model.wam_adapter.adapter_dim=${ADAPTER_DIM}" \
  "model.wam_adapter.adapter_scale=${ADAPTER_SCALE}" \
  "model.wam_adapter.use_backbone_lora=${USE_BACKBONE_LORA}" \
  "model.wam_adapter.lora_layer_indices=${LORA_LAYER_INDICES}" \
  "model.wam_adapter.lora_target_modules=${LORA_TARGET_MODULES}" \
  "model.wam_adapter.lora_rank=${LORA_RANK}" \
  "model.wam_adapter.lora_alpha=${LORA_ALPHA}" \
  "model.wam_adapter.lora_dropout=${LORA_DROPOUT}" \
  "model.video_backbone_type=wan2_1_t2v" \
  "model.video_backbone_name=Wan-AI/Wan2.1-T2V-1.3B" \
  "model.video_latent_spatial_downsample_factor=2" \
  "model.apply_video_latent_downsample_to_action_branch=false" \
  "timing_breakdown.enabled=false" \
  "timing_breakdown.sync_cuda=false" \
  "train_visualization.enabled=true" \
  "train_visualization.every=500" \
  "train_visualization.fps=8" \
  "train_visualization.tiled=false" \
  "data.train.dataset_dirs=['${ROBOTWIN_TRAIN_DIR}']" \
  "data.val.dataset_dirs=['${ROBOTWIN_VAL_DIR}']" \
  "data.train.episode_indices_file=${EPISODE_INDICES_FILE:-null}" \
  "data.val.episode_indices_file=${EPISODE_INDICES_FILE:-null}" \
  "data.train.num_frames=${NUM_FRAMES}" \
  "data.val.num_frames=${NUM_FRAMES}" \
  "data.train.action_video_freq_ratio=${ACTION_VIDEO_FREQ_RATIO}" \
  "data.val.action_video_freq_ratio=${ACTION_VIDEO_FREQ_RATIO}" \
  "data.train.video_size=${VIDEO_SIZE}" \
  "data.val.video_size=${VIDEO_SIZE}" \
  "data.train.concat_multi_camera=${CONCAT_MULTI_CAMERA}" \
  "data.val.concat_multi_camera=${CONCAT_MULTI_CAMERA}" \
  "data.train.processor.num_output_cameras=${NUM_OUTPUT_CAMERAS}" \
  "data.val.processor.num_output_cameras=${NUM_OUTPUT_CAMERAS}" \
  "data.train.pretrained_norm_stats='${ROBOTWIN_NORM_STATS}'" \
  "data.val.pretrained_norm_stats='${ROBOTWIN_NORM_STATS}'" \
  "data.train.text_embedding_cache_dir='${TEXT_EMBED_CACHE_DIR}'" \
  "data.val.text_embedding_cache_dir='${TEXT_EMBED_CACHE_DIR}'" \
  "data.train.use_latent_cache=true" \
  "data.train.latent_cache_dir=${LATENT_CACHE_DIR}" \
  "data.val.use_latent_cache=false" \
  "data.val.latent_cache_dir=null" \
  "${EXTRA_ARGS[@]}"
