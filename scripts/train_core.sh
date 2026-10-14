#!/usr/bin/env bash
# Shared training launcher; task-specific wrappers provide dataset and readout overrides.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export HF_HOME="${WAM_ADAPTER_HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${WAM_ADAPTER_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${WAM_ADAPTER_HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

TASK_NAME="${TASK_NAME:-libero_uncond_2cam224_1e-4}"
SUITE_NAME="${SUITE_NAME:-libero}"
EXTRA_ARGS=("$@")
ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "$(command -v python)")/accelerate}"
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
  echo "Error: accelerate executable not found: ${ACCELERATE_BIN}" >&2
  exit 1
fi

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29577}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_$$}"
RUN_TAG="${RUN_TAG:-${SUITE_NAME}_wam_adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/${RUN_TAG}/${RUN_ID}_wam_adapter}"
WANDB_PROJECT="${WANDB_PROJECT:-wam-adapter}"
WANDB_NAME="${WANDB_NAME:-${RUN_TAG}_${RUN_ID}}"
RESUME="${RESUME:-null}"

DATASET_DIR="${DATASET_DIR:-}"
DATASET_DIRS="${DATASET_DIRS:-}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${REPO_ROOT}/data/text_embeds_cache/libero}"
MODEL_ID="${MODEL_ID:-}"
TOKENIZER_MODEL_ID="${TOKENIZER_MODEL_ID:-}"
VIDEO_BACKBONE_NAME="${VIDEO_BACKBONE_NAME:-Wan-AI/Wan2.1-T2V-1.3B}"
VIDEO_BACKBONE_TYPE="${VIDEO_BACKBONE_TYPE:-wan2_1_t2v}"

NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_VIDEO_FREQ_RATIO="${ACTION_VIDEO_FREQ_RATIO:-4}"
VIDEO_SIZE="${VIDEO_SIZE:-[224,448]}"
CONCAT_MULTI_CAMERA="${CONCAT_MULTI_CAMERA:-horizontal}"
NUM_OUTPUT_CAMERAS="${NUM_OUTPUT_CAMERAS:-2}"
SAMPLE_START_STRIDE="${SAMPLE_START_STRIDE:-1}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EVAL_EVERY="${EVAL_EVERY:-0}"
MAX_STEPS="${MAX_STEPS:-150000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-25}"
RETARGET_LR_SCHEDULER_ON_RESUME="${RETARGET_LR_SCHEDULER_ON_RESUME:-false}"

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
TEMPORAL_PREFIX_STEPS="${TEMPORAL_PREFIX_STEPS:-8}"
TEMPORAL_PREFIX_WEIGHT="${TEMPORAL_PREFIX_WEIGHT:-1.0}"
TEMPORAL_TAIL_WEIGHT="${TEMPORAL_TAIL_WEIGHT:-0.0}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LR_SCHEDULER_TOTAL_STEPS="${LR_SCHEDULER_TOTAL_STEPS:-null}"
PARAMETER_REPORT_ENABLED="${PARAMETER_REPORT_ENABLED:-true}"
MOT_CHECKPOINT_MIXED_ATTN="${MOT_CHECKPOINT_MIXED_ATTN:-False}"
VIDEO_LATENT_SPATIAL_DOWNSAMPLE_FACTOR="${VIDEO_LATENT_SPATIAL_DOWNSAMPLE_FACTOR:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET_OVERRIDE_ARGS=()
MODEL_PATH_OVERRIDE_ARGS=()
if [[ -n "${MODEL_ID}" ]]; then
  MODEL_PATH_OVERRIDE_ARGS+=("model.model_id=${MODEL_ID}")
fi
if [[ -n "${TOKENIZER_MODEL_ID}" ]]; then
  MODEL_PATH_OVERRIDE_ARGS+=("model.tokenizer_model_id=${TOKENIZER_MODEL_ID}")
fi
if [[ -n "${DATASET_DIRS}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.dataset_dirs=${DATASET_DIRS}")
elif [[ -n "${DATASET_DIR}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.dataset_dirs=['${DATASET_DIR}']")
fi
if [[ -n "${LATENT_CACHE_DIR}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.latent_cache_dir=${LATENT_CACHE_DIR}")
fi
if [[ -n "${TEXT_EMBED_CACHE_DIR}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.text_embedding_cache_dir='${TEXT_EMBED_CACHE_DIR}'")
fi

echo "[launch] task=${TASK_NAME}"
echo "[launch] mode=wam_adapter gpus=${GPU_IDS} num_processes=${NUM_PROCESSES} port=${MAIN_PROCESS_PORT}"
echo "[launch] suite=${SUITE_NAME}"
echo "[launch] run_tag=${RUN_TAG}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] resume=${RESUME}"
echo "[launch] dataset_dir=${DATASET_DIR:-<config default>}"
echo "[launch] dataset_dirs=${DATASET_DIRS:-<config default>}"
echo "[launch] latent_cache_dir=${LATENT_CACHE_DIR:-<config default>}"
echo "[launch] text_embed_cache_dir=${TEXT_EMBED_CACHE_DIR}"
echo "[launch] wandb.project=${WANDB_PROJECT} wandb.name=${WANDB_NAME} wandb.mode=offline"
echo "[launch] batch_size=${BATCH_SIZE} grad_acc=${GRAD_ACC} num_workers=${NUM_WORKERS} eval_every=${EVAL_EVERY}"
echo "[launch] max_steps=${MAX_STEPS} save_every=${SAVE_EVERY} warmup_steps=${WARMUP_STEPS} num_epochs=${NUM_EPOCHS}"
echo "[launch] learning_rate=${LEARNING_RATE} lr_scheduler_type=${LR_SCHEDULER_TYPE} lr_scheduler_total_steps=${LR_SCHEDULER_TOTAL_STEPS}"
echo "[launch] retarget_lr_scheduler_on_resume=${RETARGET_LR_SCHEDULER_ON_RESUME}"
echo "[launch] model.loss.action_temporal_weighting.num_prefix_steps=${TEMPORAL_PREFIX_STEPS}"

# A segmented run can change memory/compute policy at a checkpoint boundary
# without changing its optimizer or scheduler state.  The marker is scoped to
# one OUTPUT_DIR, so it cannot silently affect other experiments.  It is read
# by each newly launched segment; an already-running segment is unchanged.
RUNTIME_CONTROL_OVERRIDES=()
DISABLE_READOUT_CHECKPOINT_MARKER="${OUTPUT_DIR}/.disable_readout_gradient_checkpointing"
if [[ -f "${DISABLE_READOUT_CHECKPOINT_MARKER}" ]]; then
  RUNTIME_CONTROL_OVERRIDES+=("model.action_decoder_config.readout_gradient_checkpointing=false")
  echo "[launch] runtime_control: readout_gradient_checkpointing=false marker=${DISABLE_READOUT_CHECKPOINT_MARKER}"
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
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
  "lr_scheduler_total_steps=${LR_SCHEDULER_TOTAL_STEPS}" \
  "retarget_lr_scheduler_on_resume=${RETARGET_LR_SCHEDULER_ON_RESUME}" \
  "max_steps=${MAX_STEPS}" \
  "save_every=${SAVE_EVERY}" \
  "warmup_steps=${WARMUP_STEPS}" \
  "num_epochs=${NUM_EPOCHS}" \
  "parameter_report.enabled=${PARAMETER_REPORT_ENABLED}" \
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
  "model.video_backbone_type=${VIDEO_BACKBONE_TYPE}" \
  "model.video_backbone_name=${VIDEO_BACKBONE_NAME}" \
  "model.video_latent_spatial_downsample_factor=${VIDEO_LATENT_SPATIAL_DOWNSAMPLE_FACTOR}" \
  "model.apply_video_latent_downsample_to_action_branch=false" \
  "timing_breakdown.enabled=false" \
  "timing_breakdown.sync_cuda=false" \
  "train_visualization.enabled=true" \
  "train_visualization.every=500" \
  "train_visualization.fps=8" \
  "train_visualization.tiled=false" \
  "data.train.use_latent_cache=${TRAIN_USE_LATENT_CACHE:-true}" \
  "data.train.num_frames=${NUM_FRAMES}" \
  "data.train.action_video_freq_ratio=${ACTION_VIDEO_FREQ_RATIO}" \
  "++data.train.sample_start_stride=${SAMPLE_START_STRIDE}" \
  "data.train.video_size=${VIDEO_SIZE}" \
  "data.train.concat_multi_camera=${CONCAT_MULTI_CAMERA}" \
  "data.train.processor.num_output_cameras=${NUM_OUTPUT_CAMERAS}" \
  "${MODEL_PATH_OVERRIDE_ARGS[@]}" \
  "${DATASET_OVERRIDE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "${RUNTIME_CONTROL_OVERRIDES[@]}"
