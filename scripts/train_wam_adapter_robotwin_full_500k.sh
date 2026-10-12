#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
EXTRA_ARGS=("$@")

export PATH="${WAM_ADAPTER_ENV_BIN:-$(dirname "$(command -v python)")}:/usr/bin:/bin"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export PYTHONNOUSERSITE=1

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARRAY[@]}}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/robotwin2.0/robotwin2.0}"
NORM_STATS="${NORM_STATS:-${REPO_ROOT}/data/robotwin2.0/dataset_stats.json}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${REPO_ROOT}/data/text_embeds_cache/robotwin}"
# The shared RoboTwin cache was produced with Wan2.2's UMT5-XXL encoder and is
# keyed with the `wan22ti2v5b` cache id.  The resulting [128, 4096] BF16 context
# is compatible with the Wan2.1 1.3B VideoDiT cross-attention used here.
TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-${REPO_ROOT}/data/latent_cache/robotwin50_3cam384_factor2}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/data/models/Wan2.1-T2V-1.3B}"
# The Wan loader resolves text-encoder/VAE assets through DiffSynth's model
# manager. Point its base path at the parent of MODEL_PATH and never re-download.
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$(dirname "${MODEL_PATH}")}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
PUBLIC_ROOT="${PUBLIC_ROOT:-${REPO_ROOT}/runs}"
RUN_NAME="${RUN_NAME:-wam_adapter_robotwin50_3cam384_factor2_r64_gbs128_500k}"
RUN_DIR="${OUTPUT_DIR:-${RUN_DIR:-${PUBLIC_ROOT}/runs/${RUN_NAME}}}"
LOG_DIR="${LOG_DIR:-${PUBLIC_ROOT}/logs/${RUN_NAME}}"
MICRO_BATCH="${MICRO_BATCH:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
MAX_STEPS="${MAX_STEPS:-500000}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-16}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate || true)}"
mkdir -p "${LOG_DIR}"

[[ -n "${ACCELERATE_BIN}" && -x "${ACCELERATE_BIN}" ]] || {
  echo "Error: accelerate executable not found; set ACCELERATE_BIN/WAM_ADAPTER_ENV_BIN." >&2
  exit 2
}
GLOBAL_BATCH=$((MICRO_BATCH * NUM_PROCESSES * GRAD_ACC))
if (( GLOBAL_BATCH != 128 )); then
  echo "Error: expected global batch 128, got ${MICRO_BATCH}*${NUM_PROCESSES}*${GRAD_ACC}=${GLOBAL_BATCH}." >&2
  exit 2
fi

# TRAIN_USE_LATENT_CACHE=false (train-only mode) skips the latent-cache inputs
# and validator; training then decodes video and the model encodes VAE latents
# on the fly (see WAM-Adapter `build_inputs`). Default keeps the cached path.
if [[ "${TRAIN_USE_LATENT_CACHE:-true}" != "false" ]]; then
  REQUIRED_INPUTS=("${DATASET_DIR}" "${NORM_STATS}" "${TEXT_EMBED_CACHE_DIR}" \
    "${LATENT_CACHE_DIR}/_SUCCESS" "${LATENT_CACHE_DIR}/meta.json" \
    "${LATENT_CACHE_DIR}/index.pt" "${MODEL_PATH}")
else
  REQUIRED_INPUTS=("${DATASET_DIR}" "${NORM_STATS}" "${TEXT_EMBED_CACHE_DIR}" "${MODEL_PATH}")
fi
for required in "${REQUIRED_INPUTS[@]}"; do
  [[ -e "${required}" ]] || { echo "Error: missing required input ${required}" >&2; exit 1; }
done

if [[ "${TRAIN_USE_LATENT_CACHE:-true}" != "false" ]]; then
  python "${SCRIPT_DIR}/validate_video_latent_cache.py" \
    "${LATENT_CACHE_DIR}" --expected-factor 2 --expected-shape 16 3 24 20 \
    | tee "${LOG_DIR}/cache_validation.json"
else
  echo "[train] TRAIN_USE_LATENT_CACHE=false: skipping latent-cache inputs and validation (on-the-fly VAE encoding)"
fi

echo "[train] model=WAMAdapter suite=RoboTwin-50"
echo "[train] micro_batch=${MICRO_BATCH} GPUs=${NUM_PROCESSES} grad_acc=${GRAD_ACC} global_batch=${GLOBAL_BATCH}"
echo "[train] input=384x320 cached_factor2=24x20 stride=1 action_loss_steps=32"
echo "[train] LoRA=all30-r64-alpha128 action_blocks=2 save_every=${SAVE_EVERY} total=${MAX_STEPS} epochs=${NUM_EPOCHS}"

env \
  TASK_NAME=robotwin_uncond_3cam_384_1e-4 SUITE_NAME=robotwin \
  GPU_IDS="${GPU_IDS}" NUM_PROCESSES="${NUM_PROCESSES}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29931}" \
  DATASET_DIR="${DATASET_DIR}" LATENT_CACHE_DIR="${LATENT_CACHE_DIR}" \
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR}" \
  MODEL_ID="${MODEL_PATH}" TOKENIZER_MODEL_ID="${MODEL_PATH}" VIDEO_BACKBONE_NAME="${MODEL_PATH}" \
  RUN_TAG="${RUN_NAME}" OUTPUT_DIR="${RUN_DIR}" RESUME="${RESUME:-null}" \
  VIDEO_SIZE='[384,320]' CONCAT_MULTI_CAMERA=robotwin NUM_OUTPUT_CAMERAS=3 \
  NUM_FRAMES=33 ACTION_VIDEO_FREQ_RATIO=4 SAMPLE_START_STRIDE=1 \
  BATCH_SIZE="${MICRO_BATCH}" GRAD_ACC="${GRAD_ACC}" NUM_WORKERS="${NUM_WORKERS}" NUM_EPOCHS="${NUM_EPOCHS}" \
  ACCELERATE_BIN="${ACCELERATE_BIN}" \
  MAX_STEPS="${MAX_STEPS}" SAVE_EVERY="${SAVE_EVERY}" EVAL_EVERY=0 WARMUP_STEPS=1000 \
  LEARNING_RATE=1e-4 LR_SCHEDULER_TYPE=cosine LR_SCHEDULER_TOTAL_STEPS=500000 \
  USE_WAM_ADAPTER=false FREEZE_BACKBONE=true REMOVE_ORIGINAL_ACTION_EXPERT=true \
  USE_BACKBONE_LORA=true LORA_RANK=64 LORA_ALPHA=128.0 LORA_DROPOUT=0.0 \
  TEMPORAL_WEIGHTING_ENABLED=false TEMPORAL_PREFIX_STEPS=32 \
  TEMPORAL_PREFIX_WEIGHT=1.0 TEMPORAL_TAIL_WEIGHT=1.0 \
  MOT_CHECKPOINT_MIXED_ATTN=False VIDEO_LATENT_SPATIAL_DOWNSAMPLE_FACTOR=2 \
  PARAMETER_REPORT_ENABLED=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  bash "${SCRIPT_DIR}/train_libero_core.sh" \
    model=wam_adapter \
    model._target_=wam_adapter.runtime.create_wam_adapter \
    +model.action_decoder_config.num_future_slots=2 \
    +model.action_decoder_config.action_decoder_layers=2 \
    +model.action_decoder_config.readout_gradient_checkpointing=false \
    +model.action_decoder_config.group_positions='[0,16,32]' \
    +model.action_decoder_config.temporal_position_stride=16 \
    +model.action_decoder_config.share_temporal_group_layer_weights="${SHARE_TEMPORAL_GROUP_LAYER_WEIGHTS:-false}" \
    +model.action_decoder_config.cached_latent_spatial_downsample_factor=2 \
    +model.action_decoder_config.cached_latent_spatial_shape='[24,20]' \
    +model.action_decoder_config.readout_layer_indices='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]' \
    data.train.global_sample_stride=1 data.train.sample_start_stride=1 \
    data.val.global_sample_stride=1 ++data.val.sample_start_stride=1 \
    "data.train.pretrained_norm_stats=${NORM_STATS}" \
    "data.val.dataset_dirs=['${DATASET_DIR}']" \
    "data.val.pretrained_norm_stats=${NORM_STATS}" \
    "data.val.text_embedding_cache_dir=${TEXT_EMBED_CACHE_DIR}" \
    "data.train.text_embedding_model_id=${TEXT_EMBED_MODEL_ID}" \
    "data.val.text_embedding_model_id=${TEXT_EMBED_MODEL_ID}" \
    data.val.video_size='[384,320]' data.val.concat_multi_camera=robotwin \
    data.val.processor.num_output_cameras=3 \
    model.video_dit_config.use_gradient_checkpointing=false \
    model.action_dit_config.use_gradient_checkpointing=false \
    train_visualization.enabled=false \
    "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_DIR}/train.log"
