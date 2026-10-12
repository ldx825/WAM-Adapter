#!/usr/bin/env bash
# WAMAdapter -- RoboTwin-2.0 full-task 500k training on 8 GPUs.
#
# Schedule
#   * trains in SEGMENT_STEPS segments (default 20000) up to TOTAL_STEPS (500000)
#   * saves a complete trainer state plus an eval-ready weight copy at every
#     segment boundary, so an interruption resumes exactly where it stopped
#     (optimizer, LR scheduler, RNG and dataloader states are all restored)
#   * performs NO evaluation during training; the weights are meant to be
#     migrated to a separate evaluation machine afterwards
#
# Batch configuration (keep global batch at 128 for this recipe)
#   micro-batch 16 x 8 GPUs x grad-acc 1 = 128
#
# Required inputs (all overridable through the environment)
#   DATASET_DIR           RoboTwin-2.0 LeRobot root (75G)
#   LATENT_CACHE_DIR      factor-2 episode-packed latent cache
#                         (built by precompute_wam_adapter_robotwin_full.sh)
#   TEXT_EMBED_CACHE_DIR  umT5 text-embedding cache
#   NORM_STATS            dataset_stats.json
#   MODEL_PATH            Wan2.1-T2V-1.3B weights (DiT + VAE + umT5)
#
# Typical launch (paths shown as repo-relative defaults)
#   GPU_IDS=0,1,2,3,4,5,6,7 RUN_NAME=wam_adapter_robotwin50_500k_v1 \
#   DATASET_DIR=data/robotwin2.0/robotwin2.0 \
#   LATENT_CACHE_DIR=data/latent_cache/robotwin50_3cam384_factor2 \
#   TEXT_EMBED_CACHE_DIR=data/text_embeds_cache/robotwin \
#   NORM_STATS=data/robotwin2.0/dataset_stats.json \
#   MODEL_PATH=data/models/Wan2.1-T2V-1.3B \
#   WAM_ADAPTER_ENV_BIN=/path/to/env/bin PYTHON_BIN=/path/to/env/bin/python \
#   bash scripts/run_wam_adapter_robotwin_50task_500k_8gpu.sh
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MICRO_BATCH="${MICRO_BATCH:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
SEGMENT_STEPS="${SEGMENT_STEPS:-20000}"
MAX_TRAIN_ATTEMPTS="${MAX_TRAIN_ATTEMPTS:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"
ALLOW_ANY_GLOBAL_BATCH="${ALLOW_ANY_GLOBAL_BATCH:-false}"
SKIP_SMOKE="${SKIP_SMOKE:-false}"

RUN_NAME="${RUN_NAME:-wam_adapter_robotwin50_3cam384_factor2_r64_gbs128_500k}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
STATUS_FILE="${LOG_DIR}/pipeline_status.log"

DATASET_DIR="${DATASET_DIR:-}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-}"
TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID:-${MODEL_PATH:-}}"
NORM_STATS="${NORM_STATS:-}"
MODEL_PATH="${MODEL_PATH:-}"
SHARE_TEMPORAL_GROUP_LAYER_WEIGHTS="${SHARE_TEMPORAL_GROUP_LAYER_WEIGHTS:-false}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SEGMENT_SCRIPT="${TRAIN_SEGMENT_SCRIPT:-${SCRIPT_DIR}/train_wam_adapter_robotwin_full_500k.sh}"
CACHE_EXPECTED_FACTOR="${CACHE_EXPECTED_FACTOR:-2}"
CACHE_EXPECTED_SHAPE="${CACHE_EXPECTED_SHAPE:-16 3 24 20}"
# 6,075,103 = sum of episode lengths of the RoboTwin2.0 snapshot in use (same
# value as the cache builder); the previous 6,011,575 was from an older
# snapshot and would fail the cache validation right before training.
CACHE_EXPECTED_NUM_SAMPLES="${CACHE_EXPECTED_NUM_SAMPLES:-6075103}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate || true)}"

mkdir -p "${LOG_DIR}"

# The Wan loader resolves text-encoder/VAE assets through DiffSynth's model
# manager; point its base path at the parent of MODEL_PATH (so that
# <base>/Wan-AI/Wan2.1-T2V-1.3B resolves) and never re-download weights.
if [[ -n "${MODEL_PATH}" ]]; then
  export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$(dirname "${MODEL_PATH}")}"
  export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
fi

status() {
  printf '%s status=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${STATUS_FILE}"
}

on_exit() {
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    status "FAILED exit_code=${rc}"
  fi
}
trap on_exit EXIT

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS <= 0 )); then
  echo "Error: at least one GPU is required, got GPU_IDS=${GPU_IDS}." >&2
  exit 2
fi
GLOBAL_BATCH=$(( MICRO_BATCH * NUM_GPUS * GRAD_ACC ))
if (( GLOBAL_BATCH != 128 )) && [[ "${ALLOW_ANY_GLOBAL_BATCH}" != true ]]; then
  echo "Error: micro_batch(${MICRO_BATCH}) * ${NUM_GPUS} GPUs * grad_acc(${GRAD_ACC}) = ${GLOBAL_BATCH}, expected 128." >&2
  echo "       Set ALLOW_ANY_GLOBAL_BATCH=true only if the recipe was retuned." >&2
  exit 2
fi
if (( TOTAL_STEPS <= 0 || SEGMENT_STEPS <= 0 || TOTAL_STEPS % SEGMENT_STEPS != 0 )); then
  echo "Error: TOTAL_STEPS must be a positive multiple of SEGMENT_STEPS." >&2
  exit 2
fi

# TRAIN_USE_LATENT_CACHE=false (train-only mode) skips the latent-cache inputs
# and the video-cache validator; training then runs with on-the-fly VAE encoding.
if [[ "${TRAIN_USE_LATENT_CACHE:-true}" != "false" ]]; then
  PRECHECK_INPUTS=("${DATASET_DIR}" "${LATENT_CACHE_DIR}/meta.json" "${LATENT_CACHE_DIR}/index.pt" \
                   "${LATENT_CACHE_DIR}/_SUCCESS" "${TEXT_EMBED_CACHE_DIR}" "${NORM_STATS}" \
                   "${MODEL_PATH}" "${TRAIN_SEGMENT_SCRIPT}" "${ACCELERATE_BIN}")
else
  PRECHECK_INPUTS=("${DATASET_DIR}" "${TEXT_EMBED_CACHE_DIR}" "${NORM_STATS}" \
                   "${MODEL_PATH}" "${TRAIN_SEGMENT_SCRIPT}" "${ACCELERATE_BIN}")
fi
for required in "${PRECHECK_INPUTS[@]}"; do
  if [[ -z "${required}" || ! -e "${required}" ]]; then
    echo "Error: required input is missing: '${required}'" >&2
    exit 2
  fi
done

status "RESOURCE_CONFIG gpus=${GPU_IDS} num_gpus=${NUM_GPUS} micro_batch=${MICRO_BATCH} grad_acc=${GRAD_ACC} global_batch=${GLOBAL_BATCH}"
status "SCHEDULE total_steps=${TOTAL_STEPS} segment_steps=${SEGMENT_STEPS} save_every=${SEGMENT_STEPS} eval=during_training_off"
status "PATHS dataset=${DATASET_DIR} latent_cache=${LATENT_CACHE_DIR} model=${MODEL_PATH} run_dir=${RUN_DIR}"

if [[ "${TRAIN_USE_LATENT_CACHE:-true}" != "false" ]]; then
  # shellcheck disable=SC2206
  cache_expected_shape=( ${CACHE_EXPECTED_SHAPE} )
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_video_latent_cache.py" \
    "${LATENT_CACHE_DIR}" --expected-factor "${CACHE_EXPECTED_FACTOR}" \
    --expected-shape "${cache_expected_shape[@]}" \
    --expected-num-samples "${CACHE_EXPECTED_NUM_SAMPLES}" > "${LOG_DIR}/cache_validation.json"
  status "CACHE_VALIDATED factor=${CACHE_EXPECTED_FACTOR} shape=${CACHE_EXPECTED_SHAPE} samples=${CACHE_EXPECTED_NUM_SAMPLES}"
else
  status "CACHE_SKIPPED reason=train_use_latent_cache_false mode=on_the_fly_vae"
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_text_embedding_cache.py" \
  --dataset-dir "${DATASET_DIR}" --cache-dir "${TEXT_EMBED_CACHE_DIR}" \
  --model-id "${TEXT_EMBED_MODEL_ID}" > "${LOG_DIR}/text_cache_validation.json"

# ---------------------------------------------------------------------------
# Checkpoint completeness / resume helpers
# ---------------------------------------------------------------------------
checkpoint_complete_at() {
  local root_dir="$1" step_tag="$2"
  local state_dir="${root_dir}/checkpoints/state/step_${step_tag}"
  local deepspeed_dir="${state_dir}/pytorch_model"
  local rank
  [[ -s "${root_dir}/checkpoints/weights/step_${step_tag}.pt" \
    && -s "${state_dir}/trainer_state.json" \
    && -s "${deepspeed_dir}/mp_rank_00_model_states.pt" \
    && -s "${state_dir}/scheduler.bin" ]] || return 1
  for rank in $(seq 0 $((NUM_GPUS - 1))); do
    [[ -s "${deepspeed_dir}/bf16_zero_pp_rank_${rank}_mp_rank_00_optim_states.pt" \
      && -s "${state_dir}/random_states_${rank}.pkl" ]] || return 1
  done
  "${PYTHON_BIN}" - "${state_dir}/trainer_state.json" "$((10#${step_tag}))" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert int(payload["global_step"]) == int(sys.argv[2]), (payload, sys.argv[2])
PY
}

checkpoint_complete() { checkpoint_complete_at "${RUN_DIR}" "$1"; }

latest_complete_state() {
  local best=-1 path tag value
  shopt -s nullglob
  for path in "${RUN_DIR}"/checkpoints/state/step_*; do
    tag="$(basename "${path}")"
    [[ "${tag}" =~ ^step_([0-9]{6})$ ]] || continue
    checkpoint_complete "${BASH_REMATCH[1]}" || continue
    value=$((10#${BASH_REMATCH[1]}))
    if (( value > best )); then best=${value}; fi
  done
  shopt -u nullglob
  printf '%d\n' "${best}"
}

run_train() {
  local target="$1" resume="$2" port="$3" log="$4"
  WAM_ADAPTER_ENV_BIN="${WAM_ADAPTER_ENV_BIN:-}" \
  OUTPUT_DIR="${RUN_DIR}" MAX_STEPS="${target}" SAVE_EVERY="${SEGMENT_STEPS}" \
    TOTAL_STEPS="${TOTAL_STEPS}" RESUME="${resume}" GPU_IDS="${GPU_IDS}" \
    MICRO_BATCH="${MICRO_BATCH}" GRAD_ACC="${GRAD_ACC}" MAIN_PROCESS_PORT="${port}" \
    RUN_TAG="${RUN_NAME}" DATASET_DIR="${DATASET_DIR}" \
    LATENT_CACHE_DIR="${LATENT_CACHE_DIR}" TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR}" \
    TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID}" NORM_STATS="${NORM_STATS}" \
    MODEL_PATH="${MODEL_PATH}" ACCELERATE_BIN="${ACCELERATE_BIN}" \
    SHARE_TEMPORAL_GROUP_LAYER_WEIGHTS="${SHARE_TEMPORAL_GROUP_LAYER_WEIGHTS}" \
    bash "${TRAIN_SEGMENT_SCRIPT}" \
    2>&1 | tee -a "${log}"
}

# ---------------------------------------------------------------------------
# Smoke test: 2 steps, then exact resume for a 4th step on the same run dir.
# Proves forward/backward, checkpoint saving and state restore before 500k.
# ---------------------------------------------------------------------------
if [[ "${SKIP_SMOKE}" != true && ! -f "${RUN_DIR}/.smoke_passed" && ! -e "${RUN_DIR}/checkpoints" ]]; then
  SMOKE_DIR="${RUN_DIR}_smoke"
  if ! checkpoint_complete_at "${SMOKE_DIR}" "000002"; then
    rm -f "${SMOKE_DIR}/checkpoints/weights/step_000002.pt"
    rm -rf "${SMOKE_DIR}/checkpoints/state/step_000002"
    status "SMOKE_TRAIN_START target=2"
    OUTPUT_DIR="${SMOKE_DIR}" MAX_STEPS=2 SAVE_EVERY=2 TOTAL_STEPS="${TOTAL_STEPS}" \
      RESUME=null GPU_IDS="${GPU_IDS}" MICRO_BATCH="${MICRO_BATCH}" GRAD_ACC="${GRAD_ACC}" \
      MAIN_PROCESS_PORT=29941 RUN_TAG="${RUN_NAME}_smoke" \
      DATASET_DIR="${DATASET_DIR}" LATENT_CACHE_DIR="${LATENT_CACHE_DIR}" \
      TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR}" TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID}" \
      NORM_STATS="${NORM_STATS}" MODEL_PATH="${MODEL_PATH}" ACCELERATE_BIN="${ACCELERATE_BIN}" \
      SHARE_TEMPORAL_GROUP_LAYER_WEIGHTS="${SHARE_TEMPORAL_GROUP_LAYER_WEIGHTS}" \
      bash "${TRAIN_SEGMENT_SCRIPT}" 2>&1 | tee "${LOG_DIR}/smoke_train.log"
  else
    status "SMOKE_TRAIN_SKIP_COMPLETE target=2"
  fi
  checkpoint_complete_at "${SMOKE_DIR}" "000002"

  if ! checkpoint_complete_at "${SMOKE_DIR}" "000004"; then
    rm -f "${SMOKE_DIR}/checkpoints/weights/step_000004.pt"
    rm -rf "${SMOKE_DIR}/checkpoints/state/step_000004"
    status "SMOKE_RESUME_START source=2 target=4"
    OUTPUT_DIR="${SMOKE_DIR}" MAX_STEPS=4 SAVE_EVERY=4 TOTAL_STEPS="${TOTAL_STEPS}" \
      RESUME="${SMOKE_DIR}/checkpoints/state/step_000002" GPU_IDS="${GPU_IDS}" \
      MICRO_BATCH="${MICRO_BATCH}" GRAD_ACC="${GRAD_ACC}" MAIN_PROCESS_PORT=29942 \
      RUN_TAG="${RUN_NAME}_smoke_resume" \
      DATASET_DIR="${DATASET_DIR}" LATENT_CACHE_DIR="${LATENT_CACHE_DIR}" \
      TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR}" TEXT_EMBED_MODEL_ID="${TEXT_EMBED_MODEL_ID}" \
      NORM_STATS="${NORM_STATS}" MODEL_PATH="${MODEL_PATH}" ACCELERATE_BIN="${ACCELERATE_BIN}" \
      SHARE_TEMPORAL_GROUP_LAYER_WEIGHTS="${SHARE_TEMPORAL_GROUP_LAYER_WEIGHTS}" \
      bash "${TRAIN_SEGMENT_SCRIPT}" 2>&1 | tee "${LOG_DIR}/smoke_resume.log"
  else
    status "SMOKE_RESUME_SKIP_COMPLETE source=2 target=4"
  fi
  checkpoint_complete_at "${SMOKE_DIR}" "000004"
  mkdir -p "${RUN_DIR}"
  touch "${RUN_DIR}/.smoke_passed"
  rm -rf "${SMOKE_DIR}"
  status "SMOKE_COMPLETE forward_backward=true exact_resume=true"
fi

# ---------------------------------------------------------------------------
# Staged loop with retries
# ---------------------------------------------------------------------------
latest_state_step="$(latest_complete_state)"
resume_state=null
if (( latest_state_step >= 0 )); then
  latest_tag="$(printf '%06d' "${latest_state_step}")"
  resume_state="${RUN_DIR}/checkpoints/state/step_${latest_tag}"
  status "RECOVER_STATE step=${latest_state_step} path=${resume_state}"
fi

for target in $(seq "${SEGMENT_STEPS}" "${SEGMENT_STEPS}" "${TOTAL_STEPS}"); do
  tag="$(printf '%06d' "${target}")"
  weight="${RUN_DIR}/checkpoints/weights/step_${tag}.pt"
  state="${RUN_DIR}/checkpoints/state/step_${tag}"

  if checkpoint_complete "${tag}"; then
    status "TRAIN_SKIP_COMPLETE target=${target}"
    if (( target >= latest_state_step )); then
      latest_state_step=${target}
      resume_state="${state}"
    fi
    continue
  fi

  train_ok=false
  for attempt in $(seq 1 "${MAX_TRAIN_ATTEMPTS}"); do
    status "TRAIN_START target=${target} attempt=${attempt}/${MAX_TRAIN_ATTEMPTS} resume=${resume_state}"
    command_ok=false
    if run_train "${target}" "${resume_state}" 29943 "${LOG_DIR}/train_to_step_${tag}.log"; then
      command_ok=true
    fi
    if checkpoint_complete "${tag}"; then
      train_ok=true
      break
    fi
    status "TRAIN_RETRY target=${target} attempt=${attempt} command_ok=${command_ok}"
    rm -rf "${state}"
    rm -f "${weight}"
    sleep "${RETRY_SLEEP_SECONDS}"
  done
  if [[ "${train_ok}" != true ]]; then
    echo "Error: training failed before step ${target}." >&2
    exit 3
  fi

  old_resume="${resume_state}"
  latest_state_step=${target}
  resume_state="${state}"
  status "TRAIN_COMPLETE target=${target} weight=${weight} state=${state}"
  # A newer complete state proves the previous one can be discarded. Weight
  # copies (step_XXXXXX.pt) are kept for later evaluation on another machine.
  if [[ "${old_resume}" != null && "${old_resume}" != "${state}" && -d "${old_resume}" ]]; then
    rm -rf "${old_resume}"
    status "OLD_STATE_REMOVED path=${old_resume}"
  fi
done

status "PIPELINE_COMPLETE total_steps=${TOTAL_STEPS} run_dir=${RUN_DIR}"
touch "${RUN_DIR}/_TRAINING_SUCCESS"
trap - EXIT
