#!/usr/bin/env bash
# Wait for one RoboTwin-50 checkpoint to land, then evaluate it on the free GPUs.
#
# Usage (only the requested checkpoint is ever evaluated):
#   WAIT_STEP=240000 bash scripts/run_robotwin50_eval_wait_step.sh
#
# Ordering: before polling it waits for any evaluation of this weights directory
# to drain, so two evaluations never share the same cards.  Intermediate
# checkpoints are intentionally skipped — set WAIT_STEP to the step you want.
#
# Everything is resumable: re-running against an existing OUTPUT_DIR only
# evaluates the shards that have no published result yet.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WEIGHTS_DIR="${WEIGHTS_DIR:-${REPO_ROOT:-.}/runs/wam_adapter_robotwin50_500k}"
EVAL_TAG="${EVAL_TAG:-eval_robotwin50_alltasks_trials50}"
WAIT_STEP="${WAIT_STEP:-240000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-300}"
WAIT_MAX_SECONDS="${WAIT_MAX_SECONDS:-172800}"   # 48 h

WATCH_LOG="${WEIGHTS_DIR}/eval_watch.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${WATCH_LOG}"; }
tag_of() { printf '%06d' "$1"; }
ckpt_of() { echo "${WEIGHTS_DIR}/step_$(tag_of "$1").pt"; }
out_of() { echo "${WEIGHTS_DIR}/${EVAL_TAG}_step_$(tag_of "$1")"; }
queue_running() { pgrep -f "run_robotwin_eval_queue.py .*$1" >/dev/null 2>&1; }
any_eval_running() { pgrep -f "run_robotwin_eval_queue.py .*${WEIGHTS_DIR}" >/dev/null 2>&1; }

mkdir -p "${WEIGHTS_DIR}"
touch "${WATCH_LOG}"

wait_step_tag="$(tag_of "${WAIT_STEP}")"
wait_ckpt="$(ckpt_of "${WAIT_STEP}")"
wait_out="$(out_of "${WAIT_STEP}")"

log "watcher armed: target=step_${wait_step_tag} (intermediate checkpoints skipped)"

# 1) Let any in-flight evaluation of this weights directory drain first: the two
#    runs would otherwise share the same cards.
if any_eval_running; then
  log "waiting for an in-flight evaluation to drain before starting step_${wait_step_tag}"
  while any_eval_running; do sleep "${POLL_SECONDS}"; done
  log "in-flight evaluation drained"
fi

# 2) Block until the target checkpoint lands.
deadline=$(( $(date +%s) + WAIT_MAX_SECONDS ))
waited_since_log=0
while [[ ! -f "${wait_ckpt}" ]]; do
  if (( $(date +%s) > deadline )); then
    log "ERROR: timed out after ${WAIT_MAX_SECONDS}s waiting for ${wait_ckpt}; exiting"
    exit 1
  fi
  if (( waited_since_log >= 3600 )); then
    log "still waiting for ${wait_ckpt} ..."
    waited_since_log=0
  fi
  sleep "${POLL_SECONDS}"
  waited_since_log=$(( waited_since_log + POLL_SECONDS ))
done

# 3) Verify the upload is complete (size stable across one poll interval).
prev_size=-1
while true; do
  cur_size=$(stat -c %s "${wait_ckpt}" 2>/dev/null || echo -1)
  if [[ "${cur_size}" == "${prev_size}" && "${cur_size}" -gt 0 ]]; then break; fi
  prev_size="${cur_size}"
  sleep 30
done
log "checkpoint detected and stable: ${wait_ckpt} ($(numfmt --to=iec "${cur_size}"))"

# 4) Evaluate it.
mkdir -p "${wait_out}"
log "eval start step=${wait_step_tag} out=${wait_out}"
CKPT="${wait_ckpt}" OUTPUT_DIR="${wait_out}" GPU_IDS="${GPU_IDS}" \
  bash "${SCRIPT_DIR}/run_robotwin50_eval_trials50.sh" >> "${wait_out}/pipeline.log" 2>&1
rc=$?

if [[ -f "${wait_out}/summary.json" ]]; then
  log "eval complete step=${wait_step_tag} rc=${rc} (summary.json present)"
else
  log "eval INCOMPLETE step=${wait_step_tag} rc=${rc} — re-run ${wait_out} to resume"
fi
log "watcher done"
