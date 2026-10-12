#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

BUNDLE_DIR="${BUNDLE_DIR:-}"
if [[ -z "${BUNDLE_DIR}" ]]; then
  echo "Error: BUNDLE_DIR is required."
  echo "Example:"
  echo "  BUNDLE_DIR=/path/to/query_bundle HOST=127.0.0.1 PORT=5566 bash experiments/real_robot/query_real_robot_policy_bundle.sh"
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5566}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ACTION_HORIZON="${ACTION_HORIZON:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
APPLY_GRIPPER_POSTPROCESS="${APPLY_GRIPPER_POSTPROCESS:-false}"
EXTRA_ARGS=("$@")

echo "[query-real-robot-bundle] bundle_dir=${BUNDLE_DIR}"
echo "[query-real-robot-bundle] host=${HOST} port=${PORT}"
echo "[query-real-robot-bundle] output_dir=${OUTPUT_DIR:-<bundle-dir>/server_query_response}"
echo "[query-real-robot-bundle] apply_gripper_postprocess=${APPLY_GRIPPER_POSTPROCESS}"

cmd=(
  python "${SCRIPT_DIR}/query_real_robot_policy_bundle.py"
  --bundle-dir "${BUNDLE_DIR}"
  --host "${HOST}"
  --port "${PORT}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${ACTION_HORIZON}" ]]; then
  cmd+=(--action-horizon "${ACTION_HORIZON}")
fi
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  cmd+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi
if [[ "${APPLY_GRIPPER_POSTPROCESS}" == "true" ]]; then
  cmd+=(--apply-gripper-postprocess)
fi
cmd+=("${EXTRA_ARGS[@]}")

"${cmd[@]}"
