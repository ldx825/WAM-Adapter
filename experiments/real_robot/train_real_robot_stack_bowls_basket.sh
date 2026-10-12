#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/real_robot_hdf5_raw}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/stack_bowls_basket}"

export DATASET_DIRS="${DATASET_DIRS:-${DATASET_DIR}}"
export PREP_ROOT="${PREP_ROOT:-${REPO_ROOT}/data/real_robot_hdf5_prepared/stack_bowls_basket}"
export RUN_TAG="${RUN_TAG:-wam_adapter_real_robot_stack_bowls_basket_3cam384_1e-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29590}"

bash "${SCRIPT_DIR}/train_real_robot_hdf5.sh" "$@"
