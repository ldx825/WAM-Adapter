#!/usr/bin/env bash
set -euo pipefail
TARGET="robotwin" bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/precompute.sh" "$@"
