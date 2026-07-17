#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_KEYS="${MODEL_KEYS:-qwen3_0_6b}" \
BASE_WINDOWS="${BASE_WINDOWS:-8}" \
CONTEXT_512_WINDOWS="${CONTEXT_512_WINDOWS:-2}" \
CONTEXT_1024_WINDOWS="${CONTEXT_1024_WINDOWS:-2}" \
CONTEXT_4096_WINDOWS="${CONTEXT_4096_WINDOWS:-2}" \
SEGMENT_WINDOWS="${SEGMENT_WINDOWS:-2}" \
SEEDS="${SEEDS:-0 1 2}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/table7_quick}" \
  exec "${ROOT}/experiments/table7/run.sh"
