#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_KEYS="${MODEL_KEYS:-qwen3_0_6b}" NUM_EXAMPLES="${NUM_EXAMPLES:-1}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}" \
  exec "${ROOT}/experiments/table3/run.sh"
