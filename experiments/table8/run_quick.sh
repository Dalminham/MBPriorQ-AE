#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_KEYS="${MODEL_KEYS:-qwen3_0_6b}" NUM_SAMPLES="${NUM_SAMPLES:-1}" \
  exec "${ROOT}/experiments/table8/run.sh"
