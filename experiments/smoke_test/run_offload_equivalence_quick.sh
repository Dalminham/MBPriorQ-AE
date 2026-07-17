#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NUM_SAMPLES="${NUM_SAMPLES:-1}" exec "${ROOT}/experiments/smoke_test/run_offload_equivalence.sh"
