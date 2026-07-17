#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NUM_SAMPLES="${NUM_SAMPLES:-4}" exec "${ROOT}/experiments/smoke_test/run.sh"
