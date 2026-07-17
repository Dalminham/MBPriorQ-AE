#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NUM_SAMPLES="${NUM_SAMPLES:-1}" CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-1}" \
  exec "${ROOT}/experiments/table4/run.sh"
