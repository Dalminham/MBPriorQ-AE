#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

exec "${PYTHON}" "${ROOT}/scripts/run_table2_ppl.py" --resume "$@"
