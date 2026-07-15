#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/software${PYTHONPATH:+:${PYTHONPATH}}"

python -m unittest discover -s "${ROOT}/software/tests" -v
python "${ROOT}/scripts/run_smoke.py"
