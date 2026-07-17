#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT}/hardware/run_modules.sh"
"${ROOT}/hardware/run_system.sh"
