#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

static_checks() {
  echo "[validate] static repository checks"
  find "${ROOT}/experiments" "${ROOT}/hardware" -type f -name '*.sh' -print0 \
    | xargs -0 -n1 bash -n
  "${PYTHON}" - "${ROOT}" <<'PY'
import ast
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
source_roots = [root / name for name in ("software", "hardware", "experiments", "data")]
for source_root in source_roots:
    for path in source_root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in source_root.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in source_root.rglob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            if not next(reader, None):
                raise ValueError(f"CSV has no header: {path}")
json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
print("[PASS] Python, JSON, CSV, and shell syntax")
PY
}

unit_checks() {
  echo "[validate] dependency-light software tests"
  "${PYTHON}" - <<'PY'
try:
    import pytest  # noqa: F401
    from transformers.cache_utils import DynamicLayer  # noqa: F401
except (ImportError, ModuleNotFoundError) as exc:
    raise SystemExit(
        "Validation dependencies are unavailable. Activate the mbpriorq-ae "
        "environment created from environment.yml.\n"
        f"Original import error: {exc}"
    )
PY
  PYTHONPATH="${ROOT}/software" "${PYTHON}" -m pytest -q "${ROOT}/software/tests"
}

require_software_inputs() {
  local name
  for name in QWEN_MODEL_PATH LLAMA_MODEL_PATH WIKITEXT_PATH PTB_PATH \
    MMLU_PRO_PATH GSM8K_DATASET_PATH MMLU_DATASET_PATH; do
    if [[ -z "${!name:-}" ]]; then
      echo "Missing ${name} for software-quick validation" >&2
      return 2
    fi
  done
}

software_quick() {
  require_software_inputs
  local output="${OUTPUT_ROOT:-${ROOT}/local_runs/acceptance}"
  local checkpoints="${CHECKPOINT_ROOT:-${ROOT}/local_runs/checkpoints}"
  echo "[validate] Qwen3-0.6B PPL smoke"
  MODEL_PATH="${QWEN_MODEL_PATH}" DATASET_PATH="${WIKITEXT_PATH}" \
    OUTPUT_ROOT="${output}/smoke" CHECKPOINT_ROOT="${checkpoints}" \
    "${ROOT}/experiments/smoke_test/run_quick.sh"

  echo "[validate] full-GPU/streamed equivalence"
  MODEL_PATH="${QWEN_MODEL_PATH}" DATASET_PATH="${WIKITEXT_PATH}" \
    OUTPUT_ROOT="${output}/offload_equivalence" CHECKPOINT_ROOT="${checkpoints}" \
    "${ROOT}/experiments/smoke_test/run_offload_equivalence_quick.sh"

  echo "[validate] Tables 2 and 10 one-model path"
  "${ROOT}/experiments/table2/run.sh" \
    --model-root "$(dirname "${QWEN_MODEL_PATH}")" --only qwen3_0_6b \
    --num-samples 1 --dataset "${WIKITEXT_PATH}" \
    --output-root "${output}/table2"

  echo "[validate] Table 3 reduced downstream path"
  QWEN_0_6B_MODEL_PATH="${QWEN_MODEL_PATH}" \
    GSM8K_DATASET_PATH="${GSM8K_DATASET_PATH}" \
    MMLU_DATASET_PATH="${MMLU_DATASET_PATH}" \
    MMLU_PRO_DATASET_PATH="${MMLU_PRO_PATH}" \
    OUTPUT_ROOT="${output}/table3" CHECKPOINT_ROOT="${checkpoints}" RESUME=0 \
    "${ROOT}/experiments/table3/run_quick.sh"

  echo "[validate] Tables 4-6 reduced ablation paths"
  QWEN_MODEL_PATH="${QWEN_MODEL_PATH}" LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH}" \
    DATASET_PATH="${WIKITEXT_PATH}" OUTPUT_ROOT="${output}/table4" \
    "${ROOT}/experiments/table4/run_quick.sh"
  QWEN_MODEL_PATH="${QWEN_MODEL_PATH}" LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH}" \
    DATASET_PATH="${WIKITEXT_PATH}" OUTPUT_ROOT="${output}/table5" \
    CHECKPOINT_ROOT="${checkpoints}" "${ROOT}/experiments/table5/run_quick.sh"
  QWEN_MODEL_PATH="${QWEN_MODEL_PATH}" LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH}" \
    DATASET_PATH="${WIKITEXT_PATH}" OUTPUT_ROOT="${output}/table6" \
    "${ROOT}/experiments/table6/run_quick.sh"

  echo "[validate] Table 7 reduced input-variation path"
  QWEN_MODEL_PATH="${QWEN_MODEL_PATH}" WIKITEXT_PATH="${WIKITEXT_PATH}" \
    PTB_PATH="${PTB_PATH}" MMLU_PRO_PATH="${MMLU_PRO_PATH}" \
    OUTPUT_ROOT="${output}/table7" CHECKPOINT_ROOT="${checkpoints}" \
    "${ROOT}/experiments/table7/run_quick.sh"

  echo "[validate] Table 8 reduced KV-cache path"
  QWEN_MODEL_PATH="${QWEN_MODEL_PATH}" LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH}" \
    DATASET_PATH="${WIKITEXT_PATH}" OUTPUT_ROOT="${output}/table8" \
    "${ROOT}/experiments/table8/run_quick.sh"
  echo "[PASS] all reduced software workflows"
}

hardware_checks() {
  echo "[validate] complete hardware functional regression"
  "${ROOT}/hardware/run_all.sh"
}

usage() {
  echo "Usage: $0 [static|unit|software-quick|hardware|all] [...]"
}

if [[ "$#" -eq 0 ]]; then
  set -- static unit
fi
for target in "$@"; do
  case "${target}" in
    static) static_checks ;;
    unit) unit_checks ;;
    software-quick) software_quick ;;
    hardware) hardware_checks ;;
    all) static_checks; unit_checks; software_quick; hardware_checks ;;
    -h|--help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
done
