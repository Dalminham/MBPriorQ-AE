#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
: "${MODEL_PATH:?Set MODEL_PATH to the public Qwen3-0.6B checkpoint directory}"

DATASET_PATH="${DATASET_PATH:-wikitext-2-raw-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/granularity_ablation}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/local_runs/checkpoints}"
IMATRIX_PATH="${IMATRIX_PATH:-${ROOT}/data/imatrix/Qwen_Qwen3-0.6B.imatrix}"
RESULTS="${OUTPUT_ROOT}/results"
mkdir -p "${RESULTS}"

for size in 8 4 2; do
  checkpoint="${CHECKPOINT_ROOT}/qwen3_0_6b_mbpriorq_rb${size}"
  if [[ ! -f "${checkpoint}/mbpriorq_ae_prequant_metadata.json" ]]; then
    "${PYTHON}" "${ROOT}/scripts/prequantize_checkpoint.py" \
      --source-model "${MODEL_PATH}" --output "${checkpoint}" \
      --method mbpriorq --refined-block-size "${size}" --imatrix "${IMATRIX_PATH}"
  fi
  "${PYTHON}" "${ROOT}/scripts/run_wikitext_ppl.py" \
    --model "${checkpoint}" --dataset "${DATASET_PATH}" --method mbpriorq \
    --model-type cloud --ablation-mode paper --refined-block-size "${size}" \
    --output "${RESULTS}/rb${size}.json" --quiet
done

"${PYTHON}" "${ROOT}/scripts/validate_ablation_results.py" \
  --results "${RESULTS}" \
  --expected "${ROOT}/experiments/granularity_ablation/expected.csv"
