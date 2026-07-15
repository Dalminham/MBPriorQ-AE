#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
: "${MODEL_PATH:?Set MODEL_PATH to the public Qwen3-0.6B checkpoint directory}"

DATASET_PATH="${DATASET_PATH:-wikitext-2-raw-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/activation_attribution}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/local_runs/checkpoints}"
IMATRIX_PATH="${IMATRIX_PATH:-${ROOT}/data/imatrix/Qwen_Qwen3-0.6B.imatrix}"
MBPRIORQ_CHECKPOINT="${CHECKPOINT_ROOT}/qwen3_0_6b_mbpriorq_rb4"
RESULTS="${OUTPUT_ROOT}/results"
mkdir -p "${RESULTS}"

if [[ ! -f "${MBPRIORQ_CHECKPOINT}/mbpriorq_ae_prequant_metadata.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/prequantize_checkpoint.py" \
    --source-model "${MODEL_PATH}" --output "${MBPRIORQ_CHECKPOINT}" \
    --method mbpriorq --refined-block-size 4 --imatrix "${IMATRIX_PATH}"
fi

"${PYTHON}" "${ROOT}/scripts/run_wikitext_ppl.py" \
  --model "${MODEL_PATH}" --dataset "${DATASET_PATH}" --method bf16 \
  --output "${RESULTS}/bf16.json" --quiet
for mode in random_same_ratio static first2_only paper oracle; do
  "${PYTHON}" "${ROOT}/scripts/run_wikitext_ppl.py" \
    --model "${MBPRIORQ_CHECKPOINT}" --dataset "${DATASET_PATH}" \
    --method mbpriorq --model-type cloud --ablation-mode "${mode}" \
    --refined-block-size 4 --output "${RESULTS}/${mode}.json" --quiet
done

"${PYTHON}" "${ROOT}/scripts/validate_ablation_results.py" \
  --results "${RESULTS}" \
  --expected "${ROOT}/experiments/activation_attribution/expected.csv"
