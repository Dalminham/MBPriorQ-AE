#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
: "${MODEL_PATH:?Set MODEL_PATH to the public Qwen3-0.6B checkpoint directory}"

DATASET_PATH="${DATASET_PATH:-wikitext-2-raw-v1}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/qwen3_0_6b_smoke_test}"
IMATRIX_PATH="${IMATRIX_PATH:-${ROOT}/data/imatrix/Qwen_Qwen3-0.6B.imatrix}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/local_runs/checkpoints}"
MBPRIORQ_CHECKPOINT="${CHECKPOINT_ROOT}/qwen3_0_6b_mbpriorq_rb4"
RESULTS="${OUTPUT_ROOT}/results"
mkdir -p "${RESULTS}"

if [[ ! -f "${MBPRIORQ_CHECKPOINT}/mbpriorq_ae_prequant_metadata.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/prequantize_checkpoint.py" \
    --source-model "${MODEL_PATH}" \
    --model-key qwen3_0_6b \
    --output "${MBPRIORQ_CHECKPOINT}" \
    --method mbpriorq \
    --refined-block-size 4 \
    --imatrix "${IMATRIX_PATH}"
fi

"${PYTHON}" "${ROOT}/scripts/run_wikitext_ppl.py" \
  --model "${MODEL_PATH}" --tokenizer "${MODEL_PATH}" \
  --model-key qwen3_0_6b --backend full_gpu \
  --dataset "${DATASET_PATH}" \
  --method bf16 \
  --num-samples "${NUM_SAMPLES}" \
  --output "${RESULTS}/bf16.json"

"${PYTHON}" "${ROOT}/scripts/run_wikitext_ppl.py" \
  --model "${MBPRIORQ_CHECKPOINT}" --tokenizer "${MODEL_PATH}" \
  --model-key qwen3_0_6b --backend full_gpu --weight-source prequant \
  --dataset "${DATASET_PATH}" \
  --method mbpriorq \
  --model-type cloud \
  --ablation-mode paper \
  --refined-block-size 4 \
  --num-samples "${NUM_SAMPLES}" \
  --output "${RESULTS}/mbpriorq.json"

VALIDATION_ARGS=()
if [[ "${NUM_SAMPLES}" == "0" ]]; then
  VALIDATION_ARGS+=(--require-full)
fi
"${PYTHON}" "${ROOT}/scripts/validate_ppl_results.py" \
  --results "${RESULTS}" \
  --expected "${ROOT}/experiments/qwen3_0_6b_smoke_test/expected.csv" \
  "${VALIDATION_ARGS[@]}"
