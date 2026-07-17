#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
: "${MODEL_PATH:?Set MODEL_PATH to the public Qwen3-0.6B checkpoint}"

DATASET_PATH="${DATASET_PATH:-wikitext-2-raw-v1}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/offload_equivalence}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/local_runs/checkpoints}"
IMATRIX_PATH="${IMATRIX_PATH:-${ROOT}/data/imatrix/Qwen_Qwen3-0.6B.imatrix}"
CHECKPOINT="${CHECKPOINT_ROOT}/qwen3_0_6b_mbpriorq_rb4"
RESULTS="${OUTPUT_ROOT}/results"
mkdir -p "${RESULTS}"

if [[ ! -f "${CHECKPOINT}/mbpriorq_ae_prequant_metadata.json" ]]; then
  "${PYTHON}" "${ROOT}/software/tools/prequantize_checkpoint.py" \
    --source-model "${MODEL_PATH}" --model-key qwen3_0_6b --output "${CHECKPOINT}" \
    --method mbpriorq --refined-block-size 4 --imatrix "${IMATRIX_PATH}"
fi

for backend in full_gpu streamed; do
  "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
    --model "${MODEL_PATH}" --tokenizer "${MODEL_PATH}" \
    --model-key qwen3_0_6b --backend "${backend}" \
    --dataset "${DATASET_PATH}" --method bf16 --num-samples "${NUM_SAMPLES}" \
    --output "${RESULTS}/bf16__${backend}.json" --quiet
  "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
    --model "${CHECKPOINT}" --tokenizer "${MODEL_PATH}" \
    --model-key qwen3_0_6b --backend "${backend}" --weight-source prequant \
    --dataset "${DATASET_PATH}" --method mbpriorq --model-type cloud \
    --num-samples "${NUM_SAMPLES}" \
    --output "${RESULTS}/mbpriorq__${backend}.json" --quiet
done

"${PYTHON}" "${ROOT}/software/tools/validate_backend_equivalence.py" --results "${RESULTS}"
