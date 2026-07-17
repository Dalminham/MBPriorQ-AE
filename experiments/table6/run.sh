#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the public Qwen3-0.6B checkpoint}"
: "${LLAMA_MODEL_PATH:?Set LLAMA_MODEL_PATH to the public Llama-2-7B checkpoint}"

DATASET_PATH="${DATASET_PATH:-wikitext-2-raw-v1}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/table6}"
IMATRIX_PATH="${IMATRIX_PATH:-${ROOT}/data/imatrix/Qwen_Qwen3-0.6B.imatrix}"
RESULTS="${OUTPUT_ROOT}/results"
mkdir -p "${RESULTS}"

for size in 8 4 2; do
  "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
    --model "${QWEN_MODEL_PATH}" --tokenizer "${QWEN_MODEL_PATH}" \
    --model-key qwen3_0_6b --backend full_gpu --weight-source online \
    --imatrix "${IMATRIX_PATH}" --dataset "${DATASET_PATH}" --method mbpriorq \
    --num-samples "${NUM_SAMPLES}" \
    --model-type cloud --ablation-mode paper --refined-block-size "${size}" \
    --output "${RESULTS}/qwen__rb${size}.json" --quiet
  "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
    --model "${LLAMA_MODEL_PATH}" --tokenizer "${LLAMA_MODEL_PATH}" \
    --model-key llama2_7b --backend streamed --weight-source online \
    --dataset "${DATASET_PATH}" --method mbpriorq \
    --num-samples "${NUM_SAMPLES}" \
    --model-type cloud --ablation-mode paper --refined-block-size "${size}" \
    --output "${RESULTS}/llama__rb${size}.json" --quiet
done

VALIDATION_ARGS=()
[[ "${NUM_SAMPLES}" == "0" ]] && VALIDATION_ARGS+=(--require-full)
"${PYTHON}" "${ROOT}/software/tools/validate_ablation_results.py" \
  --results "${RESULTS}" \
  --expected "${ROOT}/experiments/table6/expected.csv" \
  "${VALIDATION_ARGS[@]}"
