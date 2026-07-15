#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the public Qwen3-0.6B checkpoint}"
: "${LLAMA_MODEL_PATH:?Set LLAMA_MODEL_PATH to the public Llama-2-7B checkpoint}"

DATASET_PATH="${DATASET_PATH:-wikitext-2-raw-v1}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/activation_attribution}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/local_runs/checkpoints}"
IMATRIX_PATH="${IMATRIX_PATH:-${ROOT}/data/imatrix/Qwen_Qwen3-0.6B.imatrix}"
MBPRIORQ_CHECKPOINT="${CHECKPOINT_ROOT}/qwen3_0_6b_mbpriorq_rb4"
LLAMA_CHECKPOINT="${CHECKPOINT_ROOT}/llama2_7b_mbpriorq_rb4"
RESULTS="${OUTPUT_ROOT}/results"
mkdir -p "${RESULTS}"

if [[ ! -f "${MBPRIORQ_CHECKPOINT}/mbpriorq_ae_prequant_metadata.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/prequantize_checkpoint.py" \
    --source-model "${QWEN_MODEL_PATH}" --model-key qwen3_0_6b \
    --output "${MBPRIORQ_CHECKPOINT}" \
    --method mbpriorq --refined-block-size 4 --imatrix "${IMATRIX_PATH}"
fi
if [[ ! -f "${LLAMA_CHECKPOINT}/mbpriorq_ae_prequant_metadata.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/prequantize_checkpoint.py" \
    --source-model "${LLAMA_MODEL_PATH}" --model-key llama2_7b \
    --output "${LLAMA_CHECKPOINT}" \
    --method mbpriorq --refined-block-size 4
fi

"${PYTHON}" "${ROOT}/scripts/run_wikitext_ppl.py" \
  --model "${QWEN_MODEL_PATH}" --tokenizer "${QWEN_MODEL_PATH}" \
  --model-key qwen3_0_6b --backend full_gpu \
  --dataset "${DATASET_PATH}" --method bf16 \
  --num-samples "${NUM_SAMPLES}" \
  --output "${RESULTS}/qwen__bf16.json" --quiet
for mode in random_same_ratio static first2_only paper oracle; do
  "${PYTHON}" "${ROOT}/scripts/run_wikitext_ppl.py" \
    --model "${MBPRIORQ_CHECKPOINT}" --tokenizer "${QWEN_MODEL_PATH}" \
    --model-key qwen3_0_6b --backend full_gpu --weight-source prequant \
    --dataset "${DATASET_PATH}" \
    --num-samples "${NUM_SAMPLES}" \
    --method mbpriorq --model-type cloud --ablation-mode "${mode}" \
    --refined-block-size 4 --output "${RESULTS}/qwen__${mode}.json" --quiet
done

"${PYTHON}" "${ROOT}/scripts/run_wikitext_ppl.py" \
  --model "${LLAMA_MODEL_PATH}" --tokenizer "${LLAMA_MODEL_PATH}" \
  --model-key llama2_7b --backend streamed \
  --dataset "${DATASET_PATH}" --method bf16 \
  --num-samples "${NUM_SAMPLES}" \
  --output "${RESULTS}/llama__bf16.json" --quiet
for mode in random_same_ratio static first2_only paper oracle; do
  "${PYTHON}" "${ROOT}/scripts/run_wikitext_ppl.py" \
    --model "${LLAMA_CHECKPOINT}" --tokenizer "${LLAMA_MODEL_PATH}" \
    --model-key llama2_7b --backend streamed --weight-source prequant \
    --dataset "${DATASET_PATH}" \
    --num-samples "${NUM_SAMPLES}" \
    --method mbpriorq --model-type cloud --ablation-mode "${mode}" \
    --refined-block-size 4 --output "${RESULTS}/llama__${mode}.json" --quiet
done

VALIDATION_ARGS=()
[[ "${NUM_SAMPLES}" == "0" ]] && VALIDATION_ARGS+=(--require-full)
"${PYTHON}" "${ROOT}/scripts/validate_ablation_results.py" \
  --results "${RESULTS}" \
  --expected "${ROOT}/experiments/activation_attribution/expected.csv" \
  "${VALIDATION_ARGS[@]}"
