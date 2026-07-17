#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL_KEYS="${MODEL_KEYS:-qwen3_0_6b llama2_7b}"
[[ " ${MODEL_KEYS} " != *" qwen3_0_6b "* ]] || \
  : "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the public Qwen3-0.6B checkpoint}"
[[ " ${MODEL_KEYS} " != *" llama2_7b "* ]] || \
  : "${LLAMA_MODEL_PATH:?Set LLAMA_MODEL_PATH to the public Llama-2-7B checkpoint}"

DATASET_PATH="${DATASET_PATH:-wikitext-2-raw-v1}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/table5}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/local_runs/checkpoints}"
IMATRIX_PATH="${IMATRIX_PATH:-${ROOT}/data/imatrix/Qwen_Qwen3-0.6B.imatrix}"
MBPRIORQ_CHECKPOINT="${CHECKPOINT_ROOT}/qwen3_0_6b_mbpriorq_rb4"
LLAMA_CHECKPOINT="${CHECKPOINT_ROOT}/llama2_7b_mbpriorq_rb4"
RESULTS="${OUTPUT_ROOT}/results"
mkdir -p "${RESULTS}"

if [[ " ${MODEL_KEYS} " == *" qwen3_0_6b "* && ! -f "${MBPRIORQ_CHECKPOINT}/mbpriorq_ae_prequant_metadata.json" ]]; then
  "${PYTHON}" "${ROOT}/software/tools/prequantize_checkpoint.py" \
    --source-model "${QWEN_MODEL_PATH}" --model-key qwen3_0_6b \
    --output "${MBPRIORQ_CHECKPOINT}" \
    --method mbpriorq --refined-block-size 4 --imatrix "${IMATRIX_PATH}"
fi
if [[ " ${MODEL_KEYS} " == *" llama2_7b "* && ! -f "${LLAMA_CHECKPOINT}/mbpriorq_ae_prequant_metadata.json" ]]; then
  "${PYTHON}" "${ROOT}/software/tools/prequantize_checkpoint.py" \
    --source-model "${LLAMA_MODEL_PATH}" --model-key llama2_7b \
    --output "${LLAMA_CHECKPOINT}" \
    --method mbpriorq --refined-block-size 4
fi

if [[ " ${MODEL_KEYS} " == *" qwen3_0_6b "* ]]; then
  "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
    --model "${QWEN_MODEL_PATH}" --tokenizer "${QWEN_MODEL_PATH}" \
    --model-key qwen3_0_6b --backend full_gpu \
    --dataset "${DATASET_PATH}" --method bf16 \
    --num-samples "${NUM_SAMPLES}" \
    --output "${RESULTS}/qwen__bf16.json" --quiet
  for mode in random_same_ratio static first2_only paper oracle; do
    "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
      --model "${MBPRIORQ_CHECKPOINT}" --tokenizer "${QWEN_MODEL_PATH}" \
      --model-key qwen3_0_6b --backend full_gpu --weight-source prequant \
      --dataset "${DATASET_PATH}" \
      --num-samples "${NUM_SAMPLES}" \
      --method mbpriorq --model-type cloud --ablation-mode "${mode}" \
      --refined-block-size 4 --output "${RESULTS}/qwen__${mode}.json" --quiet
  done
fi

if [[ " ${MODEL_KEYS} " == *" llama2_7b "* ]]; then
  "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
    --model "${LLAMA_MODEL_PATH}" --tokenizer "${LLAMA_MODEL_PATH}" \
    --model-key llama2_7b --backend streamed \
    --dataset "${DATASET_PATH}" --method bf16 \
    --num-samples "${NUM_SAMPLES}" \
    --output "${RESULTS}/llama__bf16.json" --quiet
  for mode in random_same_ratio static first2_only paper oracle; do
    "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
      --model "${LLAMA_CHECKPOINT}" --tokenizer "${LLAMA_MODEL_PATH}" \
      --model-key llama2_7b --backend streamed --weight-source prequant \
      --dataset "${DATASET_PATH}" \
      --num-samples "${NUM_SAMPLES}" \
      --method mbpriorq --model-type cloud --ablation-mode "${mode}" \
      --refined-block-size 4 --output "${RESULTS}/llama__${mode}.json" --quiet
  done
fi

VALIDATION_ARGS=()
if [[ "${NUM_SAMPLES}" == "0" ]]; then
  VALIDATION_ARGS+=(--require-full)
else
  VALIDATION_ARGS+=(--expected-samples "${NUM_SAMPLES}")
fi
for model_key in ${MODEL_KEYS}; do
  case "${model_key}" in
    qwen3_0_6b) VALIDATION_ARGS+=(--row-prefix qwen) ;;
    llama2_7b) VALIDATION_ARGS+=(--row-prefix llama) ;;
    *) echo "Unsupported MODEL_KEYS entry: ${model_key}" >&2; exit 2 ;;
  esac
done
"${PYTHON}" "${ROOT}/software/tools/validate_ablation_results.py" \
  --results "${RESULTS}" \
  --expected "${ROOT}/experiments/table5/expected.csv" \
  "${VALIDATION_ARGS[@]}"
