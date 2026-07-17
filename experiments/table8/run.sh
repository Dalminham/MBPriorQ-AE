#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the public Qwen3-0.6B checkpoint}"
: "${LLAMA_MODEL_PATH:?Set LLAMA_MODEL_PATH to the public Llama-2-7B checkpoint}"

DATASET_PATH="${DATASET_PATH:-wikitext-2-raw-v1}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/table8}"
RESULTS="${OUTPUT_ROOT}/results"
LOGS="${OUTPUT_ROOT}/logs"
mkdir -p "${RESULTS}" "${LOGS}"

run_model() {
  local model_key="$1"
  local model_path="$2"
  for method in bf16 nvfp4 mbpriorq; do
    "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
      --model "${model_path}" --tokenizer "${model_path}" --model-key "${model_key}" \
      --dataset "${DATASET_PATH}" --method bf16 --backend full_gpu \
      --kv-cache-method "${method}" --refined-block-size 4 \
      --num-samples "${NUM_SAMPLES}" --output "${RESULTS}/${model_key}__${method}.json" \
      --quiet >"${LOGS}/${model_key}__${method}.log" 2>&1
  done
}

run_model qwen3_0_6b "${QWEN_MODEL_PATH}"
run_model llama2_7b "${LLAMA_MODEL_PATH}"

validation=()
[[ "${NUM_SAMPLES}" == "0" ]] && validation+=(--require-full)
"${PYTHON}" "${ROOT}/software/tools/validate_kv_cache_results.py" \
  --results "${RESULTS}" \
  --expected "${ROOT}/experiments/table8/expected.csv" \
  --output "${OUTPUT_ROOT}/table8_kv_cache.csv" "${validation[@]}"
