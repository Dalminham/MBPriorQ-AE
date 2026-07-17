#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the public Qwen3-0.6B checkpoint}"
: "${LLAMA_MODEL_PATH:?Set LLAMA_MODEL_PATH to the public Llama-2-7B checkpoint}"

DATASET_PATH="${DATASET_PATH:-wikitext-2-raw-v1}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/table4}"
RESULTS="${OUTPUT_ROOT}/results"
GRADIENTS="${OUTPUT_ROOT}/gradients"
LOGS="${OUTPUT_ROOT}/logs"
mkdir -p "${RESULTS}" "${GRADIENTS}" "${LOGS}"

run_model() {
  local model_key="$1"
  local model_path="$2"
  local gradient_file="${GRADIENTS}/${model_key}.pt"

  if [[ ! -f "${gradient_file}" ]]; then
    "${PYTHON}" "${ROOT}/software/tools/calibrate_activation_gradients.py" \
      --model "${model_path}" --tokenizer "${model_path}" \
      --dataset "${DATASET_PATH}" --split train \
      --sequence-length 512 --num-samples "${CALIBRATION_SAMPLES}" \
      --gradient-checkpointing --output "${gradient_file}" \
      >"${LOGS}/${model_key}__gradient.log" 2>&1
  fi

  "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
    --model "${model_path}" --tokenizer "${model_path}" --model-key "${model_key}" \
    --dataset "${DATASET_PATH}" --method bf16 --backend full_gpu \
    --num-samples "${NUM_SAMPLES}" --output "${RESULTS}/${model_key}__bf16.json" --quiet \
    >"${LOGS}/${model_key}__bf16.log" 2>&1

  for feature in std diff grad diff_grad std_grad; do
    extra=()
    if [[ "${feature}" == *grad* ]]; then
      extra+=(--gradient-file "${gradient_file}")
    fi
    "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
      --model "${model_path}" --tokenizer "${model_path}" --model-key "${model_key}" \
      --dataset "${DATASET_PATH}" --method mbpriorq --backend full_gpu \
      --weight-source none --model-type cloud --feature-mode "${feature}" \
      --num-samples "${NUM_SAMPLES}" --output "${RESULTS}/${model_key}__${feature}.json" \
      --quiet "${extra[@]}" >"${LOGS}/${model_key}__${feature}.log" 2>&1
  done
}

run_model qwen3_0_6b "${QWEN_MODEL_PATH}"
run_model llama2_7b "${LLAMA_MODEL_PATH}"

validation=()
[[ "${NUM_SAMPLES}" == "0" ]] && validation+=(--require-full)
"${PYTHON}" "${ROOT}/software/tools/validate_feature_selection.py" \
  --results "${RESULTS}" \
  --expected "${ROOT}/experiments/table4/expected.csv" \
  --output "${OUTPUT_ROOT}/table4_ppl.csv" "${validation[@]}"
