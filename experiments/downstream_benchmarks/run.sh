#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
: "${GSM8K_DATASET_PATH:?Set GSM8K_DATASET_PATH to its Kaggle dataset directory}"
: "${MMLU_DATASET_PATH:?Set MMLU_DATASET_PATH to its Kaggle dataset directory}"
: "${MMLU_PRO_DATASET_PATH:?Set MMLU_PRO_DATASET_PATH to its saved dataset directory}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/downstream_benchmarks}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/local_runs/checkpoints}"
IMATRIX_PATH="${IMATRIX_PATH:-${ROOT}/data/imatrix/Qwen_Qwen3-0.6B.imatrix}"
NUM_EXAMPLES="${NUM_EXAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
RESUME="${RESUME:-1}"
MODEL_KEYS="${MODEL_KEYS:-qwen3_0_6b qwen3_14b}"
BENCHMARKS="${BENCHMARKS:-gsm8k mmlu mmlu_pro}"
QWEN_0_6B_MBP="${CHECKPOINT_ROOT}/qwen3_0_6b_mbpriorq_rb4"
QWEN_14B_MBP="${CHECKPOINT_ROOT}/qwen3_14b_mbpriorq_rb4"
mkdir -p "${OUTPUT_ROOT}"

if [[ " ${MODEL_KEYS} " == *" qwen3_0_6b "* ]]; then
  : "${QWEN_0_6B_MODEL_PATH:?Set QWEN_0_6B_MODEL_PATH}"
fi
if [[ " ${MODEL_KEYS} " == *" qwen3_14b "* ]]; then
  : "${QWEN_14B_MODEL_PATH:?Set QWEN_14B_MODEL_PATH}"
fi

if [[ " ${MODEL_KEYS} " == *" qwen3_0_6b "* && ! -f "${QWEN_0_6B_MBP}/mbpriorq_ae_prequant_metadata.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/prequantize_checkpoint.py" \
    --source-model "${QWEN_0_6B_MODEL_PATH}" --model-key qwen3_0_6b \
    --output "${QWEN_0_6B_MBP}" \
    --method mbpriorq --imatrix "${IMATRIX_PATH}"
fi
if [[ " ${MODEL_KEYS} " == *" qwen3_14b "* && ! -f "${QWEN_14B_MBP}/mbpriorq_ae_prequant_metadata.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/prequantize_checkpoint.py" \
    --source-model "${QWEN_14B_MODEL_PATH}" --model-key qwen3_14b \
    --output "${QWEN_14B_MBP}" \
    --method mbpriorq
fi

for model_key in ${MODEL_KEYS}; do
  if [[ "${model_key}" == "qwen3_0_6b" ]]; then
    source_model="${QWEN_0_6B_MODEL_PATH}"
    mbp_model="${QWEN_0_6B_MBP}"
    load_backend="full_gpu"
  else
    source_model="${QWEN_14B_MODEL_PATH}"
    mbp_model="${QWEN_14B_MBP}"
    load_backend="auto"
  fi
  for method in bf16 mbpriorq; do
    model="${source_model}"
    [[ "${method}" == "mbpriorq" ]] && model="${mbp_model}"
    for benchmark in ${BENCHMARKS}; do
      case "${benchmark}" in
        gsm8k) dataset="${GSM8K_DATASET_PATH}" ;;
        mmlu) dataset="${MMLU_DATASET_PATH}" ;;
        mmlu_pro) dataset="${MMLU_PRO_DATASET_PATH}" ;;
      esac
      resume_args=()
      [[ "${RESUME}" == "1" ]] && resume_args+=(--resume)
      "${PYTHON}" "${ROOT}/scripts/run_downstream_benchmark.py" \
        --model "${model}" --tokenizer "${source_model}" --method "${method}" \
        --model-type cloud --benchmark "${benchmark}" --dataset "${dataset}" \
        --load-backend "${load_backend}" --num-examples "${NUM_EXAMPLES}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" "${resume_args[@]}" \
        --output "${OUTPUT_ROOT}/${model_key}__${method}__${benchmark}.json"
    done
  done
done

if [[ "${NUM_EXAMPLES}" == "0" ]]; then
  read -r -a model_key_args <<< "${MODEL_KEYS}"
  read -r -a benchmark_args <<< "${BENCHMARKS}"
  "${PYTHON}" "${ROOT}/scripts/validate_downstream_results.py" \
    --output-root "${OUTPUT_ROOT}" \
    --expected "${ROOT}/experiments/downstream_benchmarks/expected.csv" \
    --model-keys "${model_key_args[@]}" \
    --benchmarks "${benchmark_args[@]}"
fi
