#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL_KEYS="${MODEL_KEYS:-qwen3_0_6b llama2_7b}"
: "${WIKITEXT_PATH:?Set WIKITEXT_PATH to a WikiText2 load_from_disk directory}"
: "${PTB_PATH:?Set PTB_PATH to a PTB load_from_disk directory}"
: "${MMLU_PRO_PATH:?Set MMLU_PRO_PATH to an MMLU-Pro load_from_disk directory}"

BASE_WINDOWS="${BASE_WINDOWS:-16}"
CONTEXT_512_WINDOWS="${CONTEXT_512_WINDOWS:-64}"
CONTEXT_1024_WINDOWS="${CONTEXT_1024_WINDOWS:-32}"
CONTEXT_4096_WINDOWS="${CONTEXT_4096_WINDOWS:-8}"
SEGMENT_WINDOWS="${SEGMENT_WINDOWS:-16}"
SEEDS="${SEEDS:-0 1 2 3 4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/local_runs/table7}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/local_runs/checkpoints}"
IMATRIX_PATH="${IMATRIX_PATH:-${ROOT}/data/imatrix/Qwen_Qwen3-0.6B.imatrix}"
DATASETS="${OUTPUT_ROOT}/datasets"
RESULTS="${OUTPUT_ROOT}/results"
PROFILES="${OUTPUT_ROOT}/profiles"
LOGS="${OUTPUT_ROOT}/logs"
mkdir -p "${RESULTS}" "${PROFILES}" "${LOGS}" "${CHECKPOINT_ROOT}"

if [[ ! -f "${DATASETS}/manifest.json" ]]; then
  read -r -a seed_array <<<"${SEEDS}"
  "${PYTHON}" "${ROOT}/software/tools/prepare_vmb_input_variations.py" \
    --wikitext "${WIKITEXT_PATH}" --mmlu-pro "${MMLU_PRO_PATH}" \
    --output-root "${DATASETS}" --seeds "${seed_array[@]}"
fi

make_checkpoint() {
  local model_key="$1"
  local source_path="$2"
  local output_path="$3"
  shift 3
  if [[ ! -f "${output_path}/mbpriorq_ae_prequant_metadata.json" ]]; then
    "${PYTHON}" "${ROOT}/software/tools/prequantize_checkpoint.py" \
      --source-model "${source_path}" --model-key "${model_key}" \
      --output "${output_path}" --method mbpriorq --refined-block-size 4 "$@"
  fi
}

run_profile() {
  local model_key="$1"
  local model_path="$2"
  local tokenizer_path="$3"
  local tag="$4"
  local dataset_path="$5"
  local sequence_length="$6"
  local windows="$7"
  local batch_size="$8"
  local result_path="${RESULTS}/${model_key}__${tag}.json"
  local profile_path="${PROFILES}/${model_key}__${tag}.csv"
  if [[ "${RESUME:-0}" == "1" && -s "${result_path}" && -s "${profile_path}" ]]; then
    return
  fi
  "${PYTHON}" "${ROOT}/software/tools/run_wikitext_ppl.py" \
    --model "${model_path}" --tokenizer "${tokenizer_path}" \
    --model-key "${model_key}" --dataset "${dataset_path}" \
    --method mbpriorq --backend full_gpu --weight-source prequant \
    --model-type cloud --ablation-mode paper --refined-block-size 4 \
    --sequence-length "${sequence_length}" --num-samples "${windows}" \
    --batch-size "${batch_size}" --vmb-profile-output "${profile_path}" \
    --output "${result_path}" --quiet >"${LOGS}/${model_key}__${tag}.log" 2>&1
}

run_model() {
  local model_key="$1"
  local source_path="$2"
  local checkpoint_path="$3"
  shift 3
  make_checkpoint "${model_key}" "${source_path}" "${checkpoint_path}" "$@"

  run_profile "${model_key}" "${checkpoint_path}" "${source_path}" \
    baseline_wt2 "${WIKITEXT_PATH}" 2048 "${BASE_WINDOWS}" 1
  run_profile "${model_key}" "${checkpoint_path}" "${source_path}" \
    domain_ptb "${PTB_PATH}" 2048 "${BASE_WINDOWS}" 1
  run_profile "${model_key}" "${checkpoint_path}" "${source_path}" \
    prompt_mmlu_pro "${DATASETS}/mmlu_pro_prompts" 2048 "${BASE_WINDOWS}" 1
  run_profile "${model_key}" "${checkpoint_path}" "${source_path}" \
    context_512 "${WIKITEXT_PATH}" 512 "${CONTEXT_512_WINDOWS}" 1
  run_profile "${model_key}" "${checkpoint_path}" "${source_path}" \
    context_1024 "${WIKITEXT_PATH}" 1024 "${CONTEXT_1024_WINDOWS}" 1
  run_profile "${model_key}" "${checkpoint_path}" "${source_path}" \
    context_4096 "${WIKITEXT_PATH}" 4096 "${CONTEXT_4096_WINDOWS}" 1

  local seed
  for seed in ${SEEDS}; do
    run_profile "${model_key}" "${checkpoint_path}" "${source_path}" \
      "segment_seed${seed}" "${DATASETS}/wikitext2_segment_seed_${seed}" \
      2048 "${SEGMENT_WINDOWS}" 1
  done
  run_profile "${model_key}" "${checkpoint_path}" "${source_path}" \
    batch_2 "${WIKITEXT_PATH}" 2048 "${BASE_WINDOWS}" 2
  run_profile "${model_key}" "${checkpoint_path}" "${source_path}" \
    batch_4 "${WIKITEXT_PATH}" 2048 "${BASE_WINDOWS}" 4
}

model_validation=()
for model_key in ${MODEL_KEYS}; do
  case "${model_key}" in
    qwen3_0_6b)
      : "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH for qwen3_0_6b}"
      run_model qwen3_0_6b "${QWEN_MODEL_PATH}" \
        "${CHECKPOINT_ROOT}/qwen3_0_6b_mbpriorq_rb4" --imatrix "${IMATRIX_PATH}"
      ;;
    llama2_7b)
      : "${LLAMA_MODEL_PATH:?Set LLAMA_MODEL_PATH for llama2_7b}"
      run_model llama2_7b "${LLAMA_MODEL_PATH}" \
        "${CHECKPOINT_ROOT}/llama2_7b_mbpriorq_rb4"
      ;;
    *)
      echo "Unsupported MODEL_KEYS entry: ${model_key}" >&2
      exit 2
      ;;
  esac
  model_validation+=(--model-key "${model_key}")
done

validation=()
if [[ "${BASE_WINDOWS}" == "16" && "${CONTEXT_512_WINDOWS}" == "64" \
      && "${CONTEXT_1024_WINDOWS}" == "32" && "${CONTEXT_4096_WINDOWS}" == "8" \
      && "${SEGMENT_WINDOWS}" == "16" && "${SEEDS}" == "0 1 2 3 4" \
      && "${MODEL_KEYS}" == "qwen3_0_6b llama2_7b" ]]; then
  validation+=(--require-full)
fi
"${PYTHON}" "${ROOT}/software/tools/summarize_vmb_input_variations.py" \
  --results "${RESULTS}" --profiles "${PROFILES}" \
  --expected "${ROOT}/experiments/table7/expected.csv" \
  --output "${OUTPUT_ROOT}/table7_vmb_prior_robustness.csv" \
  --details-output "${OUTPUT_ROOT}/table7_run_details.csv" \
  "${model_validation[@]}" "${validation[@]}"
