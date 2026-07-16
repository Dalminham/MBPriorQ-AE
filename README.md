# MBPriorQ Artifact Evaluation

This repository is the self-contained artifact for:

> **Software-Hardware Co-Design of Prior-Aware W4A4 Micro-Block Quantization
> for Robust LLM Inference Across Model Families**

It contains only the author-generated MBPriorQ software, paper experiment
drivers, SpinalHDL design sources, and functional simulations. Model weights
and datasets are downloaded separately. Implementations of comparison methods
and all commercial synthesis material are intentionally excluded.

## What Can Be Reproduced?

| Paper location | Public entry point | Scope |
|---|---|---|
| **Table 2**, model-family WikiText2 PPL | [`experiments/table2_ppl/run.sh`](experiments/table2_ppl/README.md) | One command runs all 19 models under BF16 and MBPriorQ. |
| Lightweight model smoke | [`experiments/qwen3_0_6b_smoke_test/run.sh`](experiments/qwen3_0_6b_smoke_test/README.md) | Reproduces the Qwen3-0.6B BF16/MBPriorQ row without large-model storage or offload requirements. |
| **Table 3**, downstream accuracy | [`experiments/downstream_benchmarks/run.sh`](experiments/downstream_benchmarks/README.md) | GSM8K, MMLU, and MMLU-Pro for Qwen3-0.6B and Qwen3-14B. |
| **Table 5**, accuracy attribution | [`experiments/activation_attribution/run.sh`](experiments/activation_attribution/README.md) | All rows for both Qwen3-0.6B and Llama2-7B. |
| **Table 6**, refined granularity | [`experiments/granularity_ablation/run.sh`](experiments/granularity_ablation/README.md) | 16-to-{8,4,2} for both Qwen3-0.6B and Llama2-7B. |
| **Sec. 6**, accelerator dataflow | [`scripts/run_hardware_all.sh`](hardware/README.md) | Functional SpinalHDL simulation of metadata, MultiMSA, shared-FPU, synchronization, and packet paths. |
| AE execution validation | [`experiments/offload_equivalence/run.sh`](experiments/offload_equivalence/README.md) | Confirms that full-GPU and layer-streamed PPL produce the same BF16/MBPriorQ NLL. |

The exact claim boundary is in [`docs/AE_SCOPE.md`](docs/AE_SCOPE.md). Hardware
simulation does **not** claim to reproduce area, power, or comparison-accelerator
speedup.

## Ten-Minute Start

1. Create the environment and install the local package:

   ```bash
   conda env create -f environment/software.yml
   conda activate mbpriorq-ae
   python -m pip install -e software
   ```

2. Run tests that require no model download:

   ```bash
   ./scripts/run_smoke.sh
   python -m pytest -q software/tests
   ```

3. Download `Qwen/Qwen3-0.6B` and WikiText2, then run a four-window GPU check:

   ```bash
   MODEL_PATH=/path/to/Qwen3-0.6B \
   DATASET_PATH=/path/to/wikitext-2-raw-v1 \
   ./experiments/qwen3_0_6b_smoke_test/run_quick.sh
   ```

4. Validate that offloading does not change the result:

   ```bash
   MODEL_PATH=/path/to/Qwen3-0.6B \
   DATASET_PATH=/path/to/wikitext-2-raw-v1 \
   ./experiments/offload_equivalence/run_quick.sh
   ```

Generated checkpoints, logs, and result JSONs go to ignored `local_runs/`.

## Required Models And Imatrix Rule

The artifact never redistributes model weights. Model identifiers and local
directory aliases are listed in
[`experiments/table2_ppl/models.json`](experiments/table2_ppl/models.json).

**Only Qwen3-0.6B uses an imatrix.** Its weight quantization requires:

```text
https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B.imatrix
```

The checked file is already included at
[`data/imatrix/Qwen_Qwen3-0.6B.imatrix`](data/imatrix/README.md). Table 2 and
checkpoint scripts reject both a missing Qwen3-0.6B imatrix and an imatrix
supplied to any other model. Llama, the remaining Qwen models, DeepSeek, and
Mixtral do **not** consume this file.

## Complete Table 2

Place model directories below one or more roots. The runner recursively finds
the aliases in `models.json`, validates every path before starting, updates
`observation.md` after each row, and resumes completed JSON results.

```bash
MODEL_ROOTS=/models/huggingface:/models/modelscope \
./experiments/table2_ppl/run.sh \
  --dataset /path/to/wikitext-2-raw-v1
```

This single command produces the 38 BF16/MBPriorQ rows in the current Table 2.
Qwen3-0.6B uses the true full-GPU path. Other entries use safetensors
layer streaming so the complete model need not fit VRAM. Qwen3-VL explicitly
instantiates only its language model because Table 2 evaluates text-only
WikiText2. See the Table 2 README for explicit path overrides and
resource notes. A full run validates each observed PPL against the rounded
paper value; `--num-samples 1` is only a backend smoke and skips that numerical
claim.

## Two-Model Ablations

Both central ablation drivers require Qwen3-0.6B and Llama2-7B, matching the
paper tables:

```bash
QWEN_MODEL_PATH=/path/to/Qwen3-0.6B \
LLAMA_MODEL_PATH=/path/to/Llama-2-7b-hf \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/activation_attribution/run.sh

QWEN_MODEL_PATH=/path/to/Qwen3-0.6B \
LLAMA_MODEL_PATH=/path/to/Llama-2-7b-hf \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/granularity_ablation/run.sh
```

Llama2 access may require accepting the model license. Llama rows use the
streamed backend and never use the Qwen imatrix.

## Downstream Benchmarks

Table 3 uses the paper protocols: 500 seeded GSM8K test examples, 100 seeded
MMLU examples, and all 410 MMLU-Pro computer-science examples with five-shot
CoT prompts. Set the two models and three local dataset directories:

```bash
QWEN_0_6B_MODEL_PATH=/path/to/Qwen3-0.6B \
QWEN_14B_MODEL_PATH=/path/to/Qwen3-14B \
GSM8K_DATASET_PATH=/path/to/gsm8k-kaggle-directory \
MMLU_DATASET_PATH=/path/to/mmlu-kaggle-directory \
MMLU_PRO_DATASET_PATH=/path/to/MMLU-Pro \
./experiments/downstream_benchmarks/run.sh
```

`run_quick.sh` defaults to one example from each benchmark on Qwen3-0.6B.
The full workflow is long because the paper permits up to 2048 generated
tokens per question and evaluates both BF16 and MBPriorQ. On resume, BF16 rows
are skipped directly; MBPriorQ replays completed prompts before continuing so
the online activation-prior state is identical to an uninterrupted run.

## Hardware Functional Validation

Create the separate hardware environment described in
[`environment/README.md`](environment/README.md), then run:

```bash
./scripts/run_hardware_modules.sh
./scripts/run_hardware_system.sh
# or both:
./scripts/run_hardware_all.sh
```

These tests elaborate the actual SpinalHDL and compare functional CSV traces
with [`evidence/hardware/`](evidence/hardware/README.md).

## Resource Guide

| Workflow | Practical requirement |
|---|---|
| Python tests and tensor smoke | Linux CPU, about 8 GB RAM |
| Qwen3-0.6B full-GPU quick/full PPL | CUDA GPU with at least 8 GB free VRAM |
| Llama2-7B streamed PPL | GPU large enough for one decoder layer, plus CPU RAM for hidden-state windows |
| Complete 19-model Table 2 | All public checkpoints, substantial SSD capacity, and long runtime; only one decoder layer is resident on GPU, while hidden windows and weight-quantization temporaries use CPU RAM |
| Qwen3-VL language-side row | About 439 GB source checkpoint storage on the tested setup; BF16 one-window streamed validation took 457 s, while a full MBPriorQ tensor-level quantization run is a day-scale experiment |
| Qwen3-14B downstream | `device_map=auto`; about 30 GB checkpoint storage plus generated MBPriorQ checkpoint |
| SpinalHDL simulation | Java 17, sbt 1.10.2, Verilator, GNU Make, C++ compiler |

The validated host used an RTX 5090 (32 GiB), 186 GiB usable RAM, and Linux
x86-64. This is the tested configuration, not a hidden requirement.

## Repository Guide

- [`docs/AE_SCOPE.md`](docs/AE_SCOPE.md): paper claim and evidence map.
- [`docs/VALIDATION.md`](docs/VALIDATION.md): what has actually been run.
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md): source and license boundaries.
- [`experiments/`](experiments/README.md): all stable experiment entry points.
- [`software/`](software/README.md): curated quantization/runtime modules.
- [`hardware/`](hardware/README.md): SpinalHDL source and simulations.
- [`artifact_appendix/`](artifact_appendix/README.md): appendix source and the
  paper-plus-appendix build workflow.
- [`MBPriorQ_AE_Submission.pdf`](MBPriorQ_AE_Submission.pdf): compiled paper
  followed by the Artifact Appendix, ready for the HotCRP Submission field.

Before publication, maintainers run `python scripts/check_release.py --strict`.
The Zenodo archive must not contain checkpoints, datasets, credentials,
workstation paths, third-party method reproductions, or commercial EDA data.
