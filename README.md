# MBPriorQ Artifact Evaluation

This repository contains the software and hardware artifact for:

> **Software-Hardware Co-Design of Prior-Aware W4A4 Micro-Block Quantization
> for Robust LLM Inference Across Model Families**

## Repository Structure

- [`software/`](software/README.md): MBPriorQ quantization, low-memory execution,
  shared experiment tools, and unit tests.
- [`hardware/`](hardware/README.md): SpinalHDL accelerator sources and functional
  verification.
- [`experiments/`](experiments/README.md): paper-result workflows organized by
  table number.
- [`data/imatrix/`](data/imatrix/README.md): the Qwen3-0.6B importance matrix used
  for weight quantization.

## Setup

```bash
conda env create -f environment.yml
conda activate mbpriorq-ae
python -m pytest -q software/tests
```

## Lightweight Reproduction

Qwen3-0.6B provides the shortest complete path from a public BF16 checkpoint to
the paper's BF16 and W4A4 MBPriorQ WikiText2 results:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/smoke_test/run.sh
```

The full workflow evaluates all 146 contiguous 2048-token windows. Use
`run_quick.sh` for a four-window execution check. Generated checkpoints, logs,
and results are written below the ignored `local_runs/` directory.

## Artifact Validation

Run syntax checks and dependency-light tests with:

```bash
./validate.sh
```

`./validate.sh software-quick` exercises every software workflow with reduced
inputs. It requires the two model paths and five dataset paths documented by
the experiment READMEs. The three downstream benchmarks use one example each.
Run `./validate.sh hardware` for the complete SpinalHDL functional regression.

## Paper Reproduction

| Paper result | Entry point | Coverage |
|---|---|---|
| Tables 2 and 10 | [`experiments/table2/`](experiments/table2/README.md) | BF16/MBPriorQ PPL and side-metadata EBW for 19 model entries |
| Table 3 | [`experiments/table3/`](experiments/table3/README.md) | GSM8K, MMLU, and MMLU-Pro |
| Table 4 | [`experiments/table4/`](experiments/table4/README.md) | Feature-selection PPL on Qwen3-0.6B and Llama2-7B |
| Table 5 | [`experiments/table5/`](experiments/table5/README.md) | Activation-prior accuracy attribution on both models |
| Table 6 | [`experiments/table6/`](experiments/table6/README.md) | 16-to-{8,4,2} refined-granularity ablation on both models |
| Table 7 | [`experiments/table7/`](experiments/table7/README.md) | VMB-prior robustness under five input-variation axes |
| Table 8 | [`experiments/table8/`](experiments/table8/README.md) | BF16, NVFP4, and MBPriorQ KV-cache quantization |

Each directory provides a complete `run.sh`, a reduced `run_quick.sh` where
useful, the paper reference values, required inputs, and generated outputs.

## Hardware Verification

```bash
conda activate mbpriorq-ae
./hardware/run_all.sh
```

The hardware workflow checks metadata and scale handling, regular/refined
MultiMSA execution, shared FPU allocation, matrix-scale synchronization, and the
complete 1024-bit packet input/output path.

## Models And Data

Model identifiers and local aliases for the complete model-family study are in
[`experiments/table2/models.json`](experiments/table2/models.json). WikiText2 is
used for PPL, PTB and MMLU-Pro provide Table 7 input variations, and GSM8K,
MMLU, and MMLU-Pro provide downstream tasks.

Only Qwen3-0.6B weight quantization consumes the bundled imatrix. All model
checkpoints and datasets are supplied locally through the paths documented by
each experiment.

For models that do not fit entirely in GPU memory, the streamed backend keeps
the hidden states in CPU memory and transfers one decoder layer to the GPU at a
time. This reduces peak GPU-memory requirements and enables the large-model PPL
workflows on memory-constrained GPUs.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The artifact is
released under the Apache License 2.0.
