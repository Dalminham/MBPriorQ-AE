# MBPriorQ Artifact Evaluation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21505490.svg)](https://doi.org/10.5281/zenodo.21505490)

This repository contains the software and hardware artifacts for:

> **Software-Hardware Co-Design of Prior-Aware W4A4 Micro-Block Quantization
> for Robust LLM Inference Across Model Families**

The software reproduces Tables 2, 3, 5-8, and 10. The hardware artifact
provides functional verification of the MBPriorQ accelerator.

## Repository Structure

```text
MBPriorQ-AE/
├── software/          MBPriorQ quantization and evaluation implementation
├── experiments/       software workflows organized by paper table
├── hardware/          SpinalHDL sources and functional simulations
├── data/imatrix/      Qwen3-0.6B importance matrix
├── environment.yml    common software and hardware environment
└── validate.sh        repository and workflow validation entry point
```

- [`software/`](software/README.md) contains W4A4 quantization, VMB selection,
  online activation-prior refinement, layer-streamed execution, KV-cache
  quantization, EBW accounting, and shared experiment drivers.
- [`experiments/`](experiments/README.md) contains the commands, reference
  values, and validators for each reproduced paper table.
- [`hardware/`](hardware/README.md) contains the accelerator RTL, module-level
  testbenches, complete-system testbench, and checked-in expected results.
- [`data/imatrix/`](data/imatrix/README.md) contains the importance matrix used
  automatically by Qwen3-0.6B weight quantization.

All generated checkpoints, logs, CSVs, and JSON summaries are written below the
ignored `local_runs/` directory.

## Software Reproduction

Create the common environment from the repository root:

```bash
conda env create -f environment.yml
conda activate mbpriorq-ae
python -m pytest -q software/tests
```

### Lightweight Reproduction

The shortest complete reproduction evaluates BF16 and W4A4 MBPriorQ on all 146
contiguous 2048-token WikiText2 windows with Qwen3-0.6B:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/smoke_test/run.sh
```

The expected WikiText2 PPL values are `20.9240` for BF16 and `24.2289` for
W4A4 MBPriorQ. A successful run ends with a validation PASS line.

For a short execution-path check, use:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/smoke_test/run_quick.sh
```

The layer-streamed backend reduces GPU-memory demand by keeping hidden states
in CPU memory and loading one decoder layer onto the GPU at a time. The
following workflow compares full-GPU and layer-streamed execution on the same
Qwen3-0.6B checkpoints:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/smoke_test/run_offload_equivalence.sh
```

### Paper Results

| Paper result | Reproduction workflow |
|---|---|
| Tables 2 and 10: model-family PPL and side-metadata EBW | [`experiments/table2/`](experiments/table2/README.md) |
| Table 3: GSM8K, MMLU, and MMLU-Pro accuracy | [`experiments/table3/`](experiments/table3/README.md) |
| Table 5: activation-prior accuracy attribution | [`experiments/table5/`](experiments/table5/README.md) |
| Table 6: refined-granularity ablation | [`experiments/table6/`](experiments/table6/README.md) |
| Table 7: VMB-prior robustness under input variation | [`experiments/table7/`](experiments/table7/README.md) |
| Table 8: KV-cache quantization | [`experiments/table8/`](experiments/table8/README.md) |

Each directory provides `run.sh` for the paper protocol and an `expected.csv`
or equivalent reference file for result validation. A `run_quick.sh` entry is
also provided where a reduced execution-path check is useful. Tables 2 and 10
share one execution: every MBPriorQ PPL result is emitted together with its
mask, scale, and total EBW.

Run repository checks with:

```bash
./validate.sh static unit
```

## Hardware Reproduction

Activate the same environment and run:

```bash
conda activate mbpriorq-ae

./hardware/run_modules.sh   # five module-level simulations
./hardware/run_system.sh    # complete 16-lane 1024-bit packet path
./hardware/run_all.sh       # both workflows
```

The module workflow verifies the scale reconstructor, packet scheduler, shared
FPU pool, regular/refined MultiMSA paths, and output-pair join. The system
workflow drives the complete packet interface with independent weight and
activation masks, reordered metadata, mixed regular/refined blocks, and output
backpressure.

Each testbench computes or declares its expected functional result before
writing a CSV. [`hardware/validate_results.py`](hardware/validate_results.py)
then checks the generated CSV semantically and compares it with the
corresponding file under [`hardware/expected/`](hardware/expected/).

Generated results are written to:

```text
local_runs/hardware_modules/
local_runs/hardware_system/
```

A successful complete run ends with:

```text
Hardware functional and golden-result validation passed.
```
