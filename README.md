# MBPriorQ Artifact Evaluation

This repository contains the curated artifact for:

> Software-Hardware Co-Design of Prior-Aware W4A4 Micro-Block Quantization
> for Robust LLM Inference Across Model Families

It packages the paper-relevant MBPriorQ software, curated SpinalHDL accelerator
sources, functional simulations, and stable experiment entry points. It does not include unrelated EasyLLM
features, model weights, commercial EDA material, or reproductions of
comparison methods.

## Status And Evidence Levels

The source closure and representative workflows have been assembled and tested.
The remaining public-release blockers are the author-approved source license,
final citation metadata, release tag, and Zenodo DOI. See
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).

Results are deliberately separated into two evidence levels:

| Level | Meaning |
|---|---|
| **Reproduced** | A public workflow regenerates and validates the result. |
| **Functional** | A bounded workflow validates the same mechanism and output contract. |

The core Results Reproduced target is Qwen3-0.6B on WikiText2. Open hardware
simulation establishes the functionality of MBPriorQ's scale path, scheduler,
MultiMSA, shared FPU pool, output synchronization, and public packet interface.
It is not presented as a reproduction of paper-level speedup, area, or power.

## Repository Map

- [`docs/AE_SCOPE.md`](docs/AE_SCOPE.md): claim-to-evidence map and exact
  reproduction boundary.
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md): pinned source revisions,
  adaptations, and third-party boundaries.
- [`docs/VALIDATION.md`](docs/VALIDATION.md): completed software/hardware
  validation and remaining release gates.
- [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md): publication and
  clean-room validation gates.
- [`environment/`](environment/README.md): software and hardware environments.
- [`software/`](software/README.md): curated MBPriorQ implementation.
- [`experiments/`](experiments/README.md): stable PPL and ablation entry points.
- [`hardware/`](hardware/README.md): curated SpinalHDL source and functional simulations.
- [`evidence/`](evidence/README.md): expected outputs from the public workflows.
- [`artifact_appendix/`](artifact_appendix/README.md): two-page AE appendix.

## Requirements

The functional software checks need a Linux x86-64 host with at least four CPU
cores and approximately 8 GB RAM. The core PPL reproduction needs an NVIDIA GPU
with at least 8 GB VRAM, 16 GB system RAM, and about 10 GB free disk space for
the public model, dataset cache, and generated fake-quantized checkpoints.

The open hardware workflow needs Java 17, sbt 1.10.2, Verilator 4.034 or newer,
GNU Make, and a C++ compiler. The validated machine used an AMD Ryzen 7 9800X3D
(8 cores/16 threads), 186 GiB RAM, and an NVIDIA RTX 5090 with 32 GiB VRAM.
This is the tested configuration, not the minimum requirement.

## Installation

Create the software environment:

```bash
conda env create -f environment/software.yml
conda activate mbpriorq-ae
python -m pip install -e software
```

For open hardware simulation, follow
[`environment/README.md`](environment/README.md) to create the separate
`mbpriorq-ae-hw` environment and install sbt.

## Functional Checks

Run the deterministic tensor/metadata smoke and Python unit tests:

```bash
./scripts/run_smoke.sh
python -m unittest discover -s software/tests -v
```

Run the source-equivalence check when the pinned EasyLLM source tree is locally
available:

```bash
python scripts/compare_software_source.py --easyllm /path/to/EasyLLM
```

The smoke completes in under a minute on a typical desktop. It checks regular
and refined 16-value micro-block behavior, independent weight/activation masks,
the three-extra-scale rule, and effective-bit-width accounting.

## Core Accuracy Reproduction

Download the public `Qwen/Qwen3-0.6B` checkpoint through Hugging Face and set
`MODEL_PATH` to its local directory. `DATASET_PATH` can be a local
`wikitext-2-raw-v1` directory; omitting it lets Hugging Face Datasets download
the public WikiText2 data.

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
./experiments/core_accuracy/run.sh
```

The complete run evaluates all 146 contiguous 2048-token windows and validates:

| Method | Expected PPL |
|---|---:|
| BF16 | 20.9240 |
| W4A4 MBPriorQ | 24.2289 |

For an execution-path check using only two windows:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
./experiments/core_accuracy/run_quick.sh
```

Generated checkpoints and results are written below ignored `local_runs/` and
are never included in the release archive. On the validated RTX 5090, cached
full PPL rows take minutes; a clean run also creates one approximately 1.2 GB
fake-quantized checkpoint.

## Central Ablations

The following workflows share the checkpoint cache created above:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
./experiments/activation_attribution/run.sh

MODEL_PATH=/path/to/Qwen3-0.6B \
./experiments/granularity_ablation/run.sh
```

They reproduce the Qwen3-0.6B rows for activation-prior attribution and
16-to-{8,4,2} refinement. Each script validates PPL and, where applicable,
weight/activation effective bit width against its local `expected.csv`. A clean
combined run is expected to take tens of minutes and use up to roughly 5 GB for
shared generated checkpoints on the tested machine.

## Open Hardware Functional Validation

After activating the hardware environment, run:

```bash
./scripts/run_hardware_modules.sh
./scripts/run_hardware_system.sh
```

The first command validates five actual SpinalHDL modules: scale reconstruction,
metadata-before-matrix scheduling, the shared FPU pool, regular/refined MultiMSA
execution, and matrix/dequant-factor joining. The second elaborates the public
16-lane accelerator top and verifies a mixed regular/refined 1024-bit packet
stream through output packetization. On the validated host, a warm module run
took about 30 seconds and the final system build took 8 minutes 18 seconds.
Generated CSVs are compared with
[`evidence/hardware/`](evidence/hardware/README.md). Run both with:

```bash
./scripts/run_hardware_all.sh
```

## Release Audit

Run the non-release audit at any time:

```bash
python scripts/check_release.py
```

The strict audit additionally requires the approved license, final
`CITATION.cff`, final `.zenodo.json`, appendix, and release checksum manifest:

```bash
python scripts/check_release.py --strict
```

The Zenodo archive must be generated from the evaluated Git tag. It must not
contain model weights, access tokens, workstation paths, proprietary PDK/library
files, commercial EDA data, or reproductions of comparison methods.
