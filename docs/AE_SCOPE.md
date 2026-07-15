# Artifact Scope And Claim Map

## Evidence Policy

The artifact distinguishes full result reproduction from bounded functional
validation. Every covered claim has one of two evidence paths:

1. **Reproduced**: an executable workflow regenerates and validates the result.
2. **Functional**: a bounded workflow validates the same mechanism and output
   contract at reduced cost.

Paper results outside the representative badge workflow are not implicitly
claimed merely because they appear in the paper. In particular, this artifact
does not include reproductions of comparison methods or commercial synthesis
data.

## Claim Map

| ID | Paper location | Result | Public AE path | Coverage |
|---|---|---|---|---|
| KR1 | Table 2, Sec. 7.2 | W4A4 accuracy across model families | [`experiments/core_accuracy/`](../experiments/core_accuracy/README.md) | **Reproduced** for Qwen3-0.6B; the complete 19-entry sweep is outside the default rerun. |
| KR2 | Tables 4-7, Secs. 7.3-7.4 | Attribution and refined-granularity ablations | [`experiments/activation_attribution/`](../experiments/activation_attribution/README.md), [`experiments/granularity_ablation/`](../experiments/granularity_ablation/README.md) | **Reproduced** for Qwen3-0.6B; the remaining model rows are outside the AE package. |
| KR3 | Tables 9-10, Sec. 7.5 | Calibration and side-metadata overhead | [`scripts/run_smoke.py`](../scripts/run_smoke.py), [`software/mbpriorq_ae/ebw.py`](../software/mbpriorq_ae/ebw.py) | Metadata arithmetic is **functional/reproduced**; the full-model calibration table is not rerun by default. |
| KR4 | Figs. 10-11 and Table 11, Sec. 7.6 | Hardware speedup and quantitative latency results | Paper results | Outside the AE rerun; no comparison-method reproduction is included. |
| KR5 | Sec. 6 | Packet parsing, mask-aware MultiMSA/VPU execution, joining, and packetization | [`scripts/run_hardware_modules.sh`](../scripts/run_hardware_modules.sh), [`scripts/run_hardware_system.sh`](../scripts/run_hardware_system.sh) | **Functional** SpinalHDL/Verilator validation. |
| KR6 | Tables 3 and 8, Secs. 7.2-7.4 | Downstream-task and KV-cache extensions | Paper results | Outside the default AE rerun; no fresh reproduction is claimed here. |

## Functional Workflow

The Functional workflow finishes without downloading a model:

- validate deterministic MBPriorQ regular/refined tensor operations;
- exercise regular and refined 16-value micro-block selection;
- verify that each independently asserted tensor-side mask adds exactly three
  FP8 scales;
- verify effective-bit-width formulas and known fixtures;
- simulate the actual scale reconstructor, packet scheduler, FPU pool,
  MultiMSA, output join, and public packet top;
- compare generated outputs with checksummed expected files.

Entry points are `scripts/run_smoke.sh`, `scripts/run_hardware_modules.sh`, and
`scripts/run_hardware_system.sh`.

## Results Reproduced Workflow

The default Results Reproduced target is a representative public model rather
than all 19 paper entries:

- `Qwen/Qwen3-0.6B` and WikiText2 raw test;
- contiguous 2048-token windows, all 146 available windows;
- BF16 and paper-spec W4A4 MBPriorQ;
- activation-prior attribution and 16-to-{8,4,2} granularity rows;
- parser-based comparison against recorded expected PPL and EBW values.

Llama2-7B is not a required rerun because checkpoint access can be gated. The
remaining Table 2 models require substantially more storage/runtime and are not
part of the default badge workflow.

## Hardware Reproduction Boundary

Open SpinalHDL/Verilator simulation establishes MBPriorQ mechanism and protocol
functionality: independent tensor-side scale selection, metadata-first VPU
issue, pooled FPU scheduling, regular/refined MultiMSA execution, packet
parsing, output joining, and packet segmentation. Cycle numbers in the bounded
module fixtures are regression observables, not paper-level performance claims.
The AE package intentionally excludes all commercial synthesis material and all
implementations or validators for comparison accelerators. Hardware speedup,
area, and power remain paper results and are not an AE badge target.

## Badge Target

The intended submission requests Artifact Available, Artifact Evaluated -
Functional, and Results Reproduced. These are targets until the approved public
license, tagged clean-room validation, Zenodo archive, and DOI are complete.
