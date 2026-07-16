# Validation Record

This document is the validation ledger for the curated MICRO 2026 artifact.
It records completed checks, their evidence boundary, and the remaining public
release gates. Generated build products and model checkpoints remain under the
ignored `local_runs/` directory and are not part of the release archive.

## Validated Host And Toolchain

The release candidate was validated on 2026-07-15 and rechecked on 2026-07-16
with:

- AMD Ryzen 7 9800X3D, 8 cores/16 threads;
- 186 GiB usable system memory;
- NVIDIA GeForce RTX 5090, 32 GiB VRAM, driver 575.57.08;
- Linux 6.8 x86-64;
- Python 3.12.12, PyTorch 2.10.0+cu129, Transformers 4.57.6, and
  Datasets 4.7.0;
- OpenJDK 17.0.18, sbt 1.10.2, Scala 2.12.19, SpinalHDL 1.8, and
  Verilator 4.034;
- Latexmk 4.83.

The hardware project pins the effective sbt version through
`hardware/spinal/project/build.properties`; the installed launcher version is
not part of the experiment contract.

## Software Validation

| Check | Result | Evidence |
|---|---|---|
| Curated-source equivalence against pinned EasyLLM `MICRO` commit | Passed for every imported MBPriorQ prior mode and refined block size | `scripts/compare_software_source.py`, `docs/PROVENANCE.md` |
| Python unit tests | 14/14 passed, including one-token decode prior refinement and strict streamed-checkpoint loading | `software/tests/` |
| Deterministic tensor and EBW smoke | Passed | `scripts/run_smoke.sh` |
| Qwen3-0.6B lightweight smoke | Passed over all 146 contiguous 2048-token WikiText2 windows | `evidence/ppl/local_validation.json` |
| Full-GPU/streamed equivalence | Exact one-window BF16 and MBPriorQ NLL/PPL match | `experiments/offload_equivalence/` |
| Online/prequant weight-path equivalence | Qwen3-0.6B online paper-semantics quantization reproduced the prequant one-window MBPriorQ PPL `16.890377` and NLL `5789.171875` exactly | `scripts/run_wikitext_ppl.py` |
| Table 2 preflight | All 19 model paths resolved | `experiments/table2_ppl/models.json` |
| Streamed structure coverage | All 19 model skeletons and checkpoint prefixes were rechecked successfully, including Qwen3-VL language-only | `scripts/check_offload_structure.py` |
| Real streamed PPL | Llama2-7B BF16 completed one window; Qwen3-VL-235B language-side BF16 completed 94 layers with PPL 2.937617 | `local_runs/backend_validation/` (generated, not distributed) |
| Downstream smoke | Real Qwen3-0.6B BF16 and MBPriorQ generation completed on one GSM8K, MMLU, and MMLU-Pro example; MBPriorQ resume replay also passed | `scripts/run_downstream_benchmark.py` |
| Activation attribution | Public driver and expected matrix cover 12 Qwen3-0.6B/Llama2-7B rows; prior Qwen full rows are retained as evidence | `experiments/activation_attribution/expected.csv` |
| Refined granularity | Public driver and expected matrix cover all six Qwen3-0.6B/Llama2-7B rows; prior Qwen full rows are retained as evidence | `experiments/granularity_ablation/expected.csv` |

The complete lightweight smoke result is:

| Method | PPL | Weight EBW | Activation EBW |
|---|---:|---:|---:|
| BF16 | 20.923949 | -- | -- |
| W4A4 MBPriorQ | 24.228947 | 4.673069 | 4.994211 |

These checkpoints are fake-quantized BF16/FP16-style Hugging Face checkpoints,
not physically packed FP4 files. They are generated locally and excluded from
the artifact archive.

The one-window backend equivalence values were:

| Method | Full-GPU PPL / NLL | Streamed PPL / NLL |
|---|---:|---:|
| BF16 | 14.224318 / 5437.343750 | 14.224318 / 5437.343750 |
| MBPriorQ | 16.890377 / 5789.171875 | 16.890377 / 5789.171875 |

These one-window values are execution validation, not substitutes for the
146-window paper result.

The Qwen3-VL execution check used the 439 GB local source checkpoint, kept only
one of 94 language layers on the RTX 5090, and completed in 456.68 seconds
without a non-finite value or an out-of-memory failure. Its one-window PPL is a
backend check, not the full Table 2 row. The complete MBPriorQ Qwen3-VL path
retains paper-compatible tensor-level weight quantization and is intentionally
documented as a day-scale complete workflow rather than a quick check.

## Open Hardware Validation

The following functional workflows completed from the curated repository and
matched the recorded CSVs exactly:

| Workflow | Result |
|---|---|
| Four regular/refined tensor-side scale combinations | Passed |
| Metadata-first VPU issue before MultiMSA data readiness | Passed |
| Shared FPU-pool allocation and completion | Passed |
| Regular/refined MultiMSA output contract | Passed |
| Matrix/dequant-factor output join | Passed |
| Public 1024-bit packet input/output path | Passed |

The module workflow took about 30 seconds with a warm dependency/build cache.
The final public-top regression completed in 8 minutes 18 seconds, including
7 minutes 56 seconds of Verilator compilation. The checked outputs are under
`evidence/hardware/`;
the entry points are `scripts/run_hardware_modules.sh` and
`scripts/run_hardware_system.sh`. These tests establish functionality and do
not claim to reproduce hardware speedup, area, or power.

## Artifact Boundary Audit

The release candidate contains only MBPriorQ's author-generated software,
hardware, inputs, and expected outputs, plus the Qwen model license needed to
identify the external model dependency. Implementations and result validators
for comparison methods are excluded. Commercial EDA inputs, outputs,
screenshots, and area/power data are also excluded.

## Documentation And Packaging Validation

- The standalone artifact appendix compiles in ACM two-column format as one
  page, below the two-page limit, with no overfull lines.
- `MBPriorQ_AE_Submission.pdf` contains the unchanged 15-page paper followed by
  the one-page Artifact Appendix, with continuous page numbering.
- The non-strict public-release audit passes.
- `metadata/release_files.sha256` verifies every distributable file but must be
  regenerated after any source or documentation edit.
- A `git archive` extraction without repository metadata passed the software
  smoke and the then-current unit tests, all five hardware-module comparisons, the public
  packet-top comparison, the appendix build, the release audit, and all 103
  release checksums. This confirms that the candidate does not rely on
  untracked source files or its original workstation path.
- The final tagged archive must be checked again after resolving release
  metadata. This final pass is a release gate, not an unresolved executable
  artifact issue.

## Remaining Release Gates

The executable artifact is validated. Public release is still blocked on:

1. creation of the final public repository and `v1.0.0` release tag;
2. publication of the Zenodo record and assignment of its version DOI;
3. DOI insertion into the artifact appendix, followed by strict audit and
   clean-tag validation.
