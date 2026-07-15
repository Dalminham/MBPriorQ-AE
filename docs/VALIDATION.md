# Validation Record

This document is the validation ledger for the curated MICRO 2026 artifact.
It records completed checks, their evidence boundary, and the remaining public
release gates. Generated build products and model checkpoints remain under the
ignored `local_runs/` directory and are not part of the release archive.

## Validated Host And Toolchain

The release candidate was validated on 2026-07-15 with:

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
| Python unit tests | 10/10 passed | `software/tests/` |
| Deterministic tensor and EBW smoke | Passed | `scripts/run_smoke.sh` |
| Qwen3-0.6B core accuracy | Passed over all 146 contiguous 2048-token WikiText2 windows | `evidence/ppl/local_validation.json` |
| Activation attribution | All 6 expected rows passed | `experiments/activation_attribution/expected.csv` |
| Refined granularity | All 16-to-8/4/2 rows passed | `experiments/granularity_ablation/expected.csv` |

The representative full accuracy result is:

| Method | PPL | Weight EBW | Activation EBW |
|---|---:|---:|---:|
| BF16 | 20.923949 | -- | -- |
| W4A4 MBPriorQ | 24.228947 | 4.673069 | 4.994211 |

These checkpoints are fake-quantized BF16/FP16-style Hugging Face checkpoints,
not physically packed FP4 files. They are generated locally and excluded from
the artifact archive.

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
- The non-strict public-release audit passes.
- `metadata/release_files.sha256` verifies every distributable file but must be
  regenerated after any source or documentation edit.
- A `git archive` extraction without repository metadata passed the software
  smoke and all 10 unit tests, all five hardware-module comparisons, the public
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
