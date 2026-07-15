# Software Artifact

This directory contains the minimal executable closure for MBPriorQ, VMB
profiling, EBW accounting, streamed checkpoint generation, and the
paper-compatible WikiText2 PPL path.

It does not import every EasyQuant baseline at module-import time. The curated
entry point exposes only algorithms required by the MBPriorQ paper artifact.

Install the package in an isolated environment:

```bash
python -m pip install -e software
```

Run the dependency-light tests from the repository root:

```bash
PYTHONPATH=software python -m unittest discover -s software/tests -v
```

`scripts/compare_software_source.py` is a maintainer check. It compares all
curated quantization paths against the pinned EasyLLM `MICRO` revision and is
not required by artifact evaluators.

The optional development calibration-mask plot was replaced by a no-op hook
because it wrote to a workstation-specific path and does not affect any paper
result. Quantization outputs are checked element-for-element against the source
revision after this import-only change.

The model-level integration is split by responsibility:

- `checkpoint.py` generates bounded-memory fake-quantized HF checkpoints;
- `integration.py` wraps only paper-scope Linear activations;
- `perplexity.py` implements contiguous 2048-token, layer-wise evaluation.
