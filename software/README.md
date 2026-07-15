# Software Artifact

`software/mbpriorq_ae/` is the minimal executable MBPriorQ closure used by the
public experiments. Install it from the repository root with:

```bash
python -m pip install -e software
```

## Components

| Module | Responsibility |
|---|---|
| `mbpriorq.py` | W4A4 fake quantization, VMB selection, online prior, and 16-to-{8,4,2} refinement. |
| `integration.py` | Hugging Face Linear and Qwen3-VL stacked-expert activation wrappers. |
| `checkpoint.py` | Bounded-memory generation of loadable fake-quantized checkpoints, including `lm_head`. |
| `perplexity.py` | Paper-compatible contiguous-window PPL with a true full-GPU backend. |
| `offload.py` | Correctness-oriented safetensors layer streaming for standard decoder models and Qwen3-VL's language path. |
| `ebw.py` | Formal mask/scale/effective-bit-width accounting. |

The streamed backend is included so large paper models do not need to fit in
VRAM. It keeps hidden states in CPU DRAM and loads one decoder layer from the
checkpoint at a time. It raises on NaN/Inf and does not silently alter outputs.
It is an AE execution mechanism, not a claim of serving-system performance.

Run all dependency-light tests with:

```bash
PYTHONPATH=software python -m pytest -q software/tests
```

`scripts/compare_software_source.py` is a maintainer provenance check against
the pinned private development revision. Evaluators do not need that source
tree.
