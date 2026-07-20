# MBPriorQ Software

## Setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate mbpriorq-ae
```

## Components

| Path | Responsibility |
|---|---|
| `mbpriorq_ae/mbpriorq.py` | W4A4 quantization, VMB selection, online prior, and selective refinement |
| `mbpriorq_ae/integration.py` | Hugging Face Linear and stacked-expert integration |
| `mbpriorq_ae/checkpoint.py` | Bounded-memory fake-quantized checkpoint generation |
| `mbpriorq_ae/perplexity.py` | Paper-compatible PPL and KV-cache evaluation |
| `mbpriorq_ae/offload.py` | Layer-streamed execution for large checkpoints and Qwen3-VL's language path |
| `mbpriorq_ae/kv_cache.py` | BF16, NVFP4, and MBPriorQ KV-cache paths |
| `mbpriorq_ae/ebw.py` | Mask, scale, and effective-bit-width accounting |
| `tools/` | Shared command-line drivers used by the paper experiments |

The streamed backend keeps hidden states in CPU memory and loads one decoder
layer at a time, allowing model-family PPL evaluation when the complete
checkpoint does not fit GPU memory.

## Tests

```bash
python -m pytest -q software/tests
```

The tests cover FP4 arithmetic, regular/refined scale selection, online-prior
state, checkpoint integration, streamed loading, KV-cache quantization, true
batch execution, final partial PPL chunks, and deterministic EBW accounting.
