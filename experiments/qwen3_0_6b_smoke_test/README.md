# Qwen3-0.6B Lightweight Smoke Test

This is the shortest model-level path from a public BF16 checkpoint to the
paper's BF16 and W4A4 MBPriorQ WikiText2 results. It is the recommended first
GPU experiment because Qwen3-0.6B fits entirely in modest VRAM and still
exercises weight quantization, the online activation prior, 16-to-4 refinement,
`lm_head`, EBW accounting, and result validation.

The workflow first creates a local fake-quantized Hugging Face checkpoint and
then evaluates BF16 and MBPriorQ. Only this model consumes the bundled
Qwen3-0.6B imatrix.

## Complete Lightweight Reproduction

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/qwen3_0_6b_smoke_test/run.sh
```

This processes all 146 contiguous 2048-token windows and validates:

| Method | Expected PPL | Tolerance |
|---|---:|---:|
| BF16 | 20.9240 | 0.0500 |
| W4A4 MBPriorQ | 24.2289 | 0.0750 |

## Quick Execution Check

For a four-window check that does not claim the paper value:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/qwen3_0_6b_smoke_test/run_quick.sh
```

Generated checkpoints and results remain under ignored `local_runs/`. On the
validated RTX 5090, a clean complete run should be budgeted at approximately
10--20 minutes; cached checkpoints reduce subsequent runtime.

## Troubleshooting

- `MODEL_PATH` must point to the public Qwen3-0.6B Hugging Face checkpoint.
- Set `DATASET_PATH` to a local WikiText2 directory if network dataset access
  is unavailable.
- Keep at least 8 GB CUDA VRAM free for this full-GPU smoke path.
- Delete `local_runs/checkpoints/qwen3_0_6b_mbpriorq_rb4` to rebuild a stale
  generated checkpoint.
