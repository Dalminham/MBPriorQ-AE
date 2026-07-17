# Qwen3-0.6B Smoke Test

This is the shortest model-level path from a public BF16 checkpoint to the
paper's BF16 and W4A4 MBPriorQ WikiText2 results. It exercises weight
quantization, the online activation prior, 16-to-4 refinement, `lm_head`, EBW
accounting, and result validation.

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/smoke_test/run.sh
```

The complete run evaluates all 146 contiguous 2048-token windows and validates:

| Method | PPL |
|---|---:|
| BF16 | 20.9240 |
| W4A4 MBPriorQ | 24.2289 |

`run_quick.sh` evaluates four windows. Qwen3-0.6B weight quantization
automatically uses the bundled imatrix.

## Streamed-Execution Equivalence

The following workflow compares full-GPU and one-layer-at-a-time execution on
the same BF16 and MBPriorQ checkpoints. It validates total NLL with a relative
tolerance of `2e-4`.

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/smoke_test/run_offload_equivalence.sh
```

Use `run_offload_equivalence_quick.sh` for a one-window check.
