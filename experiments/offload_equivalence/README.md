# Full-GPU / Streamed Equivalence

This AE-only ablation checks whether layer streaming changes numerical results.
It evaluates the same Qwen3-0.6B BF16 checkpoint and the same generated
MBPriorQ checkpoint with:

1. the complete model resident on GPU;
2. one safetensors decoder layer resident at a time.

```bash
MODEL_PATH=/models/Qwen3-0.6B \
DATASET_PATH=/datasets/wikitext-2-raw-v1 \
./experiments/offload_equivalence/run.sh
```

`run_quick.sh` uses one 2048-token window. The validator compares total NLL,
not rounded PPL, and fails if relative drift exceeds `2e-4`. This experiment
does not claim that streaming is faster; it establishes that the low-memory
execution path preserves the software result.
