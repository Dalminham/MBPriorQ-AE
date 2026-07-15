# Activation Attribution Reproduction

This workflow reproduces the Qwen3-0.6B half of the paper's activation-prior
attribution table. It compares BF16, random same-ratio refinement, the stored
first-batch mask, first-two-token refinement, deployed MBPriorQ, and
complete-current-activation VMB recomputation. It does not rerun a comparison
method.

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/activation_attribution/run.sh
```

The run uses all 146 contiguous 2048-token windows. It shares generated weight
checkpoints with `core_accuracy` under `local_runs/checkpoints/`, writes JSON
results under `local_runs/activation_attribution/`, and validates both PPL and
effective-bit-width accounting against `expected.csv`.

## Expected Results

| Row | PPL | Activation EBW |
|---|---:|---:|
| BF16 | 20.9240 | N/A |
| Random same-ratio | 25.3423 | 4.994391 |
| Static first-batch | 24.5446 | 4.683127 |
| First-2-token only | 24.2756 | 4.966852 |
| MBPriorQ | 24.2289 | 4.994211 |
| Full-activation recompute | 24.3563 | 4.683410 |

All quantized attribution rows use weight EBW 4.673069. The full workflow took
about ten minutes with cached checkpoints on the validated RTX 5090; budget
20-30 minutes for a clean run.

## Troubleshooting

- Run `core_accuracy` first to populate the shared 16-to-4 MBPriorQ checkpoint,
  or allow this script to generate it.
- Do not rename wrapped `ModuleList` layers: bracket-index names participate in
  the deterministic same-ratio random seed.
- If validation reports a sample-count mismatch, ensure `NUM_SAMPLES=0` for the
  complete 146-window run.
