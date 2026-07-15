# Refined-Granularity Reproduction

This workflow reproduces the Qwen3-0.6B half of the paper's 16-to-8/4/2
refinement ablation. Each row quantizes both weights and activations at the
selected refined block size and reports PPL plus weight/activation EBW.

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/granularity_ablation/run.sh
```

The first run creates up to three approximately 1.2 GB fake-quantized
checkpoints under the ignored `local_runs/checkpoints/` directory. Subsequent
runs reuse them. Results are validated against `expected.csv`.

## Expected Results

| Refinement | PPL | Weight EBW | Activation EBW |
|---|---:|---:|---:|
| 16-to-8 | 24.8590 | 4.598690 | 4.704345 |
| 16-to-4 | 24.2289 | 4.673069 | 4.994211 |
| 16-to-2 | 23.9532 | 4.837506 | 5.613351 |

With cached checkpoints the validated RTX 5090 run completed in several
minutes. A clean run creates up to 3.6 GB of generated checkpoints; budget
20-30 minutes for checkpoint generation and all 146-window PPL rows.

## Troubleshooting

- Check that the supplied importance matrix is present at
  `data/imatrix/Qwen_Qwen3-0.6B.imatrix` and passes the release checksum.
- Checkpoint and activation refined block sizes must match. Remove only the
  stale `qwen3_0_6b_mbpriorq_rb*` checkpoint directory before rebuilding it.
- Set `NUM_SAMPLES=0` for a paper-compatible complete run; smaller values are
  functional checks and intentionally fail `--require-full` validation.
