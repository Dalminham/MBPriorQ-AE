# PPL Evidence

The CSV files in this directory are compact transcriptions of completed paper
runs. They are expected values, not substitutes for executing the public
workflows.

- `qwen3_0_6b_smoke_test.csv`: lightweight BF16 and MBPriorQ reproduction.
- `accuracy_attribution.csv`: activation-prior attribution under W4A4.
- `granularity_ablation.csv`: 16-to-{8,4,2} refinement tradeoff.
- `local_validation.json`: sanitized exact outputs and tested environment from
  the completed public Qwen3-0.6B workflows.

The dual-model Table 5/6 expectations and complete 19-model Table 2 expectations
live beside their executable drivers under `experiments/`; this directory
retains the compact completed Qwen3-0.6B evidence rather than duplicating those
matrices.

All rows use WikiText2 raw test, BF16 execution, 2048-token contiguous windows,
and the paper-compatible layer-wise PPL denominator. The full Qwen3-0.6B run
contains 146 windows. No comparison-method reproduction is included.
