# PPL Evidence

The CSV files in this directory are compact transcriptions of completed
revision runs. They are expected values, not substitutes for executing the
public workflows.

- `core_accuracy.csv`: representative BF16 and MBPriorQ results.
- `accuracy_attribution.csv`: activation-prior attribution under W4A4.
- `granularity_ablation.csv`: 16-to-{8,4,2} refinement tradeoff.
- `local_validation.json`: sanitized exact outputs and tested environment from
  the completed public Qwen3-0.6B workflows.

All rows use WikiText2 raw test, BF16 execution, 2048-token contiguous windows,
and the paper-compatible layer-wise PPL denominator. The full Qwen3-0.6B run
contains 146 windows. No comparison-method reproduction is included.
