# Experiment Entry Points

Experiment directories use paper-oriented names and expose a stable `run.sh`
plus machine-readable expected results. Reviewer identifiers and historical
debug labels are intentionally excluded from the public interface.

- [`core_accuracy/`](core_accuracy/README.md): representative Qwen3-0.6B BF16
  and MBPriorQ WikiText2 reproduction.
- [`activation_attribution/`](activation_attribution/README.md): Qwen3-0.6B
  activation-prior attribution.
- [`granularity_ablation/`](granularity_ablation/README.md): Qwen3-0.6B
  16-to-8/4/2 refinement trade-off.

Generated models and results are stored under the ignored `local_runs/`
directory. The workflows share a checkpoint cache to avoid repeating identical
weight-quantization work.
