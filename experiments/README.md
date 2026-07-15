# Experiment Entry Points

| Directory | Paper result | Full command |
|---|---|---|
| [`model_family_ppl/`](model_family_ppl/README.md) | Table 2, 16/19-model BF16 and MBPriorQ PPL | `run_paper16.sh` / `run_table2_19.sh` |
| [`core_accuracy/`](core_accuracy/README.md) | Table 2, Qwen3-0.6B representative row | `run.sh` |
| [`downstream_benchmarks/`](downstream_benchmarks/README.md) | Table 3 | `run.sh` |
| [`activation_attribution/`](activation_attribution/README.md) | Table 5, both models | `run.sh` |
| [`granularity_ablation/`](granularity_ablation/README.md) | Table 6, both models | `run.sh` |
| [`offload_equivalence/`](offload_equivalence/README.md) | AE backend correctness check | `run.sh` |

Every full paper workflow writes machine-readable JSON below ignored
`local_runs/`. Quick scripts validate execution with reduced samples; their
numbers are not paper evidence. No directory runs or validates another
author's method.
