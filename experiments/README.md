# Experiment Entry Points

| Directory | Paper result | Full command |
|---|---|---|
| [`table2_ppl/`](table2_ppl/README.md) | Table 2, all 19 models under BF16 and MBPriorQ | `run.sh` |
| [`qwen3_0_6b_smoke_test/`](qwen3_0_6b_smoke_test/README.md) | Lightweight model-level BF16/MBPriorQ reproduction | `run.sh` / `run_quick.sh` |
| [`downstream_benchmarks/`](downstream_benchmarks/README.md) | Table 3 | `run.sh` |
| [`activation_attribution/`](activation_attribution/README.md) | Table 5, both models | `run.sh` |
| [`granularity_ablation/`](granularity_ablation/README.md) | Table 6, both models | `run.sh` |
| [`offload_equivalence/`](offload_equivalence/README.md) | AE backend correctness check | `run.sh` |

Every full paper workflow writes machine-readable JSON below ignored
`local_runs/`. Quick scripts validate execution with reduced samples; their
numbers are not paper evidence. No directory runs or validates another
author's method.
