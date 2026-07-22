# Paper Experiments

| Directory | Paper result | Main output |
|---|---|---|
| [`smoke_test/`](smoke_test/README.md) | Lightweight BF16/MBPriorQ reproduction and streamed-execution equivalence | PPL, NLL, and EBW |
| [`table2/`](table2/README.md) | Tables 2 and 10 | 19-model PPL and side-metadata EBW |
| [`table3/`](table3/README.md) | Table 3 | GSM8K, MMLU, and MMLU-Pro accuracy |
| [`table5/`](table5/README.md) | Table 5 | Accuracy attribution PPL and EBW |
| [`table6/`](table6/README.md) | Table 6 | Refined-granularity PPL and EBW |
| [`table7/`](table7/README.md) | Table 7 | VMB-prior robustness statistics |
| [`table8/`](table8/README.md) | Table 8 | KV-cache PPL and EBW |

Run `run.sh` for the complete paper protocol. A `run_quick.sh` entry uses a
reduced workload to check the execution path. All generated files are written
below `local_runs/`.
