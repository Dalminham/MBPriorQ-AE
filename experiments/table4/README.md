# Table 4: Feature Selection PPL

This workflow reproduces the activation-isolated W16A4 comparison of five VMB
selection features on Qwen3-0.6B and Llama2-7B. Its output contains the absolute
PPL for every run and the Table 4 relative PPL increase derived from each
model's BF16 PPL.

```bash
QWEN_MODEL_PATH=/path/to/Qwen3-0.6B \
LLAMA_MODEL_PATH=/path/to/Llama-2-7b-hf \
./experiments/table4/run.sh
```

`run.sh` first collects model-specific activation-gradient statistics from four
512-token WikiText2 training windows, then runs BF16 and the five feature rows on
the complete WikiText2 test set. Results are written to
`local_runs/table4/table4_ppl.csv`.

`run_quick.sh` uses one calibration window and one PPL window to validate the
execution path; it is not paper evidence.
