# Table 7: VMB Prior Robustness

This workflow profiles the deployed two-token VMB prior on Qwen3-0.6B and
Llama2-7B. It reproduces the five input-variation axes in Table 7:

- WikiText2 versus PTB domain inputs;
- MMLU-Pro multiple-choice prompt text;
- 512, 1024, 2048, and 4096-token contexts at a fixed token budget;
- five deterministic WikiText2 starting segments;
- true batch sizes 1, 2, and 4 over the same WikiText2 windows.

The domain, prompt, and context rows report the fraction of VMBs from the
complete current activation that the deployed prior misses. The input-segment
and batch rows report pairwise overlap of selected micro-block columns.

```bash
QWEN_MODEL_PATH=/path/to/Qwen3-0.6B \
LLAMA_MODEL_PATH=/path/to/Llama-2-7b-hf \
WIKITEXT_PATH=/path/to/wikitext-2-raw-v1 \
PTB_PATH=/path/to/ptb-text-only \
MMLU_PRO_PATH=/path/to/MMLU-Pro \
./experiments/table7/run.sh
```

The driver generates MBPriorQ checkpoints once and reuses them across the
profiles. Qwen3-0.6B weight quantization uses the included imatrix. The final
outputs are
`local_runs/table7/table7_vmb_prior_robustness.csv` and a
run-level detail table beside it.

`run_quick.sh` exercises all five axes on Qwen3-0.6B with reduced input counts.
