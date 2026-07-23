# Table 6 Refined-Granularity Ablation

This workflow reproduces 16-to-8, 16-to-4, and 16-to-2 MBPriorQ for **both
Qwen3-0.6B and Llama2-7B**, including weight and activation EBW.

```bash
QWEN_MODEL_PATH=/models/Qwen3-0.6B \
LLAMA_MODEL_PATH=/models/Llama-2-7b-hf \
DATASET_PATH=/datasets/wikitext-2-raw-v1 \
./experiments/table6/run.sh
```

Weights are quantized in memory for each granularity instead of writing six
large intermediate checkpoints. Qwen3-0.6B uses the bundled imatrix. Full
results are validated against [`expected.csv`](expected.csv). `run_quick.sh`
checks the three Qwen3-0.6B granularities with one window; the complete
`run.sh` remains the two-model protocol.
