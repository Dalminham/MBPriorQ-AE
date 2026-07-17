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
large intermediate checkpoints. Qwen3-0.6B alone consumes the bundled
imatrix; Llama2-7B explicitly does not. Full results are validated against
[`expected.csv`](expected.csv). `run_quick.sh` uses one window and validates
execution rather than paper values.
