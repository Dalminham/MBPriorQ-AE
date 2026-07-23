# Table 5 Accuracy Attribution

This workflow reproduces all Table 5 rows for **both Qwen3-0.6B and
Llama2-7B**: BF16, random same-ratio refinement, static first-batch reuse,
first-two-token refinement, deployed MBPriorQ, and full-activation recompute.

```bash
QWEN_MODEL_PATH=/models/Qwen3-0.6B \
LLAMA_MODEL_PATH=/models/Llama-2-7b-hf \
DATASET_PATH=/datasets/wikitext-2-raw-v1 \
./experiments/table5/run.sh
```

Qwen3-0.6B uses the bundled imatrix and the full-GPU backend. Llama2-7B uses no
imatrix and follows the paper-compatible by-layer path: the model remains in CPU
memory while one decoder layer at a time moves to the GPU. The script reuses one
16-to-4 fake-quantized checkpoint per model because all attribution rows share
identical weights.

Full results are validated against [`expected.csv`](expected.csv): 146 windows
for Qwen3-0.6B and 166 for Llama2-7B. `run_quick.sh` checks all six Qwen3-0.6B
rows with one window; the complete `run.sh` remains the two-model protocol.
