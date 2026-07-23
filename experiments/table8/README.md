# Table 8: KV-Cache Quantization

This workflow keeps Linear weights and activations in BF16 and quantizes only
the KV cache. It compares BF16 KV cache (`W16A16KV16`), NVFP4, and MBPriorQ
(`W16A16KV4`) on Qwen3-0.6B and Llama2-7B. Both FP4 rows quantize K along the
sequence axis and V along the head dimension.

```bash
QWEN_MODEL_PATH=/path/to/Qwen3-0.6B \
LLAMA_MODEL_PATH=/path/to/Llama-2-7b-hf \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/table8/run.sh
```

The full workflow evaluates the complete WikiText2 token stream, including its
final partial chunk, and writes `local_runs/table8/table8_kv_cache.csv`.
`run_quick.sh` validates the three Qwen3-0.6B paths with one chunk; the complete
`run.sh` remains the two-model protocol.
