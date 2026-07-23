# Table 2: Model-Family Perplexity

This directory has one public workflow for the complete Table 2 in the paper.
It runs BF16 and W4A4 MBPriorQ on WikiText2 for all 19 models declared in
[`models.json`](models.json), producing 38 PPL result rows. The same MBPriorQ
runs emit the scale, mask, and total EBW values reported in Table 10.

Qwen3-0.6B runs fully on the GPU. Every other model uses one-layer-at-a-time
safetensors streaming, so the complete checkpoint does not need to fit VRAM.
Qwen3-VL is evaluated through its language path on text-only WikiText2, matching
the paper protocol.

## Reproduce Tables 2 and 10

Place the public model checkpoints below one or more roots, then run:

```bash
MODEL_ROOTS=/models/hf:/models/modelscope \
./experiments/table2/run.sh \
  --dataset /datasets/wikitext-2-raw-v1
```

When local directory names differ, provide an explicit JSON map:

```json
{
  "llama2_7b": "/models/Llama-2-7b-hf",
  "qwen3_0_6b": "/models/Qwen3-0.6B",
  "qwen3_vl_235b_a22b": "/models/Qwen3-VL-235B-A22B-Instruct"
}
```

Pass the file with `--model-map paths.json`. Before execution, the runner
resolves all 19 checkpoints and fails as one preflight error if any are
missing. It resumes valid result JSONs, updates `observation.md` after every
row, and writes two final summaries:

- `summary.csv`: the 38 BF16/MBPriorQ PPL rows for Table 2;
- `side_metadata_overhead.csv`: one MBPriorQ side-metadata row per model plus
  the 19-model average for Table 10.

The complete run validates model coverage and PPL values against
[`models.json`](models.json), and side-metadata results against
[`expected_side_metadata.csv`](expected_side_metadata.csv).

## Check Model Compatibility

Validate model paths and checkpoint layouts without reading weights:

```bash
MODEL_ROOTS=/models/hf:/models/modelscope \
./experiments/table2/run.sh --preflight-only
```

Exercise architecture, streaming, and quantization compatibility on the first
five layers:

```bash
MODEL_ROOTS=/models/hf:/models/modelscope \
./experiments/table2/run.sh \
  --layer-smoke 5 \
  --dataset /datasets/wikitext-2-raw-v1
```

The runner retains existing complete MBPriorQ PPL entries and applies this check
to the remaining models. It loads one 2048-token window, streams layers 0-4,
applies MBPriorQ weight and activation quantization, verifies finite outputs,
and writes the results under `local_runs/table2/layer_smoke/`.
