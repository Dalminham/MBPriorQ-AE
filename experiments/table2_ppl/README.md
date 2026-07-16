# Table 2: Model-Family Perplexity

This directory has one public workflow for the complete Table 2 in the paper.
It runs BF16 and W4A4 MBPriorQ on WikiText2 for all 19 models declared in
[`models.json`](models.json), producing 38 result rows.

Qwen3-0.6B runs fully on the GPU. Every other model uses one-layer-at-a-time
safetensors streaming, so the complete checkpoint does not need to fit VRAM.
Qwen3-VL is evaluated through its language path on text-only WikiText2, matching
the paper protocol.

## Run The Complete Table

Place the public model checkpoints below one or more roots, then run:

```bash
MODEL_ROOTS=/models/hf:/models/modelscope \
./experiments/table2_ppl/run.sh \
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
row, and writes the final 38-row `summary.csv`.

## Fast Checks

Validate model paths and checkpoint layouts without reading weights:

```bash
MODEL_ROOTS=/models/hf:/models/modelscope \
./experiments/table2_ppl/run.sh --preflight-only
```

Exercise one model and one PPL window:

```bash
MODEL_ROOTS=/models/hf \
./experiments/table2_ppl/run.sh \
  --only qwen3_0_6b --num-samples 1 \
  --dataset /datasets/wikitext-2-raw-v1
```

Only a complete run validates the observed values against Table 2. Reduced
sample counts are backend checks and must not be reported as paper results.

## Resource Notes

The complete table is storage- and time-intensive. Paper-compatible weight
fake quantization remains tensor-level and CPU-resident before each streamed
layer is sent to the GPU. The largest MBPriorQ rows can therefore take a day or
longer even though only one layer occupies VRAM. Only Qwen3-0.6B consumes the
bundled imatrix; the runner rejects that metadata for every other model.
