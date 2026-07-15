# Table 2 Model-Family PPL

This workflow runs BF16 and W4A4 MBPriorQ on WikiText2 for every model listed
in [`models.json`](models.json).

## Suites

- `run_paper16.sh`: the original 16-model suite used by the submitted paper.
- `run_table2_19.sh`: the revised Table 2, adding Qwen2.5-72B,
  Qwen3-Next-80B-A3B, and Qwen3-VL-235B-A22B.

Qwen3-VL is evaluated on its language path with text-only WikiText2, exactly as
stated in the revised paper. Qwen3-0.6B runs fully on GPU; other models use
one-layer-at-a-time safetensors streaming.

## Run

```bash
MODEL_ROOTS=/models/hf:/models/modelscope \
./experiments/model_family_ppl/run_paper16.sh \
  --dataset /datasets/wikitext-2-raw-v1
```

Use a JSON path map when local directory names differ:

```json
{
  "llama2_7b": "/models/Llama-2-7b-hf",
  "qwen3_0_6b": "/models/Qwen3-0.6B"
}
```

Pass it with `--model-map paths.json`. The runner checks all requested model
paths before launching, supports `--resume`, writes one JSON per row, and keeps
an immediately readable `observation.md` and final `summary.csv`. Complete runs
also fail if a PPL differs from the rounded paper value by more than the
declared tolerance; reduced-window runs report values without making that
paper-result check.

For a path/backend preflight without reading model tensors:

```bash
MODEL_ROOTS=/models/hf:/models/modelscope \
./experiments/model_family_ppl/run_paper16.sh --preflight-only
```

To test one model and one PPL window:

```bash
MODEL_ROOTS=/models/hf \
./experiments/model_family_ppl/run_paper16.sh \
  --only qwen3_0_6b --num-samples 1 \
  --dataset /datasets/wikitext-2-raw-v1
```

The complete suite is a long, storage-intensive reproduction. Weight fake
quantization remains tensor-level and CPU-resident before each streamed layer
is sent to the GPU, matching the checkpoint-generation semantics used for the
paper. The optional Qwen3-VL MBPriorQ row is therefore a day-scale run even
though BF16 streaming is much faster. Reduced sample counts are execution
checks and must not be reported as Table 2 values.
