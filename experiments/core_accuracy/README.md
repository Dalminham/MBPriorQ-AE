# Core Accuracy Reproduction

This workflow reproduces the representative Qwen3-0.6B WikiText2 rows used to
validate the central W4A4 accuracy claim:

1. BF16;
2. W4A4 MBPriorQ with the paper's cloud prior and 16-to-4 refinement.

The MBPriorQ row starts from the public BF16 checkpoint. The workflow first
creates a fake-quantized Hugging Face checkpoint, including `lm_head`, and then
applies activation fake quantization during a paper-compatible full-GPU PPL
run. The supplied importance matrix is used by the MBPriorQ weight checkpoint
generator; no other model consumes it.

Set the model and, optionally, a local WikiText2 dataset path:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/core_accuracy/run.sh
```

The complete run uses all 146 contiguous 2048-token windows and may take
several minutes to generate each checkpoint. For a functional check:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
DATASET_PATH=/path/to/wikitext-2-raw-v1 \
./experiments/core_accuracy/run_quick.sh
```

Generated checkpoints and results remain under the ignored `local_runs/`
directory. They are not part of the Zenodo archive.

## Expected Results

The full validator expects 146 windows and accepts small numerical variation:

| Row | PPL | Tolerance |
|---|---:|---:|
| BF16 | 20.9240 | 0.0500 |
| W4A4 MBPriorQ | 24.2289 | 0.0750 |

On the validated RTX 5090, cached rows complete in minutes. A clean run also
creates one approximately 1.2 GB checkpoint and should be budgeted at 10-20
minutes. Runtime depends on storage and GPU generation.

## Troubleshooting

- A missing `MODEL_PATH` or model-file error means the public Qwen3-0.6B
  checkpoint has not been downloaded to the selected directory.
- A dataset connection error can be avoided by downloading WikiText2 with
  Hugging Face Datasets and setting `DATASET_PATH` to that local directory.
- CUDA out-of-memory errors usually indicate another process is using the GPU;
  this representative full-GPU workflow is validated with at least 8 GB free
  VRAM. Use `offload_equivalence` to exercise the lower-memory backend.
- Delete the corresponding directory under `local_runs/checkpoints/` to rebuild
  a stale generated checkpoint from scratch.
