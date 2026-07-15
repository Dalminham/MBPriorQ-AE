# Table 3 Downstream Benchmarks

The driver reproduces the BF16 and MBPriorQ rows of Table 3 for Qwen3-0.6B and
Qwen3-14B. It does not include any comparison method.

| Benchmark | Paper protocol |
|---|---|
| GSM8K | 500 examples sampled with `random.Random(0)` from the 1,319-row test set; last numerical answer; tolerance `1e-6`. |
| MMLU | 100 examples sampled with `random.Random(0)` from `mmlu.csv`; final A-D answer. |
| MMLU-Pro | All 410 `computer science` test rows; first five matching validation examples as CoT demonstrations; final A-J answer. |

Generation uses the submitted prompt templates, BF16, temperature 0.1, top-p
0.9, repetition penalty 1.1, and up to 2048 new tokens. A deterministic
per-example seed makes interrupted runs resumable.

```bash
QWEN_0_6B_MODEL_PATH=/models/Qwen3-0.6B \
QWEN_14B_MODEL_PATH=/models/Qwen3-14B \
GSM8K_DATASET_PATH=/datasets/gsm8k \
MMLU_DATASET_PATH=/datasets/mmlu \
MMLU_PRO_DATASET_PATH=/datasets/MMLU-Pro \
./experiments/downstream_benchmarks/run.sh
```

The Qwen3-0.6B MBPriorQ checkpoint uses the bundled imatrix. Qwen3-14B does
not. Qwen3-14B loads with Hugging Face `device_map=auto` so CPU offload can be
used when it does not fit VRAM.

For a one-example smoke on Qwen3-0.6B only:

```bash
QWEN_0_6B_MODEL_PATH=/models/Qwen3-0.6B \
GSM8K_DATASET_PATH=/datasets/gsm8k \
MMLU_DATASET_PATH=/datasets/mmlu \
MMLU_PRO_DATASET_PATH=/datasets/MMLU-Pro \
./experiments/downstream_benchmarks/run_quick.sh
```

Each benchmark keeps an append-only JSONL journal. BF16 resumes at the first
unfinished row. MBPriorQ first replays completed prompts to reconstruct the
online activation-prior state, then appends new rows; otherwise a resumed run
would not be numerically equivalent to an uninterrupted run. Set `RESUME=0`
to restart. Complete runs compare the six accuracies with the paper values in
`expected.csv`; quick runs validate execution only.
