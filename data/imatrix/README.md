# Qwen3-0.6B Importance Matrix

`Qwen_Qwen3-0.6B.imatrix` is the precomputed llama.cpp-style importance matrix
used by the paper's Qwen3-0.6B MBPriorQ weight fake quantization.

- SHA-256: `86c939f5185eff29eb6f94fec0c37b95b5614119709f8f78d1cc66127d99a386`
- Size: 1,153,382 bytes
- Scope: weight scale fitting only; activation prior selection does not read it.
- Source model: `Qwen/Qwen3-0.6B`.
- Public source URL:
  `https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B.imatrix`

This is the **only** model in the artifact that uses an imatrix. Public drivers
require it for Qwen3-0.6B and reject it for every other model.

The file contains per-input-channel calibration statistics, not model weights.
The associated Qwen license is retained at `third_party/qwen3/LICENSE`.
