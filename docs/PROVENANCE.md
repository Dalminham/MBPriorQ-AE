# Source Provenance

## Author Repositories

| Component | Source repository | Pinned source | Imported scope |
|---|---|---|---|
| MBPriorQ software/evaluation | `https://github.com/Dalminham/EasyLLM.git` | branch `MICRO`, commit `e48ba9c618553bf1036d83c9dbed35e30b5ad3ba`, tag `micro-ae-baseline-20260701` | MBPriorQ tensor semantics, checkpoint generation, layer-wise PPL integration, and EBW accounting. |
| MBPriorQ accelerator | `https://github.com/Dalminham/MBPriorQ-Accelerator.git` | branch `main`, commit `476c7bc9846e377309e93a69d66626cab63b3ddd` | Public-top SpinalHDL design closure and functional simulations. |
| Paper evidence | MICRO 2026 accepted-paper workspace | per-file SHA-256 in artifact manifests | Author-generated expected outputs and the Qwen3-0.6B importance matrix only. |

The author-generated source curated into this artifact is released under
Apache-2.0. This is an explicit license for this artifact and does not alter the
licensing of files outside its source closure. Retained third-party notices
continue to govern their respective materials.

## Third-Party Boundary

- Model weights and WikiText2 are not redistributed. Evaluators obtain them
  from their official providers under the providers' terms.
- The Qwen3-0.6B license accompanying the calibration source model is retained
  at `third_party/qwen3/LICENSE`.
- No source, binary, or result-validation workflow for a comparison method is
  included.
- Commercial EDA inputs, outputs, screenshots, and area/power data are excluded.
- Reproductions of unrelated baseline software in EasyLLM are excluded.

## Documented Software Adaptations

The curated package removes EasyLLM's import-time dependency on unrelated
quantizers and replaces a workstation-specific plotting side effect with a
no-op hook. Quantization arithmetic is unchanged. The source-comparison script
checks all imported MBPriorQ prior modes and refined sizes plus MBPriorQ weight
quantization element-for-element against the pinned EasyLLM revision. The
standalone uniform-format comparison workflow from EasyLLM is not included.
Identifiers containing `NVFP4` inside the MBPriorQ implementation denote the
regular 16-value FP4 numeric path that MBPriorQ selectively refines; they do not
constitute a standalone comparison-method workflow.

The AE PPL wrapper preserves the submitted layer-wise evaluation contract:
contiguous 2048-token windows and the historical `windows * sequence_length`
denominator. One integration correction is intentional and tested: modules in a
`ModuleList` are named with bracket indices (for example,
`model.layers[0]`) to match EasyLLM. These names feed the deterministic
same-ratio random-ablation seed, so dot-index names would change that row.

Generated W4A4 checkpoints contain fake-quantized BF16 tensors, include
`lm_head`, and are local run products. They are not physically packed FP4 model
weights and are excluded from the archive.

## Importance Matrix

`data/imatrix/Qwen_Qwen3-0.6B.imatrix` is the author-provided calibration
statistic used by the paper's Qwen3-0.6B MBPriorQ weight scale fitting. It is not
a model checkpoint. Its SHA-256 is
`86c939f5185eff29eb6f94fec0c37b95b5614119709f8f78d1cc66127d99a386`
and its size is 1,153,382 bytes. Activation-prior selection does not read this
file.

## Documented Hardware Adaptations

The hardware directory is a curated transitive source closure of the public
1024-bit MBPriorQ top, not a mirror of every historical accelerator prototype.
Unreferenced prototypes, standalone analysis modules, generated Verilog, and
performance-only trace models were omitted. Refined-aware wrapper and public-top
interface identifiers were standardized to the paper terms `MultiMSA` and
`MSA`; these are terminology-only changes.

AE-specific Spinal simulations were added for scale reconstruction, packet
scheduling, the shared FPU pool, regular/refined MultiMSA output, output joining,
and the public packet top. Simulation outputs are redirected to ignored
`local_runs/` directories and compared with the checked expected CSVs. No
arithmetic or scheduling rule in the retained design path was changed.

## Import Requirements

Every distributable imported file must have a recorded origin and revision,
document whether it is copied or adapted, retain required notices, be covered by
the release checksum manifest, exclude machine-specific paths, avoid
comparison-method reproductions and commercial EDA data, and pass the public
workflow before release.
