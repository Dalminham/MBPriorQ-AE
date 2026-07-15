# Artifact Scope And Claim Map

## Evidence Policy

The artifact separates three levels:

1. **Full reproduction** runs the paper protocol and validates paper values.
2. **Extended reproduction** exposes the complete workflow but is optional due
   to model access, storage, or runtime.
3. **Functional validation** checks a mechanism or execution contract at
   bounded scale and is not presented as a paper result.

No workflow includes another author's implementation or result validator. No
commercial synthesis input, output, screenshot, area, or power data is part of
the artifact.

## Claim Map

| ID | Paper location | Public AE path | Coverage |
|---|---|---|---|
| KR1 | Table 2, Sec. 7.2 | `experiments/core_accuracy/`, `experiments/model_family_ppl/` | Qwen3-0.6B is the default full reproduction. One-click BF16/MBPriorQ drivers expose the original 16-model and revised 19-model sweeps as extended reproduction. |
| KR2 | Table 3, Sec. 7.2 | `experiments/downstream_benchmarks/` | BF16 and MBPriorQ GSM8K/MMLU/MMLU-Pro workflows for Qwen3-0.6B and Qwen3-14B. |
| KR3 | Table 5, Sec. 7.3 | `experiments/activation_attribution/` | Full attribution matrix for Qwen3-0.6B and Llama2-7B. |
| KR4 | Table 6, Sec. 7.3 | `experiments/granularity_ablation/` | 16-to-{8,4,2} PPL and EBW for Qwen3-0.6B and Llama2-7B. |
| KR5 | Tables 9-10, Secs. 7.4-7.5 | `scripts/run_smoke.py`, `software/mbpriorq_ae/ebw.py` | Metadata arithmetic and tensor-side scale rules are functional/reproduced; the full calibration sweep is not a default rerun. |
| KR6 | Sec. 6 dataflow | `scripts/run_hardware_modules.sh`, `scripts/run_hardware_system.sh` | Functional SpinalHDL/Verilator validation of metadata, MultiMSA, VPU/FPU-pool, output joining, and packetization. |
| KR7 | Figs. 10-11 and quantitative hardware tables | Paper results | Outside the AE rerun. No comparison design or commercial synthesis artifact is included. |
| KR8 | Low-memory execution used by large-model rows | `experiments/offload_equivalence/` | Full-GPU versus streamed NLL equivalence on Qwen3-0.6B; 19 model checkpoint structures are supported, including Qwen3-VL language-only. |

## Software Protocol Boundary

- Table 2 uses BF16 or paper-spec W4A4 MBPriorQ, contiguous 2048-token
  WikiText2 windows, and the submitted PPL denominator.
- Weight quantization includes `lm_head`.
- Only Qwen3-0.6B uses the declared imatrix. All other models reject it.
- Qwen3-VL Table 2 evaluates only the language path on text-only WikiText2.
- Central Tables 5 and 6 include both Qwen3-0.6B and Llama2-7B.
- Table 3 includes only the paper's BF16 and MBPriorQ rows in this artifact.

Fake-quantized checkpoints retain BF16 storage and are not physically packed
FP4 model files. This is the same fake-quantization accuracy methodology used
by the paper.

## Hardware Boundary

Open simulation establishes protocol and mechanism functionality. Module-cycle
fixtures are regression observables, not paper-level speedup claims. Hardware
area, power, speedup against other accelerators, generated RTL, and commercial
tool material remain outside the public AE package.

## Badge Target

The intended submission requests Artifact Available, Artifact Evaluated -
Functional, and Results Reproduced. Final badge eligibility depends on the
approved public tag, clean archive validation, Zenodo publication, and DOI.
