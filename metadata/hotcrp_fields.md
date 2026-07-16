# MICRO 2026 Artifact Evaluation Fields

This file is a submission draft. Replace the DOI placeholder only after the
tagged Zenodo record resolves publicly.

## Title

Software-Hardware Co-Design of Prior-Aware W4A4 Micro-Block Quantization for
Robust LLM Inference Across Model Families

## Abstract

Jointly quantizing both weights and activations to 4 bits (W4A4) has become
increasingly popular for efficient large language model inference, but robust
activation quantization across model families remains difficult. Industrial FP4
micro-block formats such as NVFP4 and MXFP4 still show large robustness
disparities across model families, indicating that current methods lack outlier
awareness at the micro-block level. We find that this disparity is mainly
determined by a small fraction of all micro-blocks. These micro-blocks have high
internal variance, and we define them as vulnerable micro-blocks (VMBs).
Existing methods often improve activation handling through mixed-precision
implementation or expensive calibration, but these choices increase hardware
cost or calibration cost.

We propose MBPriorQ, a software-hardware co-design for robust LLM inference
under W4A4 post-training quantization. MBPriorQ selectively refines only VMBs,
uses low-cost online prior refinement to track their temporal variation, and is
supported by a low-overhead mixed-granularity hardware implementation.

Across 19 model entries from multiple model families, MBPriorQ achieves the
strongest robustness in our evaluation. It reduces perplexity degradation by up
to 40% on vulnerable models relative to NVFP4 with minimal storage overhead on
average and negligible dense-calibration compute overhead. Compared with
state-of-the-art open accelerator designs, MBPriorQ delivers 1.16x-2.28x
speedups for core matrix multiplication. These results show that robust W4A4
micro-block quantization across model families can remain practical under a
low-overhead hardware implementation.

## DOI URL

`https://doi.org/10.5281/zenodo.TO_BE_ASSIGNED`

## Badges Applied For

- Artifact Available
- Artifacts Evaluated - Functional
- Results Reproduced

## Key Results To Be Reproduced

1. **Lightweight W4A4 model reproduction (Table 2):** reproduce the Qwen3-0.6B
   WikiText2 rows for BF16 and MBPriorQ over all 146 contiguous 2048-token
   windows. Expected PPL is 20.9240 and 24.2289, respectively.
2. **Central ablations (Tables 5-6):** reproduce the Qwen3-0.6B and Llama2-7B
   activation-prior attribution and 16-to-{8,4,2} refined-granularity rows,
   including PPL and effective-bit-width accounting.
3. **Hardware functionality (Sec. 6):** validate the retained SpinalHDL modules
   for independent mask/scale selection, metadata-first VPU issue, shared
   FPU-pool execution, regular/refined MultiMSA output, matrix-scale joining,
   and the public 1024-bit packet path. This is a Functional target; hardware
   speedup, area, and power are not claimed as AE-reproduced results.

## Hardware Dependencies

The functional software workflow needs a Linux x86-64 machine with at least 4
CPU cores and approximately 8 GB RAM. The lightweight PPL reproduction needs
an NVIDIA CUDA GPU with at least 8 GB free VRAM, at least 16 GB system RAM, and
approximately 10 GB free disk space. Open hardware simulation is recommended on
an 8-core host with 16 GB RAM and 10 GB temporary disk space; it does not need a
GPU. The artifact was validated on an AMD Ryzen 7 9800X3D, 186 GiB RAM, and an
NVIDIA RTX 5090 with 32 GiB VRAM.

## Software Dependencies

Linux x86-64; Python 3.12; PyTorch 2.10; Transformers 4.57; Datasets 4.7;
Safetensors 0.7; NumPy 2.2. Open hardware simulation additionally uses OpenJDK
17, sbt 1.10.2, Scala 2.12.19, SpinalHDL 1.8.0, Verilator 4.034 or newer, GNU
Make, and a C++ compiler. No commercial EDA dependency or data is part of the
artifact.

## Data Dependencies

The archive includes the small Qwen3-0.6B importance matrix used for MBPriorQ
weight scale fitting, but it does not redistribute model weights or WikiText2.
The Results Reproduced workflow requires the public `Qwen/Qwen3-0.6B` checkpoint
and the public `wikitext-2-raw-v1` test split, downloaded from their official
providers under their respective terms. The functional tensor and hardware
workflows require no external model or dataset.

## Final Submission Actions

- publish the public repository and `v1.0.0` tag;
- publish the Zenodo record and replace the DOI placeholder;
- upload the paper plus artifact appendix and paste these fields into HotCRP;
- mark the submission ready for review before the deadline.
