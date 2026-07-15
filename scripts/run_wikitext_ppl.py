#!/usr/bin/env python3
"""Run one paper-compatible WikiText2 PPL row."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software"))

import torch

from mbpriorq_ae import GlobalEBW
from mbpriorq_ae.integration import (
    ActivationQuantizationConfig,
    make_quantizer,
    wrap_activation_linears,
)
from mbpriorq_ae.logging import set_log_level
from mbpriorq_ae.offload import quantize_model_weights, streamed_ppl
from mbpriorq_ae.perplexity import (
    encode_dataset,
    load_text_dataset,
    paper_compatible_full_model_ppl,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Source or fake-quantized HF checkpoint")
    parser.add_argument("--tokenizer", help="Tokenizer path/id; defaults to --model")
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", required=True, choices=("bf16", "mbpriorq"))
    parser.add_argument(
        "--backend",
        default="full_gpu",
        choices=("full_gpu", "streamed"),
        help="Keep the whole model on GPU or materialize one safetensors layer at a time",
    )
    parser.add_argument(
        "--weight-source",
        default="auto",
        choices=("auto", "online", "prequant"),
        help="Quantize source weights in memory or consume an AE-generated checkpoint",
    )
    parser.add_argument(
        "--model-key",
        default="",
        help="Manifest key used to enforce model-specific inputs such as the Qwen3-0.6B imatrix",
    )
    parser.add_argument("--imatrix", help="Qwen3-0.6B weight-importance matrix")
    parser.add_argument(
        "--model-family",
        default="auto",
        choices=("auto", "decoder", "qwen3_vl_language"),
    )
    parser.add_argument("--model-type", default="cloud", choices=("cloud", "edge"))
    parser.add_argument(
        "--ablation-mode",
        default="paper",
        choices=("paper", "static", "first2_only", "random_same_ratio", "oracle"),
    )
    parser.add_argument("--random-seed", type=int, default=20260606)
    parser.add_argument("--refined-block-size", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _validate_checkpoint(args) -> dict | None:
    if args.method == "bf16":
        return None
    path = Path(args.model)
    metadata_path = path / "mbpriorq_ae_prequant_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Quantized PPL requires an AE-generated checkpoint: missing {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("method") != args.method:
        raise ValueError(
            f"Checkpoint method {metadata.get('method')!r} does not match {args.method!r}"
        )
    if args.method == "mbpriorq" and metadata.get("refined_block_size") != args.refined_block_size:
        raise ValueError("Checkpoint and activation refined block sizes do not match")
    if not metadata.get("lm_head_quantized"):
        raise ValueError("Checkpoint metadata does not confirm lm_head quantization")
    uses_imatrix = bool(metadata.get("using_imatrix"))
    if args.model_key == "qwen3_0_6b" and not uses_imatrix:
        raise ValueError("Qwen3-0.6B MBPriorQ checkpoint must record imatrix use")
    if args.model_key and args.model_key != "qwen3_0_6b" and uses_imatrix:
        raise ValueError("Only qwen3_0_6b may use an imatrix")
    return metadata


def _resolve_weight_source(args) -> tuple[str, dict | None]:
    metadata_path = Path(args.model) / "mbpriorq_ae_prequant_metadata.json"
    source = args.weight_source
    if source == "auto":
        source = "prequant" if metadata_path.is_file() else "online"
    if args.method == "bf16":
        if args.imatrix:
            raise ValueError("BF16 evaluation does not consume an imatrix")
        return "none", None
    if source == "prequant":
        return source, _validate_checkpoint(args)
    if source != "online":
        raise ValueError(f"Unsupported weight source: {source}")
    if args.model_key == "qwen3_0_6b" and not args.imatrix:
        raise ValueError("Qwen3-0.6B MBPriorQ weight quantization requires --imatrix")
    if args.imatrix and args.model_key != "qwen3_0_6b":
        raise ValueError("The imatrix is valid only for model-key qwen3_0_6b")
    if args.imatrix and not Path(args.imatrix).is_file():
        raise FileNotFoundError(f"Importance matrix not found: {args.imatrix}")
    return source, None


def main():
    args = parse_args()
    set_log_level("release")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    weight_source, checkpoint_metadata = _resolve_weight_source(args)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("PPL evaluation requires transformers") from error

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    GlobalEBW.reset()
    wrapped: list[str] = []
    dataset = load_text_dataset(args.dataset, args.split)
    input_ids = encode_dataset(tokenizer, dataset)
    activation_config = None
    weight_quantizer = None
    if args.method == "mbpriorq":
        activation_config = ActivationQuantizationConfig(
            method=args.method,
            model_type=args.model_type,
            ablation_mode=args.ablation_mode,
            random_seed=args.random_seed,
            refined_block_size=args.refined_block_size,
        )
        if weight_source == "online":
            weight_quantizer = make_quantizer(
                method=args.method,
                name=None,
                model_type=args.model_type,
                ablation_mode="paper",
                random_seed=args.random_seed,
                refined_block_size=args.refined_block_size,
                using_imatrix=bool(args.imatrix),
                imatrix_file_name=args.imatrix,
            )

    quantized_weight_count = 0
    resolved_family = args.model_family
    if args.backend == "full_gpu":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            use_safetensors=True,
            trust_remote_code=True,
        )
        model.eval()
        if weight_quantizer is not None:
            quantized_weight_count = quantize_model_weights(model, weight_quantizer)
        if activation_config is not None:
            wrapped = wrap_activation_linears(model, activation_config)
        ppl, windows, total_nll = paper_compatible_full_model_ppl(
            model,
            input_ids,
            sequence_length=args.sequence_length,
            num_samples=args.num_samples,
            device=args.device,
            progress=not args.quiet,
        )
        resolved_family = "full_causal_lm"
    else:
        streamed = streamed_ppl(
            checkpoint=args.model,
            input_ids=input_ids,
            sequence_length=args.sequence_length,
            num_samples=args.num_samples,
            device=args.device,
            dtype=torch.bfloat16,
            model_family=args.model_family,
            weight_quantizer=weight_quantizer,
            activation_config=activation_config,
            progress=not args.quiet,
        )
        ppl = streamed.perplexity
        windows = streamed.windows
        total_nll = streamed.total_nll
        quantized_weight_count = streamed.quantized_weight_count
        wrapped = ["streamed"] * streamed.wrapped_linear_count
        resolved_family = streamed.model_family
    result = {
        "method": args.method,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "dtype": "bfloat16",
        "sequence_length": args.sequence_length,
        "num_samples": windows,
        "perplexity": ppl,
        "total_nll": total_nll,
        "backend": args.backend,
        "model_family": resolved_family,
        "weight_source": weight_source,
        "model_key": args.model_key or None,
        "imatrix_used": (
            bool(checkpoint_metadata.get("using_imatrix"))
            if checkpoint_metadata
            else bool(args.imatrix)
        ),
        "model_type": args.model_type if args.method == "mbpriorq" else None,
        "ablation_mode": args.ablation_mode if args.method == "mbpriorq" else None,
        "refined_block_size": args.refined_block_size if args.method == "mbpriorq" else None,
        "wrapped_linear_count": len(wrapped),
        "quantized_weight_count": quantized_weight_count,
        "activation_ebw_summary": GlobalEBW.summarize("activation"),
        "weight_ebw_summary": (
            checkpoint_metadata.get("weight_ebw_summary")
            if checkpoint_metadata
            else GlobalEBW.summarize("weight")
        ),
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
