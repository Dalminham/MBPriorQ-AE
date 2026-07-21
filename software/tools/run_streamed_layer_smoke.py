#!/usr/bin/env python3
"""Validate MBPriorQ weight/activation quantization on the first streamed layers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

import torch

from mbpriorq_ae import GlobalEBW
from mbpriorq_ae.integration import ActivationQuantizationConfig, make_quantizer
from mbpriorq_ae.logging import set_log_level
from mbpriorq_ae.offload import streamed_layer_smoke
from mbpriorq_ae.perplexity import encode_dataset, load_text_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", help="Tokenizer path/id; defaults to --model")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--model-family", default="auto", choices=("auto", "decoder", "qwen3_vl_language"))
    parser.add_argument("--model-type", default="edge", choices=("cloud", "edge"))
    parser.add_argument("--imatrix")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.layers <= 0:
        raise ValueError("--layers must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.model_key == "qwen3_0_6b" and not args.imatrix:
        raise ValueError("Qwen3-0.6B layer smoke requires --imatrix")
    if args.imatrix and args.model_key != "qwen3_0_6b":
        raise ValueError("The imatrix is valid only for model-key qwen3_0_6b")
    if args.imatrix and not Path(args.imatrix).is_file():
        raise FileNotFoundError(f"Importance matrix not found: {args.imatrix}")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Layer smoke requires transformers") from error

    set_log_level("release")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True
    )
    input_ids = encode_dataset(
        tokenizer, load_text_dataset(args.dataset, args.split)
    )
    GlobalEBW.reset()
    activation_config = ActivationQuantizationConfig(
        method="mbpriorq",
        model_type=args.model_type,
        ablation_mode="paper",
        random_seed=20260606,
        refined_block_size=4,
    )
    weight_quantizer = make_quantizer(
        method="mbpriorq",
        name=None,
        model_type=args.model_type,
        ablation_mode="paper",
        random_seed=20260606,
        refined_block_size=4,
        using_imatrix=bool(args.imatrix),
        imatrix_file_name=args.imatrix,
    )
    smoke = streamed_layer_smoke(
        checkpoint=args.model,
        input_ids=input_ids,
        sequence_length=args.sequence_length,
        max_layers=args.layers,
        device=args.device,
        dtype=torch.bfloat16,
        model_family=args.model_family,
        weight_quantizer=weight_quantizer,
        activation_config=activation_config,
        progress=not args.quiet,
    )
    result = {
        "mode": "mbpriorq_streamed_layer_smoke",
        "status": "PASS",
        "model": args.model,
        "model_key": args.model_key,
        "dataset": args.dataset,
        "sequence_length": args.sequence_length,
        "requested_layers": smoke.requested_layers,
        "completed_layers": smoke.completed_layers,
        "total_layers": smoke.total_layers,
        "model_family": smoke.model_family,
        "quantized_weight_count": smoke.quantized_weight_count,
        "wrapped_linear_count": smoke.wrapped_linear_count,
        "imatrix_used": bool(args.imatrix),
        "weight_ebw_summary": GlobalEBW.summarize("weight"),
        "activation_ebw_summary": GlobalEBW.summarize("activation"),
        "layers": list(smoke.layers),
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
