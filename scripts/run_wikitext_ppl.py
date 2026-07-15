#!/usr/bin/env python3
"""Run one paper-compatible Qwen/Llama WikiText2 PPL row."""

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
from mbpriorq_ae.integration import ActivationQuantizationConfig, wrap_activation_linears
from mbpriorq_ae.logging import set_log_level
from mbpriorq_ae.perplexity import encode_dataset, load_text_dataset, paper_compatible_by_layer_ppl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Source or fake-quantized HF checkpoint")
    parser.add_argument("--tokenizer", help="Tokenizer path/id; defaults to --model")
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", required=True, choices=("bf16", "mbpriorq"))
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
    return metadata


def main():
    args = parse_args()
    set_log_level("release")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_metadata = _validate_checkpoint(args)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("PPL evaluation requires transformers") from error

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        use_safetensors=True,
        trust_remote_code=True,
    )
    model.eval()
    GlobalEBW.reset()
    wrapped: list[str] = []
    if args.method != "bf16":
        wrapped = wrap_activation_linears(
            model,
            ActivationQuantizationConfig(
                method=args.method,
                model_type=args.model_type,
                ablation_mode=args.ablation_mode,
                random_seed=args.random_seed,
                refined_block_size=args.refined_block_size,
            ),
        )

    dataset = load_text_dataset(args.dataset, args.split)
    input_ids = encode_dataset(tokenizer, dataset)
    ppl, windows = paper_compatible_by_layer_ppl(
        model,
        input_ids,
        sequence_length=args.sequence_length,
        num_samples=args.num_samples,
        device=args.device,
        progress=not args.quiet,
    )
    result = {
        "method": args.method,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "dtype": "bfloat16",
        "sequence_length": args.sequence_length,
        "num_samples": windows,
        "perplexity": ppl,
        "model_type": args.model_type if args.method == "mbpriorq" else None,
        "ablation_mode": args.ablation_mode if args.method == "mbpriorq" else None,
        "refined_block_size": args.refined_block_size if args.method == "mbpriorq" else None,
        "wrapped_linear_count": len(wrapped),
        "activation_ebw_summary": GlobalEBW.summarize("activation"),
        "weight_ebw_summary": (
            checkpoint_metadata.get("weight_ebw_summary") if checkpoint_metadata else None
        ),
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
