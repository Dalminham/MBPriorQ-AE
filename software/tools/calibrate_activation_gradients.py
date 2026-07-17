#!/usr/bin/env python3
"""Collect activation-gradient magnitudes for Table 4 feature rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbpriorq_ae.perplexity import encode_dataset, load_text_dataset


TARGET_MARKERS = ("self_attn", "mlp", "lm_head", "block_sparse_moe")


class ActivationGradientCollector:
    def __init__(self, model: nn.Module):
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(marker in name for marker in TARGET_MARKERS):
                self.handles.append(
                    module.register_full_backward_hook(self._backward_hook(name))
                )

    def _backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            if not grad_input or grad_input[0] is None:
                return
            gradient = grad_input[0].detach()
            value = gradient.reshape(-1, gradient.shape[-1]).float().square().mean(dim=0).cpu()
            if name not in self.sums:
                self.sums[name] = torch.zeros_like(value)
                self.counts[name] = 0
            self.sums[name] += value
            self.counts[name] += 1

        return hook

    def result(self) -> dict[str, torch.Tensor]:
        return {
            name: total / self.counts[name]
            for name, total in self.sums.items()
            if self.counts[name] > 0
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True
    )
    dataset = load_text_dataset(args.dataset, args.split)
    input_ids = encode_dataset(tokenizer, dataset)
    available = input_ids.shape[1] // args.sequence_length
    samples = min(args.num_samples, available)
    if samples <= 0:
        raise ValueError("The calibration dataset contains no complete token window")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        use_safetensors=True,
        trust_remote_code=True,
    ).to(args.device)
    model.config.use_cache = False
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    collector = ActivationGradientCollector(model)
    for sample in range(samples):
        start = sample * args.sequence_length
        batch = input_ids[:, start : start + args.sequence_length].to(args.device)
        logits = model(input_ids=batch, use_cache=False).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]).float(),
            batch[:, 1:].reshape(-1),
        )
        loss.backward()
        model.zero_grad(set_to_none=True)
        print(f"[gradient] sample {sample + 1}/{samples}, loss={loss.item():.6f}")

    gradients = collector.result()
    collector.close()
    if not gradients:
        raise RuntimeError("No Linear-input gradients were collected")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gradients, output)
    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "sequence_length": args.sequence_length,
        "num_samples": samples,
        "gradient_checkpointing": args.gradient_checkpointing,
        "tensor_count": len(gradients),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
