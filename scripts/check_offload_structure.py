#!/usr/bin/env python3
"""Check streamed checkpoint/model-family mapping without loading real weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software"))

import torch
from transformers import AutoConfig

from mbpriorq_ae.offload import (
    build_execution_plan,
    load_safetensors_index,
    validate_stream_structure,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-family",
        default="auto",
        choices=("auto", "decoder", "qwen3_vl_language"),
    )
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    plan = build_execution_plan(config, torch.bfloat16, args.model_family)
    index = load_safetensors_index(args.model)
    summary = validate_stream_structure(plan, index["weight_map"])
    print(json.dumps({"model_family": plan.family, "prefixes": summary}, indent=2))


if __name__ == "__main__":
    main()
