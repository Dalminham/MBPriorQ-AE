#!/usr/bin/env python3
"""Build an AE fake-quantized checkpoint from public Hugging Face weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software"))

from mbpriorq_ae.checkpoint import stream_fake_quantize_checkpoint
from mbpriorq_ae.logging import set_log_level


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument(
        "--model-key",
        required=True,
        help="Stable experiment key; qwen3_0_6b has a mandatory imatrix policy",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True, choices=("mbpriorq",))
    parser.add_argument("--refined-block-size", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--imatrix")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_log_level("release")
    if args.model_key == "qwen3_0_6b" and not args.imatrix:
        raise ValueError("Qwen3-0.6B checkpoint generation requires --imatrix")
    if args.model_key != "qwen3_0_6b" and args.imatrix:
        raise ValueError("Only Qwen3-0.6B may use the bundled imatrix")
    stats = stream_fake_quantize_checkpoint(
        source_path=args.source_model,
        output_path=args.output,
        method=args.method,
        refined_block_size=args.refined_block_size,
        using_imatrix=bool(args.imatrix),
        imatrix_file_name=args.imatrix,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
