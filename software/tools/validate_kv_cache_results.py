#!/usr/bin/env python3
"""Summarize and validate the Table 8 KV-cache rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from validation_common import finite_float, positive_int, require_fields


METHODS = ("bf16", "nvfp4", "mbpriorq")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--ppl-tolerance", type=float, default=0.001)
    parser.add_argument("--ebw-tolerance", type=float, default=0.00011)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--model-key", action="append", default=[])
    args = parser.parse_args()

    with Path(args.expected).open(newline="", encoding="utf-8") as handle:
        expected = {
            (row["model_key"], row["method"]): row for row in csv.DictReader(handle)
        }
    rows = []
    mismatches = []
    models = args.model_key or ["qwen3_0_6b", "llama2_7b"]
    for model_key in models:
        for method in METHODS:
            path = Path(args.results) / f"{model_key}__{method}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            require_fields(payload, ("method", "kv_cache_method", "num_samples", "perplexity", "total_nll"), f"{model_key}:{method}")
            positive_int(payload["num_samples"], f"{model_key}:{method}:chunks", args.expected_samples)
            ppl = finite_float(payload["perplexity"], f"{model_key}:{method}:perplexity")
            finite_float(payload["total_nll"], f"{model_key}:{method}:total_nll")
            if payload["kv_cache_method"] != method:
                mismatches.append(f"{model_key}:{method} KV-cache method metadata mismatch")
            if payload.get("method") != "bf16" or payload.get("quantized_weight_count") != 0:
                mismatches.append(f"{model_key}:{method} does not keep Linear weights BF16")
            summary = payload.get("kv_cache_ebw_summary")
            ebw = "" if summary is None else float(summary["effective_ebw"])
            row = {
                "model_key": model_key,
                "setting": "W16A16KV16" if method == "bf16" else "W16A16KV4",
                "method": method,
                "ppl": ppl,
                "effective_ebw": ebw,
                "chunks": int(payload["num_samples"]),
            }
            rows.append(row)
            if args.require_full:
                reference = expected[(model_key, method)]
                if abs(row["ppl"] - float(reference["ppl"])) > args.ppl_tolerance:
                    mismatches.append(
                        f"{model_key}:{method} PPL {row['ppl']:.6f}, paper {reference['ppl']}"
                    )
                if reference["effective_ebw"]:
                    if abs(float(ebw) - float(reference["effective_ebw"])) > args.ebw_tolerance:
                        mismatches.append(
                            f"{model_key}:{method} EBW {float(ebw):.6f}, "
                            f"paper {reference['effective_ebw']}"
                        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model_key", "setting", "method", "ppl", "effective_ebw", "chunks"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote Table 8 summary: {output}")
    if mismatches:
        raise SystemExit("Table 8 validation failed:\n- " + "\n- ".join(mismatches))


if __name__ == "__main__":
    main()
