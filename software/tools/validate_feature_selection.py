#!/usr/bin/env python3
"""Summarize and validate the Table 4 PPL rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FEATURES = ("std", "diff", "grad", "diff_grad", "std_grad")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.08)
    args = parser.parse_args()

    with Path(args.expected).open(newline="", encoding="utf-8") as handle:
        expected_rows = list(csv.DictReader(handle))
    results = Path(args.results)
    summary = []
    mismatches = []
    for model_key in ("qwen3_0_6b", "llama2_7b"):
        bf16 = json.loads((results / f"{model_key}__bf16.json").read_text(encoding="utf-8"))
        bf16_ppl = float(bf16["perplexity"])
        summary.append(
            {
                "model_key": model_key,
                "feature": "bf16",
                "ppl": bf16_ppl,
                "relative_ppl_increase_pct": 0.0,
                "num_samples": bf16["num_samples"],
            }
        )
        for feature in FEATURES:
            payload = json.loads(
                (results / f"{model_key}__{feature}.json").read_text(encoding="utf-8")
            )
            if payload.get("weight_source") != "none" or payload.get("quantized_weight_count") != 0:
                mismatches.append(f"{model_key}:{feature} is not activation-isolated W16A4")
            ppl = float(payload["perplexity"])
            relative = (ppl / bf16_ppl - 1.0) * 100.0
            summary.append(
                {
                    "model_key": model_key,
                    "feature": feature,
                    "ppl": ppl,
                    "relative_ppl_increase_pct": relative,
                    "num_samples": payload["num_samples"],
                }
            )

    if args.require_full:
        observed = {(row["model_key"], row["feature"]): row for row in summary}
        for expected in expected_rows:
            key = (expected["model_key"], expected["feature"])
            row = observed[key]
            if int(row["num_samples"]) != int(expected["expected_samples"]):
                mismatches.append(
                    f"{key} used {row['num_samples']} samples; expected {expected['expected_samples']}"
                )
            reference = float(expected["relative_ppl_increase_pct"])
            if abs(float(row["relative_ppl_increase_pct"]) - reference) > args.tolerance:
                mismatches.append(
                    f"{key} relative PPL increase {row['relative_ppl_increase_pct']:.4f}%, "
                    f"paper {reference:.2f}%"
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model_key",
                "feature",
                "ppl",
                "relative_ppl_increase_pct",
                "num_samples",
            ),
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote Table 4 PPL summary: {output}")
    if mismatches:
        raise SystemExit("Table 4 validation failed:\n- " + "\n- ".join(mismatches))


if __name__ == "__main__":
    main()
