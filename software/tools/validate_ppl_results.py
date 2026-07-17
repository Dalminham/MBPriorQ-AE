#!/usr/bin/env python3
"""Validate full-paper PPL rows or basic invariants for a quick run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from validation_common import finite_float, positive_int, require_fields


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--expected-samples", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.results)
    with Path(args.expected).open(newline="", encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))

    failures: list[str] = []
    observed: dict[str, dict] = {}
    for row in expected:
        name = row["row"]
        path = result_dir / f"{name}.json"
        if not path.is_file():
            failures.append(f"missing result {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            require_fields(data, ("method", "num_samples", "perplexity", "total_nll"), name)
            positive_int(data["num_samples"], f"{name}:num_samples", args.expected_samples)
            finite_float(data["perplexity"], f"{name}:perplexity")
            finite_float(data["total_nll"], f"{name}:total_nll")
        except (TypeError, ValueError) as error:
            failures.append(str(error))
            continue
        observed[name] = data
        if args.require_full or data["num_samples"] == int(row["num_samples"]):
            error = abs(float(data["perplexity"]) - float(row["perplexity"]))
            if error > float(row["tolerance"]):
                failures.append(
                    f"{name}: PPL {data['perplexity']:.6f}, expected "
                    f"{float(row['perplexity']):.6f} +/- {row['tolerance']}"
                )

    if {"bf16", "mbpriorq"}.issubset(observed):
        bf16 = observed["bf16"]["perplexity"]
        mbpriorq = observed["mbpriorq"]["perplexity"]
        if not bf16 < mbpriorq:
            failures.append(
                f"expected BF16 < MBPriorQ, observed {bf16:.6f}, {mbpriorq:.6f}"
            )
        summary = observed["mbpriorq"].get("activation_ebw_summary")
        if not summary or abs(float(summary["mask_ebw"]) - 0.0625) > 1e-12:
            failures.append("MBPriorQ activation mask EBW is not 0.0625")

    if failures:
        raise SystemExit("PPL validation failed:\n- " + "\n- ".join(failures))
    print(f"PPL validation passed for {len(observed)} rows")


if __name__ == "__main__":
    main()
