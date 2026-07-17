#!/usr/bin/env python3
"""Validate paper PPL/EBW ablation rows from JSON result files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--require-full", action="store_true")
    return parser.parse_args()


def _check_value(failures, row_name, label, observed, expected, tolerance):
    if expected == "":
        return
    if observed is None:
        failures.append(f"{row_name}: missing {label}")
        return
    error = abs(float(observed) - float(expected))
    if error > float(tolerance):
        failures.append(
            f"{row_name}: {label}={float(observed):.6f}, expected "
            f"{float(expected):.6f} +/- {tolerance}"
        )


def main():
    args = parse_args()
    result_dir = Path(args.results)
    with Path(args.expected).open(newline="", encoding="utf-8") as handle:
        expected_rows = list(csv.DictReader(handle))

    failures: list[str] = []
    for row in expected_rows:
        name = row["row"]
        path = result_dir / row.get("result_file", f"{name}.json")
        if not path.is_file():
            failures.append(f"missing result {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        full_row = int(data["num_samples"]) == int(row["num_samples"])
        if args.require_full and not full_row:
            failures.append(
                f"{name}: num_samples={data['num_samples']}, expected {row['num_samples']}"
            )
        if args.require_full or full_row:
            _check_value(
                failures,
                name,
                "perplexity",
                data.get("perplexity"),
                row.get("perplexity", ""),
                row.get("ppl_tolerance", "0"),
            )
        activation = data.get("activation_ebw_summary") or {}
        weight = data.get("weight_ebw_summary") or {}
        if args.require_full or full_row:
            _check_value(
                failures,
                name,
                "activation_effective_ebw",
                activation.get("effective_ebw"),
                row.get("activation_effective_ebw", ""),
                row.get("ebw_tolerance", "0"),
            )
            _check_value(
                failures,
                name,
                "weight_effective_ebw",
                weight.get("effective_ebw"),
                row.get("weight_effective_ebw", ""),
                row.get("ebw_tolerance", "0"),
            )

    if failures:
        raise SystemExit("Ablation validation failed:\n- " + "\n- ".join(failures))
    print(f"Ablation validation passed for {len(expected_rows)} rows")


if __name__ == "__main__":
    main()
