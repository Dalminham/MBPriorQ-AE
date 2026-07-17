#!/usr/bin/env python3
"""Validate completed Table 3 downstream outputs against recorded paper rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--model-keys", nargs="+", required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--tolerance-percentage-points", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    with Path(args.expected).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected = {
        (row["model_key"], row["method"], benchmark): float(row[benchmark])
        for row in rows
        for benchmark in args.benchmarks
        if row["model_key"] in args.model_keys
    }
    failures = []
    for key, paper_accuracy in expected.items():
        model_key, method, benchmark = key
        path = output_root / f"{model_key}__{method}__{benchmark}.json"
        if not path.is_file():
            failures.append(f"missing result: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = 100.0 * float(payload["accuracy"])
        difference = abs(observed - paper_accuracy)
        print(
            f"{model_key}/{method}/{benchmark}: observed={observed:.2f}% "
            f"paper={paper_accuracy:.2f}% diff={difference:.2f} pp"
        )
        if difference > args.tolerance_percentage_points:
            failures.append(
                f"{model_key}/{method}/{benchmark}: {difference:.2f} pp exceeds "
                f"{args.tolerance_percentage_points:.2f} pp"
            )
    if failures:
        raise SystemExit("Downstream validation failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
