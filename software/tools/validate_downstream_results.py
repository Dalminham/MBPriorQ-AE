#!/usr/bin/env python3
"""Validate completed Table 3 downstream outputs against recorded paper rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from validation_common import finite_float, positive_int, require_fields


FULL_COUNTS = {"gsm8k": 500, "mmlu": 100, "mmlu_pro": 410}
FULL_TOLERANCE_PERCENTAGE_POINTS = {
    "gsm8k": 1.0,
    "mmlu": 2.0,
    "mmlu_pro": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--model-keys", nargs="+", required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--expected-examples", type=int)
    parser.add_argument("--require-full", action="store_true")
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
        try:
            require_fields(
                payload,
                ("benchmark", "method", "num_examples", "correct", "accuracy", "records_journal"),
                f"{model_key}:{method}:{benchmark}",
            )
            if payload["benchmark"] != benchmark or payload["method"] != method:
                raise ValueError(
                    f"{model_key}:{method}:{benchmark}: benchmark/method metadata mismatch"
                )
            expected_count = args.expected_examples
            if args.require_full:
                expected_count = FULL_COUNTS[benchmark]
            count = positive_int(
                payload["num_examples"],
                f"{model_key}:{method}:{benchmark}:num_examples",
                expected_count,
            )
            correct = int(payload["correct"])
            accuracy = finite_float(
                payload["accuracy"], f"{model_key}:{method}:{benchmark}:accuracy"
            )
            if correct < 0 or correct > count or not 0.0 <= accuracy <= 1.0:
                raise ValueError(
                    f"{model_key}:{method}:{benchmark}: invalid correct/accuracy values"
                )
            if abs(accuracy - correct / count) > 1e-12:
                raise ValueError(
                    f"{model_key}:{method}:{benchmark}: accuracy is inconsistent with correct/count"
                )
            journal = path.parent / payload["records_journal"]
            records = [
                json.loads(line)
                for line in journal.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(records) != count or [row.get("index") for row in records] != list(range(count)):
                raise ValueError(
                    f"{model_key}:{method}:{benchmark}: journal rows are incomplete or non-contiguous"
                )
            for index, record in enumerate(records):
                require_fields(
                    record,
                    ("index", "prediction", "gold", "correct", "response"),
                    f"{model_key}:{method}:{benchmark}:record{index}",
                )
        except (FileNotFoundError, TypeError, ValueError) as error:
            failures.append(str(error))
            continue
        observed = 100.0 * accuracy
        difference = abs(observed - paper_accuracy)
        tolerance = FULL_TOLERANCE_PERCENTAGE_POINTS[benchmark]
        if args.require_full:
            print(
                f"{model_key}/{method}/{benchmark}: observed={observed:.2f}% "
                f"paper={paper_accuracy:.2f}% diff={difference:.2f} pp "
                f"tolerance={tolerance:.2f} pp"
            )
        else:
            print(
                f"{model_key}/{method}/{benchmark}: validated {count} example(s)"
            )
        if args.require_full and difference > tolerance:
            failures.append(
                f"{model_key}/{method}/{benchmark}: {difference:.2f} pp exceeds "
                f"{tolerance:.2f} pp"
            )
    if failures:
        raise SystemExit("Downstream validation failed:\n- " + "\n- ".join(failures))
    print(f"Downstream validation passed for {len(expected)} rows")


if __name__ == "__main__":
    main()
