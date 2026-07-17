#!/usr/bin/env python3
"""Validate generated hardware CSVs against curated expected results."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "hardware/expected"
FILES = {
    "modules": [
        "modules/scale_reconstructor.csv",
        "modules/packet_scheduler.csv",
        "modules/shared_fpu_pool.csv",
        "modules/multimsa_paths.csv",
        "modules/output_pair_join.csv",
    ],
    "system": [
        "system/external_1024_top.csv",
    ],
}


def normalized_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--scope", choices=("modules", "system", "all"), default="all")
    args = parser.parse_args()

    scopes = FILES if args.scope == "all" else {args.scope: FILES[args.scope]}
    failures = 0
    for files in scopes.values():
        for rel in files:
            expected = EXPECTED / rel
            actual = args.actual / rel
            if not actual.is_file():
                print(f"[FAIL] missing generated result: {actual}")
                failures += 1
                continue
            expected_lines = normalized_lines(expected)
            actual_lines = normalized_lines(actual)
            if expected_lines == actual_lines:
                print(f"[PASS] {rel}")
                continue
            print(f"[FAIL] {rel}")
            diff = difflib.unified_diff(
                expected_lines,
                actual_lines,
                fromfile=f"expected/{rel}",
                tofile=f"actual/{rel}",
                n=2,
            )
            for line in list(diff)[:40]:
                print(line)
            failures += 1

    if failures:
        print(f"Hardware validation failed for {failures} file(s).")
        return 1
    print("Hardware result validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
