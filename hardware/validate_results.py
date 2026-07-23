#!/usr/bin/env python3
"""Validate hardware behavior and compare results with golden CSVs."""

from __future__ import annotations

import argparse
import csv
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
        "system/system_summary.csv",
    ],
}


def normalized_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_modules(actual: Path) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []

    scale = csv_rows(actual / "modules/scale_reconstructor.csv")
    scale_ok = (
        len(scale) == 4
        and {(row["weight_mask"], row["activation_mask"]) for row in scale}
        == {("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")}
        and all(
            row["expected_valid_sub_blocks"] == row["actual_valid_sub_blocks"]
            and row["expected_dequant_factors_fp32"]
            == row["actual_dequant_factors_fp32"]
            and row["status"] == "PASS"
            for row in scale
        )
    )
    checks.append((scale_ok, "Scale reconstruction: all four weight/activation mask combinations produced the expected FP32 dequant factors"))

    scheduler = csv_rows(actual / "modules/packet_scheduler.csv")
    scheduler_ok = (
        len(scheduler) == 4
        and sorted(int(row["block_index"]) for row in scheduler) == list(range(4))
        and sorted(int(row["committed_output_rank"]) for row in scheduler) == [1, 2, 3, 4]
        and all(
            row["vpu_precedes_multimsa"] == "true"
            and row["expected_output_beats"] == row["actual_output_beats"]
            and row["status"] == "PASS"
            for row in scheduler
        )
    )
    checks.append((scheduler_ok, "Packet scheduler: VPU work issued before MultiMSA data readiness, while out-of-order completions committed in block order"))

    pool = csv_rows(actual / "modules/shared_fpu_pool.csv")
    pool_ok = (
        len(pool) == 8
        and sorted(int(row["block_index"]) for row in pool) == list(range(8))
        and all(
            0 < int(row["max_busy_fpus"]) <= int(row["pool_size"])
            and row["total_accepted_blocks"] == "8"
            and row["total_completed_blocks"] == "8"
            and row["status"] == "PASS"
            for row in pool
        )
    )
    checks.append((pool_ok, "Shared FPU pool: eight mixed regular/refined blocks completed once without exceeding the 16-FPU pool"))

    multimsa = csv_rows(actual / "modules/multimsa_paths.csv")
    by_path = {row["path"]: row for row in multimsa}
    multimsa_ok = (
        set(by_path) == {"regular", "refined"}
        and by_path["regular"]["actual_sub_block_indices"] == "[0]"
        and by_path["regular"]["actual_partial_count"] == "1"
        and by_path["refined"]["actual_sub_block_indices"] == "[0,1,2,3]"
        and by_path["refined"]["actual_partial_count"] == "4"
        and int(by_path["refined"]["first_partial_cycle"])
        < int(by_path["regular"]["first_partial_cycle"])
        and all(row["status"] == "PASS" for row in multimsa)
    )
    checks.append((multimsa_ok, "MultiMSA: the regular path emitted one partial matrix and the refined path emitted four indexed partial matrices"))

    joined = csv_rows(actual / "modules/output_pair_join.csv")
    join_ok = (
        len(joined) == 6
        and [int(row["actual_output_rank"]) for row in joined] == list(range(1, 7))
        and all(
            row["expected_output_rank"] == row["actual_output_rank"]
            and int(row["output_cycle"]) >= int(row["both_inputs_ready_cycle"])
            and row["matrix_match"] == "true"
            and row["scale_match"] == "true"
            and row["status"] == "PASS"
            for row in joined
        )
    )
    checks.append((join_ok, "Output join: six matrix/dequant-factor pairs were released only after both inputs were ready and in block/sub-block order"))
    return checks


def validate_system(actual: Path) -> list[tuple[bool, str]]:
    trace = csv_rows(actual / "system/external_1024_top.csv")
    summary = csv_rows(actual / "system/system_summary.csv")
    blocks = [row for row in summary if row["block_index"] != "TOTAL"]
    total = [row for row in summary if row["block_index"] == "TOTAL"]

    shape_ok = (
        len(blocks) == 16
        and len(total) == 1
        and sum(row["path"] == "regular" for row in blocks) == 5
        and sum(row["path"] == "refined" for row in blocks) == 11
        and sum(int(row["matrix_scale_pairs"]) for row in blocks) == 49
        and sum(int(row["output_packets"]) for row in blocks) == 245
        and all(
            row["matrix_matches"] == f'{row["matrix_scale_pairs"]}/{row["matrix_scale_pairs"]}'
            and row["scale_matches"] == f'{row["matrix_scale_pairs"]}/{row["matrix_scale_pairs"]}'
            and row["status"] == "PASS"
            for row in blocks
        )
        and total[0]["matrix_scale_pairs"] == "49"
        and total[0]["matrix_matches"] == "49/49"
        and total[0]["scale_matches"] == "49/49"
        and total[0]["output_packets"] == "245"
        and total[0]["status"] == "PASS"
    )

    groups = [trace[idx : idx + 5] for idx in range(0, len(trace), 5)]
    packet_ok = len(trace) == 245 and len(groups) == 49
    if packet_ok:
        packet_ok = all(
            len(group) == 5
            and len({(row["block_index"], row["sub_block_idx"]) for row in group}) == 1
            and [int(row["segment_idx"]) for row in group] == [0, 1, 2, 3, 4]
            and [row["last"] for row in group] == ["false", "false", "false", "false", "true"]
            and all(row["packet_type"] == "0x80" for row in group)
            for group in groups
        )

    return [
        (shape_ok, "Complete packet top: 16 logical blocks (5 regular, 11 refined) produced 49 matrix-scale pairs; all matrices and scales matched references"),
        (packet_ok, "Packetization and backpressure: 49 matrix-scale pairs remained ordered as 245 valid five-segment output packets"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--scope", choices=("modules", "system", "all"), default="all")
    args = parser.parse_args()

    scopes = FILES if args.scope == "all" else {args.scope: FILES[args.scope]}
    failures = 0

    semantic_scopes = scopes.keys()
    for scope in semantic_scopes:
        required = [args.actual / rel for rel in FILES[scope]]
        if not all(path.is_file() for path in required):
            continue
        try:
            checks = validate_modules(args.actual) if scope == "modules" else validate_system(args.actual)
        except (OSError, KeyError, ValueError, IndexError) as error:
            print(f"[FAIL] {scope} semantic validation could not read the generated CSVs: {error}")
            failures += 1
            continue
        for passed, description in checks:
            print(f"[{'PASS' if passed else 'FAIL'}] {description}")
            if not passed:
                failures += 1

    print("\nGolden-result comparison:")
    for files in scopes.values():
        for rel in files:
            expected = EXPECTED / rel
            actual = args.actual / rel
            if not actual.is_file():
                print(f"[FAIL] missing generated result: {actual}")
                failures += 1
                continue
            if not expected.is_file():
                print(f"[FAIL] missing golden result: {expected}")
                failures += 1
                continue
            expected_lines = normalized_lines(expected)
            actual_lines = normalized_lines(actual)
            if expected_lines == actual_lines:
                print(f"[PASS] golden match: {rel}")
                continue
            print(f"[FAIL] golden mismatch: {rel}")
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
        print(f"Hardware validation failed with {failures} error(s).")
        return 1
    print("\nHardware functional and golden-result validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
