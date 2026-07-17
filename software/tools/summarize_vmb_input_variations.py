#!/usr/bin/env python3
"""Summarize and validate the paper's Table 7 VMB-prior rows."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from itertools import combinations
from pathlib import Path


FULL_TAGS = (
    "baseline_wt2",
    "domain_ptb",
    "prompt_mmlu_pro",
    "context_512",
    "context_1024",
    "context_4096",
    "segment_seed0",
    "segment_seed1",
    "segment_seed2",
    "segment_seed3",
    "segment_seed4",
    "batch_2",
    "batch_4",
)


def _read_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = payload.get("vmb_profile_summary")
    if payload.get("method") != "mbpriorq" or not isinstance(profile, dict):
        raise ValueError(f"Not an MBPriorQ VMB-profile result: {path}")
    if int(profile.get("prior_records", 0)) <= 0:
        raise ValueError(f"VMB profile contains no prior-phase records: {path}")
    return payload


def _miss_percent(payload: dict) -> float:
    return 100.0 * float(payload["vmb_profile_summary"]["full_miss_rate"])


def _decode_mask(value: str, width: int) -> int:
    if not value:
        raise ValueError("VMB profile row has an empty selected-column mask")
    mask = int(value, 16)
    if mask.bit_length() > width:
        raise ValueError(f"Serialized mask exceeds declared width {width}")
    return mask


def _layer_masks(path: Path, *, union_all_calls: bool) -> dict[str, tuple[int, int]]:
    masks: dict[str, tuple[int, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not union_all_calls and row.get("phase") != "calibration":
                continue
            layer = row["layer_name"]
            width = int(float(row["microblock_columns"]))
            mask = _decode_mask(row["selected_column_hex"], width)
            if layer in masks:
                previous, previous_width = masks[layer]
                if previous_width != width:
                    raise ValueError(f"Mask width changes within layer {layer}")
                mask |= previous
            masks[layer] = (mask, width)
    if not masks:
        raise ValueError(f"No comparable selected-column masks in {path}")
    return masks


def _aggregate_jaccard(
    left: dict[str, tuple[int, int]], right: dict[str, tuple[int, int]]
) -> float:
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("VMB profiles have no common layers")
    intersection = 0
    union = 0
    for layer in common:
        left_mask, left_width = left[layer]
        right_mask, right_width = right[layer]
        if left_width != right_width:
            raise ValueError(f"Mask width mismatch for {layer}")
        intersection += (left_mask & right_mask).bit_count()
        union += (left_mask | right_mask).bit_count()
    return intersection / union if union else 1.0


def _pairwise_summary(paths: list[Path], *, union_all_calls: bool) -> tuple[float, float]:
    indexed = [_layer_masks(path, union_all_calls=union_all_calls) for path in paths]
    values = [
        _aggregate_jaccard(indexed[left], indexed[right])
        for left, right in combinations(range(len(indexed)), 2)
    ]
    if not values:
        raise ValueError("At least two VMB profiles are required for mask overlap")
    return statistics.fmean(values), min(values)


def _load_expected(path: str) -> dict[str, dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["model_key"]: row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output", required=True)
    parser.add_argument("--model-key", action="append", default=[])
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--miss-tolerance-pct", type=float, default=0.10)
    parser.add_argument("--overlap-tolerance", type=float, default=0.005)
    args = parser.parse_args()

    results_root = Path(args.results)
    profiles_root = Path(args.profiles)
    models = args.model_key or ["qwen3_0_6b", "llama2_7b"]
    expected = _load_expected(args.expected)
    details = []
    summary = []
    mismatches = []

    for model_key in models:
        available = {
            path.stem.removeprefix(f"{model_key}__"): path
            for path in results_root.glob(f"{model_key}__*.json")
        }
        if args.require_full:
            missing = sorted(set(FULL_TAGS) - set(available))
            if missing:
                mismatches.append(f"{model_key} missing runs: {', '.join(missing)}")
                continue
        payloads = {tag: _read_result(path) for tag, path in available.items()}
        for tag, payload in sorted(payloads.items()):
            details.append(
                {
                    "model_key": model_key,
                    "run": tag,
                    "dataset": payload["dataset"],
                    "sequence_length": payload["sequence_length"],
                    "batch_size": payload["batch_size"],
                    "num_samples": payload["num_samples"],
                    "ppl": payload["perplexity"],
                    "vmb_miss_rate_pct": _miss_percent(payload),
                    "prior_records": payload["vmb_profile_summary"]["prior_records"],
                }
            )

        required_for_summary = {
            "baseline_wt2",
            "domain_ptb",
            "prompt_mmlu_pro",
            "context_512",
            "context_4096",
            "batch_2",
            "batch_4",
        }
        segment_tags = sorted(tag for tag in payloads if tag.startswith("segment_seed"))
        if not required_for_summary.issubset(payloads) or len(segment_tags) < 2:
            continue

        segment_profiles = [
            profiles_root / f"{model_key}__{tag}.csv" for tag in segment_tags
        ]
        batch_profiles = [
            profiles_root / f"{model_key}__{tag}.csv"
            for tag in ("baseline_wt2", "batch_2", "batch_4")
        ]
        segment_mean, segment_min = _pairwise_summary(
            segment_profiles, union_all_calls=False
        )
        batch_mean, batch_min = _pairwise_summary(
            batch_profiles, union_all_calls=True
        )
        row = {
            "model_key": model_key,
            "domain_wt2_miss_pct": _miss_percent(payloads["baseline_wt2"]),
            "domain_ptb_miss_pct": _miss_percent(payloads["domain_ptb"]),
            "prompt_miss_pct": _miss_percent(payloads["prompt_mmlu_pro"]),
            "context_512_miss_pct": _miss_percent(payloads["context_512"]),
            "context_4096_miss_pct": _miss_percent(payloads["context_4096"]),
            "segment_overlap_mean": segment_mean,
            "segment_overlap_min": segment_min,
            "batch_overlap_mean": batch_mean,
            "batch_overlap_min": batch_min,
        }
        summary.append(row)

        if args.require_full:
            reference = expected[model_key]
            for field in (
                "domain_wt2_miss_pct",
                "domain_ptb_miss_pct",
                "prompt_miss_pct",
                "context_512_miss_pct",
                "context_4096_miss_pct",
            ):
                if abs(float(row[field]) - float(reference[field])) > args.miss_tolerance_pct:
                    mismatches.append(
                        f"{model_key}:{field} observed {row[field]:.4f}, "
                        f"paper {float(reference[field]):.4f}"
                    )
            for field in (
                "segment_overlap_mean",
                "segment_overlap_min",
                "batch_overlap_mean",
                "batch_overlap_min",
            ):
                if abs(float(row[field]) - float(reference[field])) > args.overlap_tolerance:
                    mismatches.append(
                        f"{model_key}:{field} observed {row[field]:.6f}, "
                        f"paper {float(reference[field]):.6f}"
                    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model_key",
        "domain_wt2_miss_pct",
        "domain_ptb_miss_pct",
        "prompt_miss_pct",
        "context_512_miss_pct",
        "context_4096_miss_pct",
        "segment_overlap_mean",
        "segment_overlap_min",
        "batch_overlap_mean",
        "batch_overlap_min",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    details_output = Path(args.details_output)
    details_output.parent.mkdir(parents=True, exist_ok=True)
    detail_fields = (
        "model_key",
        "run",
        "dataset",
        "sequence_length",
        "batch_size",
        "num_samples",
        "ppl",
        "vmb_miss_rate_pct",
        "prior_records",
    )
    with details_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(details)

    print(f"Wrote Table 7 summary: {output}")
    if mismatches:
        raise SystemExit("Table 7 validation failed:\n- " + "\n- ".join(mismatches))


if __name__ == "__main__":
    main()
