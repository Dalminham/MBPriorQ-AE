#!/usr/bin/env python3
"""Run all BF16 and MBPriorQ rows in the paper's Table 2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "experiments/table2/models.json"
DEFAULT_IMATRIX = ROOT / "data/imatrix/Qwen_Qwen3-0.6B.imatrix"
DEFAULT_METADATA_EXPECTED = ROOT / "experiments/table2/expected_side_metadata.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument(
        "--model-map",
        help="JSON object mapping manifest keys to local checkpoint directories",
    )
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--output-root", default=str(ROOT / "local_runs/table2"))
    parser.add_argument("--imatrix", default=str(DEFAULT_IMATRIX))
    parser.add_argument("--metadata-expected", default=str(DEFAULT_METADATA_EXPECTED))
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument(
        "--ppl-tolerance",
        type=float,
        default=0.075,
        help="Allowed absolute PPL difference from the rounded paper value on full runs",
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    return parser.parse_args()


def _load_models(path: str, only: set[str]) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    models = payload["models"]
    if only:
        unknown = only - {entry["key"] for entry in models}
        if unknown:
            raise ValueError(f"Unknown model keys: {sorted(unknown)}")
        models = [entry for entry in models if entry["key"] in only]
    if not only and len(models) != 19:
        raise ValueError(f"Table 2 manifest contains {len(models)} models, expected 19")
    return models


def _load_overrides(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("--model-map must contain a JSON object")
    return {str(key): str(value) for key, value in payload.items()}


def _index_model_roots(roots: list[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root_text in roots:
        root = Path(root_text).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Model root not found: {root}")
        for config in root.rglob("config.json"):
            index.setdefault(config.parent.name, config.parent)
    return index


def _resolve_paths(models: list[dict], roots: list[str], overrides: dict[str, str]):
    index = _index_model_roots(roots)
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for model in models:
        if model["key"] in overrides:
            candidate = Path(overrides[model["key"]]).expanduser()
        else:
            candidate = next(
                (index[alias] for alias in model["aliases"] if alias in index),
                None,
            )
        if candidate is None or not (candidate / "config.json").is_file():
            missing.append(
                f"{model['key']} ({model['hub_id']}; aliases={model['aliases']})"
            )
        else:
            resolved[model["key"]] = candidate.resolve()
    if missing:
        raise FileNotFoundError(
            "Could not resolve all model checkpoints:\n- " + "\n- ".join(missing)
        )
    return resolved


def _write_observation(path: Path, models: list[dict], states: dict[str, str]) -> None:
    lines = [
        "# Table 2 PPL observation",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Model | BF16 | MBPriorQ |",
        "|---|---:|---:|",
    ]
    for model in models:
        lines.append(
            f"| {model['label']} | {states.get(model['key'] + ':bf16', 'pending')} "
            f"| {states.get(model['key'] + ':mbpriorq', 'pending')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _valid_result(path: Path, *, require_metadata: bool = False) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    valid = (
        isinstance(payload.get("perplexity"), (int, float))
        and int(payload.get("num_samples", 0)) > 0
    )
    if require_metadata:
        summary = payload.get("activation_ebw_summary")
        valid = valid and isinstance(summary, dict) and all(
            isinstance(summary.get(key), (int, float))
            for key in ("scale_extra_ebw", "mask_ebw", "effective_ebw")
        )
    return valid


def _write_side_metadata(
    path: Path,
    models: list[dict],
    result_root: Path,
    *,
    include_average: bool,
) -> list[dict]:
    rows = []
    for model in models:
        result_path = result_root / f"{model['key']}__mbpriorq.json"
        if not _valid_result(result_path, require_metadata=True):
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        summary = payload["activation_ebw_summary"]
        rows.append(
            {
                "model_key": model["key"],
                "model": model["label"],
                "scale_extra_ebw": float(summary["scale_extra_ebw"]),
                "mask_ebw": float(summary["mask_ebw"]),
                "total_ebw": float(summary["effective_ebw"]),
            }
        )
    if include_average and len(rows) == len(models):
        rows.append(
            {
                "model_key": "average",
                "model": "Average",
                "scale_extra_ebw": sum(row["scale_extra_ebw"] for row in rows) / len(rows),
                "mask_ebw": sum(row["mask_ebw"] for row in rows) / len(rows),
                "total_ebw": sum(row["total_ebw"] for row in rows) / len(rows),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model_key", "model", "scale_extra_ebw", "mask_ebw", "total_ebw"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _validate_side_metadata(rows: list[dict], expected_path: str, tolerance: float) -> list[str]:
    with Path(expected_path).open(newline="", encoding="utf-8") as handle:
        expected = {row["model_key"]: row for row in csv.DictReader(handle)}
    observed = {row["model_key"]: row for row in rows}
    mismatches = []
    for model_key, expected_row in expected.items():
        if model_key not in observed:
            mismatches.append(f"side metadata missing row {model_key}")
            continue
        for field in ("scale_extra_ebw", "mask_ebw", "total_ebw"):
            value = float(observed[model_key][field])
            reference = float(expected_row[field])
            if abs(value - reference) > tolerance:
                mismatches.append(
                    f"{model_key}:{field} observed {value:.6f}, paper {reference:.4f}"
                )
    return mismatches


def main():
    args = parse_args()
    models = _load_models(args.manifest, set(args.only))
    roots = list(args.model_root)
    env_roots = os.environ.get("MODEL_ROOTS", "")
    if env_roots:
        roots.extend(item for item in env_roots.split(os.pathsep) if item)
    overrides = _load_overrides(args.model_map)
    paths = _resolve_paths(models, roots, overrides)
    if any(model.get("requires_imatrix") for model in models):
        if not Path(args.imatrix).is_file():
            raise FileNotFoundError(
                "Qwen3-0.6B requires the paper imatrix; missing " + args.imatrix
            )

    output_root = Path(args.output_root)
    result_root = output_root / "results"
    log_root = output_root / "logs"
    result_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    observation = output_root / "observation.md"
    metadata_output = output_root / "side_metadata_overhead.csv"
    states: dict[str, str] = {}
    _write_observation(observation, models, states)

    print(f"Preflight passed for {len(models)} models")
    for model in models:
        print(f"  {model['key']}: {paths[model['key']]}")
    if args.preflight_only:
        return

    runner = ROOT / "software/tools/run_wikitext_ppl.py"
    mismatches: list[str] = []
    for model in models:
        for method in ("bf16", "mbpriorq"):
            state_key = f"{model['key']}:{method}"
            output = result_root / f"{model['key']}__{method}.json"
            if args.resume and _valid_result(
                output, require_metadata=(method == "mbpriorq")
            ):
                states[state_key] = "complete (resumed)"
                _write_observation(observation, models, states)
                _write_side_metadata(
                    metadata_output,
                    models,
                    result_root,
                    include_average=(not args.only and len(models) == 19),
                )
                continue
            states[state_key] = "running"
            _write_observation(observation, models, states)
            command = [
                sys.executable,
                str(runner),
                "--model",
                str(paths[model["key"]]),
                "--tokenizer",
                str(paths[model["key"]]),
                "--model-key",
                model["key"],
                "--dataset",
                args.dataset,
                "--method",
                method,
                "--backend",
                model["backend"],
                "--model-family",
                model.get("model_family", "auto"),
                "--model-type",
                model["model_type"],
                "--weight-source",
                "online" if method == "mbpriorq" else "auto",
                "--sequence-length",
                str(args.sequence_length),
                "--num-samples",
                str(args.num_samples),
                "--device",
                args.device,
                "--output",
                str(output),
                "--quiet",
            ]
            if method == "mbpriorq" and model.get("requires_imatrix"):
                command.extend(("--imatrix", args.imatrix))
            log_path = log_root / f"{model['key']}__{method}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0:
                states[state_key] = f"failed (see {log_path.name})"
                _write_observation(observation, models, states)
                raise SystemExit(
                    f"{state_key} failed with code {completed.returncode}; see {log_path}"
                )
            observed = json.loads(output.read_text(encoding="utf-8"))["perplexity"]
            expected = model[f"{method}_ppl"]
            difference = abs(observed - expected)
            if args.num_samples == 0 and difference > args.ppl_tolerance:
                states[state_key] = (
                    f"mismatch: {observed:.4f} (paper {expected:.2f}, "
                    f"diff {difference:.4f})"
                )
                mismatches.append(
                    f"{state_key}: observed {observed:.6f}, paper {expected:.2f}, "
                    f"difference {difference:.6f}"
                )
            else:
                states[state_key] = f"{observed:.4f} (paper {expected:.2f})"
            _write_observation(observation, models, states)
            if method == "mbpriorq":
                _write_side_metadata(
                    metadata_output,
                    models,
                    result_root,
                    include_average=(not args.only and len(models) == 19),
                )

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "method", "backend", "observed_ppl", "paper_ppl"))
        for model in models:
            for method in ("bf16", "mbpriorq"):
                path = result_root / f"{model['key']}__{method}.json"
                observed = json.loads(path.read_text(encoding="utf-8"))["perplexity"]
                writer.writerow(
                    (model["label"], method, model["backend"], observed, model[f"{method}_ppl"])
                )
    print(f"Completed {len(models) * 2} rows; summary: {summary_path}")
    metadata_rows = _write_side_metadata(
        metadata_output,
        models,
        result_root,
        include_average=(not args.only and len(models) == 19),
    )
    print(f"Table 10 companion output: {metadata_output}")
    if args.num_samples == 0 and not args.only and len(models) == 19:
        mismatches.extend(
            _validate_side_metadata(metadata_rows, args.metadata_expected, tolerance=0.00011)
        )
    if mismatches:
        raise SystemExit(
            "Full-run PPL validation failed:\n- " + "\n- ".join(mismatches)
        )


if __name__ == "__main__":
    main()
