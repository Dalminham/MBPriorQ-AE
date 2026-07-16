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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments/table2_ppl/models.json"
DEFAULT_IMATRIX = ROOT / "data/imatrix/Qwen_Qwen3-0.6B.imatrix"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument(
        "--model-map",
        help="JSON object mapping manifest keys to local checkpoint directories",
    )
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--output-root", default=str(ROOT / "local_runs/table2_ppl"))
    parser.add_argument("--imatrix", default=str(DEFAULT_IMATRIX))
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


def _valid_result(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload.get("perplexity"), (int, float))
        and int(payload.get("num_samples", 0)) > 0
    )


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
    states: dict[str, str] = {}
    _write_observation(observation, models, states)

    print(f"Preflight passed for {len(models)} models")
    for model in models:
        print(f"  {model['key']}: {paths[model['key']]}")
    if args.preflight_only:
        return

    runner = ROOT / "scripts/run_wikitext_ppl.py"
    mismatches: list[str] = []
    for model in models:
        for method in ("bf16", "mbpriorq"):
            state_key = f"{model['key']}:{method}"
            output = result_root / f"{model['key']}__{method}.json"
            if args.resume and _valid_result(output):
                states[state_key] = "complete (resumed)"
                _write_observation(observation, models, states)
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
    if mismatches:
        raise SystemExit(
            "Full-run PPL validation failed:\n- " + "\n- ".join(mismatches)
        )


if __name__ == "__main__":
    main()
