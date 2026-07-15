#!/usr/bin/env python3
"""Generate and validate deterministic MBPriorQ smoke metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from mbpriorq_ae import GlobalEBW, MBPriorQ_Quantizer
from mbpriorq_ae.logging import RELEASE, set_log_level


ROOT = Path(__file__).resolve().parents[1]


def quantizer_args(**overrides) -> dict:
    args = {
        "name": "ae.smoke",
        "device": "cpu",
        "quant_bit": 4,
        "quant_sym": True,
        "model_type": "cloud",
        "ablation_mode": "paper",
        "random_seed": 20260606,
        "refined_block_size": 4,
        "using_imatrix": False,
        "imatrix_file_name": "unused",
        "vmb_profile_enable": False,
        "metadata_target": "activation",
    }
    args.update(overrides)
    return args


def reset_activation_ebw() -> None:
    GlobalEBW.activation_vmb_blocks = 0
    GlobalEBW.activation_total_blocks = 0
    GlobalEBW.configure_mbpriorq_refined_block_size(4)


def tensor_metrics(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.float()
    return {
        "sum": float(values.sum()),
        "abs_sum": float(values.abs().sum()),
        "squared_sum": float(values.square().sum()),
    }


def produce() -> dict:
    set_log_level(RELEASE)
    torch.manual_seed(1308)
    calibration = torch.randn(8, 64)
    later = torch.randn(8, 64)

    reset_activation_ebw()
    mbpriorq = MBPriorQ_Quantizer(quantizer_args())
    calibration_out = mbpriorq.fake_quantize_activation(
        calibration, name="layer", tensor_shape=calibration.shape
    )
    prior_out = mbpriorq.fake_quantize_activation(
        later, name="layer", tensor_shape=later.shape
    )
    ebw = GlobalEBW.summarize("activation")

    return {
        "schema_version": 1,
        "seed": 1308,
        "input_shape": list(calibration.shape),
        "mbpriorq_calibration": tensor_metrics(calibration_out),
        "mbpriorq_prior": tensor_metrics(prior_out),
        "activation_metadata": {
            key: ebw[key]
            for key in (
                "vmb_blocks",
                "total_blocks",
                "vmb_partition",
                "mask_ebw",
                "scale_ebw",
                "effective_ebw",
            )
        },
    }


def compare(actual, expected, path="root", tolerance=1e-6) -> list[str]:
    errors: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected object, found {type(actual).__name__}"]
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing")
            else:
                errors.extend(compare(actual[key], value, f"{path}.{key}", tolerance))
        return errors
    if isinstance(expected, list):
        if actual != expected:
            errors.append(f"{path}: expected {expected!r}, found {actual!r}")
        return errors
    if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
            errors.append(f"{path}: expected {expected!r}, found {actual!r}")
        return errors
    if actual != expected:
        errors.append(f"{path}: expected {expected!r}, found {actual!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected",
        type=Path,
        default=ROOT / "evidence/smoke/expected.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "local_runs/smoke/result.json",
    )
    args = parser.parse_args()

    actual = produce()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")

    expected = json.loads(args.expected.read_text())
    errors = compare(actual, expected)
    if errors:
        print("Smoke metric validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Smoke metric validation passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
