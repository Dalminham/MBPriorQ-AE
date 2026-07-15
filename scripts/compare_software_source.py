#!/usr/bin/env python3
"""Compare the curated quantizers against a pinned EasyLLM source checkout."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch


EXPECTED_EASYLLM_COMMIT = "e48ba9c618553bf1036d83c9dbed35e30b5ad3ba"


def args_for(mode: str = "paper", refined_block_size: int = 4) -> dict:
    return {
        "name": "ae.equivalence",
        "device": "cpu",
        "quant_bit": 4,
        "quant_sym": True,
        "model_type": "cloud",
        "ablation_mode": mode,
        "random_seed": 20260606,
        "refined_block_size": refined_block_size,
        "using_imatrix": False,
        "imatrix_file_name": "unused",
        "vmb_profile_enable": False,
        "metadata_target": "activation",
    }


def assert_equal(label: str, left: torch.Tensor, right: torch.Tensor) -> None:
    if not torch.equal(left, right):
        max_diff = (left.float() - right.float()).abs().max().item()
        raise AssertionError(f"{label} differs; max_abs_diff={max_diff}")
    print(f"[PASS] {label}: shape={tuple(left.shape)} dtype={left.dtype}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easyllm", type=Path, required=True)
    ns = parser.parse_args()
    source = ns.easyllm.resolve()

    import subprocess

    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_EASYLLM_COMMIT:
        raise RuntimeError(
            f"Expected EasyLLM {EXPECTED_EASYLLM_COMMIT}, found {commit}"
        )

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "software"))
    sys.path.insert(0, str(source))

    source_mb = importlib.import_module("EasyQuant.MBPriorQ.MBPriorQ").MBPriorQ_Quantizer
    from mbpriorq_ae import MBPriorQ_Quantizer

    torch.manual_seed(1308)
    calibration = torch.randn(8, 64)
    later = torch.randn(8, 64)
    weight = torch.randn(32, 64)

    for refined in (2, 4, 8):
        for mode in ("paper", "static", "first2_only", "random_same_ratio", "oracle"):
            source_q = source_mb(args_for(mode, refined))
            curated_q = MBPriorQ_Quantizer(args_for(mode, refined))
            assert_equal(
                f"MBPriorQ {mode} refined={refined} calibration",
                source_q.fake_quantize_activation(
                    calibration, name="layer", tensor_shape=calibration.shape
                ),
                curated_q.fake_quantize_activation(
                    calibration, name="layer", tensor_shape=calibration.shape
                ),
            )
            assert_equal(
                f"MBPriorQ {mode} refined={refined} prior",
                source_q.fake_quantize_activation(
                    later, name="layer", tensor_shape=later.shape
                ),
                curated_q.fake_quantize_activation(
                    later, name="layer", tensor_shape=later.shape
                ),
            )

    source_weight = source_mb(args_for())
    curated_weight = MBPriorQ_Quantizer(args_for())
    assert_equal(
        "MBPriorQ weight",
        source_weight.fake_quantize_weight(weight, name="layer"),
        curated_weight.fake_quantize_weight(weight, name="layer"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
