#!/usr/bin/env python3
"""Validate full-resident and streamed PPL numerical equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation_common import finite_float, positive_int, require_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--relative-tolerance", type=float, default=2e-4)
    parser.add_argument("--expected-samples", type=int)
    args = parser.parse_args()
    root = Path(args.results)
    failures = []
    for method in ("bf16", "mbpriorq"):
        full = json.loads((root / f"{method}__full_gpu.json").read_text(encoding="utf-8"))
        streamed = json.loads((root / f"{method}__streamed.json").read_text(encoding="utf-8"))
        try:
            for backend, payload in (("full_gpu", full), ("streamed", streamed)):
                context = f"{method}:{backend}"
                require_fields(payload, ("method", "backend", "num_samples", "perplexity", "total_nll"), context)
                if payload["method"] != method or payload["backend"] != backend:
                    raise ValueError(f"{context}: method/backend metadata mismatch")
                positive_int(payload["num_samples"], f"{context}:num_samples", args.expected_samples)
                finite_float(payload["perplexity"], f"{context}:perplexity")
                finite_float(payload["total_nll"], f"{context}:total_nll")
        except (TypeError, ValueError) as error:
            failures.append(str(error))
            continue
        if full["num_samples"] != streamed["num_samples"]:
            failures.append(f"{method}: window count differs")
            continue
        denominator = max(abs(float(full["total_nll"])), 1.0)
        relative = abs(float(full["total_nll"]) - float(streamed["total_nll"])) / denominator
        if relative > args.relative_tolerance:
            failures.append(
                f"{method}: NLL relative difference {relative:.6g} exceeds "
                f"{args.relative_tolerance:.6g}"
            )
    if failures:
        raise SystemExit("Backend equivalence failed:\n- " + "\n- ".join(failures))
    print("Backend equivalence passed for BF16 and MBPriorQ")


if __name__ == "__main__":
    main()
