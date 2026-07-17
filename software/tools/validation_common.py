"""Shared structural checks for artifact result validators."""

from __future__ import annotations

import math


def require_fields(payload: dict, fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"{context}: missing fields: {', '.join(missing)}")


def finite_float(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}: expected a finite value, got {value!r}")
    return number


def positive_int(value, label: str, expected: int | None = None) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label}: expected a positive integer, got {value!r}")
    if expected is not None and number != expected:
        raise ValueError(f"{label}: observed {number}, expected {expected}")
    return number
