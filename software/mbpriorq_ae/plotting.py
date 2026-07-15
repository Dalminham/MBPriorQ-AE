"""Artifact-safe plotting hooks.

The paper execution path does not require calibration-mask images. The original
development helper writes to a workstation-specific directory, so the curated
artifact keeps the call contract but disables that optional side effect.
"""

from __future__ import annotations

from typing import Any


def draw_carlibration_mask(*_args: Any, **_kwargs: Any) -> None:
    """Preserve the development-hook signature without writing local images."""
