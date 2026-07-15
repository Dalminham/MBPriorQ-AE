"""VMB prior-coverage profiling for MBPriorQ activation quantization."""

import csv
import os
from typing import Any

import torch


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return numerator / denominator, or 0.0 when the denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _as_bool(mask: torch.Tensor) -> torch.Tensor:
    """Return a detached boolean view of a mask tensor."""
    return mask.detach().to(torch.bool)


def _pair_counts(selected_mask: torch.Tensor, oracle_mask: torch.Tensor) -> dict[str, int]:
    """Count selected-vs-oracle true positives, false positives, and false negatives."""
    selected = _as_bool(selected_mask)
    oracle = _as_bool(oracle_mask)
    if selected.shape != oracle.shape:
        raise ValueError(f"VMB profile mask shape mismatch: selected={selected.shape}, oracle={oracle.shape}")

    true_positive = int((selected & oracle).sum().item())
    false_positive = int((selected & ~oracle).sum().item())
    false_negative = int((~selected & oracle).sum().item())
    true_negative = int((~selected & ~oracle).sum().item())
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
    }


def _column_mask(mask: torch.Tensor) -> torch.Tensor:
    """Collapse token rows into a per-microblock-column VMB mask."""
    mask = _as_bool(mask)
    if mask.dim() == 1:
        return mask
    return mask.reshape(-1, mask.shape[-1]).any(dim=0)


def _mask_to_hex(mask: torch.Tensor) -> str:
    """Serialize a 1-D boolean mask as a fixed-width hexadecimal string."""
    mask = _as_bool(mask).reshape(-1).detach().cpu()
    if mask.numel() == 0:
        return ""
    bit_string = "".join("1" if value else "0" for value in mask.tolist())
    width = (mask.numel() + 3) // 4
    return format(int(bit_string, 2), f"0{width}x")


def _pair_metrics(prefix: str, selected_mask: torch.Tensor, oracle_mask: torch.Tensor) -> dict[str, Any]:
    """Return count and ratio metrics for a selected mask against an oracle mask."""
    counts = _pair_counts(selected_mask, oracle_mask)
    selected_blocks = counts["tp"] + counts["fp"]
    oracle_blocks = counts["tp"] + counts["fn"]
    union_blocks = counts["tp"] + counts["fp"] + counts["fn"]
    precision = _safe_ratio(counts["tp"], selected_blocks)
    recall = _safe_ratio(counts["tp"], oracle_blocks)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        f"{prefix}_tp": counts["tp"],
        f"{prefix}_fp": counts["fp"],
        f"{prefix}_fn": counts["fn"],
        f"{prefix}_tn": counts["tn"],
        f"{prefix}_selected_blocks": selected_blocks,
        f"{prefix}_oracle_blocks": oracle_blocks,
        f"{prefix}_total_blocks": selected_mask.numel(),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_miss_rate": _safe_ratio(counts["fn"], oracle_blocks),
        f"{prefix}_f1": f1,
        f"{prefix}_jaccard": _safe_ratio(counts["tp"], union_blocks),
    }


class GlobalVMBProfiler:
    """Collect VMB prior-coverage records emitted by MBPriorQ activation quantizers."""

    enabled: bool = False
    output_path: str | None = None
    metadata: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    execution_phase: str = ""

    @classmethod
    def configure(cls, enabled: bool, output_path: str | None = None, metadata: dict[str, Any] | None = None):
        """Reset and configure the profiler for one PPL run."""
        cls.enabled = bool(enabled)
        cls.output_path = output_path
        cls.metadata = dict(metadata or {})
        cls.records = []
        cls.execution_phase = ""

    @classmethod
    def set_execution_phase(cls, phase: str):
        """Annotate subsequently recorded rows with the current model execution phase."""
        cls.execution_phase = phase

    @classmethod
    def record(
        cls,
        *,
        layer_name: str,
        model_type: str,
        call_index: int,
        phase: str,
        selected_rule: str,
        selected_mask: torch.Tensor,
        oracle_mask: torch.Tensor,
        calibration_column_mask: torch.Tensor | None = None,
        calibration_threshold: Any = None,
        selected_threshold: Any = None,
        oracle_threshold: Any = None,
        replaced_percent: Any = None,
    ):
        """Record one real activation-quantization event."""
        if not cls.enabled:
            return

        selected_mask = _as_bool(selected_mask)
        oracle_mask = _as_bool(oracle_mask)
        selected_column = _column_mask(selected_mask)
        oracle_column = _column_mask(oracle_mask)

        row = dict(cls.metadata)
        row.update(
            {
                "layer_name": layer_name,
                "model_type": model_type,
                "call_index": call_index,
                "phase": phase,
                "execution_phase": cls.execution_phase,
                "selected_rule": selected_rule,
                "activation_rows": selected_mask.shape[0] if selected_mask.dim() > 1 else 1,
                "microblock_columns": selected_mask.shape[-1],
                "selected_column_hex": _mask_to_hex(selected_column),
                "oracle_column_hex": _mask_to_hex(oracle_column),
                "calibration_threshold": _scalar_or_empty(calibration_threshold),
                "selected_threshold": _scalar_or_empty(selected_threshold),
                "oracle_threshold": _scalar_or_empty(oracle_threshold),
                "replaced_percent": _scalar_or_empty(replaced_percent),
            }
        )
        row.update(_pair_metrics("full", selected_mask, oracle_mask))
        row.update(_pair_metrics("column", selected_column, oracle_column))

        if calibration_column_mask is not None:
            calibration_column = _as_bool(calibration_column_mask).to(oracle_column.device)
            row["calibration_column_hex"] = _mask_to_hex(calibration_column)
            row.update(_pair_metrics("calibration_column", calibration_column, oracle_column))

        cls.records.append(row)

    @classmethod
    def summary(cls) -> dict[str, Any]:
        """Return aggregate prior-phase coverage metrics across all records."""
        prior_records = [row for row in cls.records if row.get("phase") == "prior"]
        full_tp = sum(int(row.get("full_tp", 0)) for row in prior_records)
        full_fp = sum(int(row.get("full_fp", 0)) for row in prior_records)
        full_fn = sum(int(row.get("full_fn", 0)) for row in prior_records)
        full_oracle = full_tp + full_fn
        full_selected = full_tp + full_fp
        full_precision = _safe_ratio(full_tp, full_selected)
        full_recall = _safe_ratio(full_tp, full_oracle)
        return {
            "records": len(cls.records),
            "prior_records": len(prior_records),
            "full_precision": full_precision,
            "full_recall": full_recall,
            "full_miss_rate": _safe_ratio(full_fn, full_oracle),
        }

    @classmethod
    def write_csv(cls, output_path: str | None = None) -> str | None:
        """Write collected records to CSV and return the path."""
        path = output_path or cls.output_path
        if not cls.enabled or not path:
            return None

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames: list[str] = []
        seen = set()
        for row in cls.records:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cls.records)
        return path


def _scalar_or_empty(value: Any) -> Any:
    """Convert a torch scalar to a Python value while keeping missing values empty."""
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return ""
        return value.detach().float().item()
    return value
