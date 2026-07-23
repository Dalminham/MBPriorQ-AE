#!/usr/bin/env python3
"""
Shared logging helpers.

DEBUG mode prints all messages. RELEASE mode prints only messages labeled
RELEASE.
"""

from enum import Enum
from datetime import datetime


class LogLabel(Enum):
    DEBUG = "debug"
    RELEASE = "release"


DEBUG = LogLabel.DEBUG
RELEASE = LogLabel.RELEASE

_global_label: LogLabel = LogLabel.DEBUG


def set_log_level(label):
    """Set the global level from a LogLabel or a debug/release string."""
    global _global_label
    if isinstance(label, str):
        _global_label = LogLabel(label.lower())
    else:
        _global_label = label


def get_log_level() -> LogLabel:
    return _global_label


def elog(msg, label: LogLabel = LogLabel.DEBUG):
    """Print a timestamped message when its label is enabled."""
    if _global_label == LogLabel.DEBUG or label == LogLabel.RELEASE:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tag = label.value.upper()
        print(f"{ts} - [{tag}] {msg}")
