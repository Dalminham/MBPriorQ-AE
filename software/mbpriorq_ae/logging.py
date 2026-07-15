#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志模块
支持 DEBUG / RELEASE 两种模式:
  - DEBUG   模式：输出所有信息（DEBUG + RELEASE 标签）
  - RELEASE 模式：仅输出 RELEASE 标签的必要信息
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
    """设置全局日志级别（接受 LogLabel 枚举或字符串 'debug'/'release'）"""
    global _global_label
    if isinstance(label, str):
        _global_label = LogLabel(label.lower())
    else:
        _global_label = label


def get_log_level() -> LogLabel:
    return _global_label


def elog(msg, label: LogLabel = LogLabel.DEBUG):
    """统一打印函数

    DEBUG  全局模式 → 打印所有消息（DEBUG + RELEASE）
    RELEASE 全局模式 → 仅打印 RELEASE 标签的消息
    """
    if _global_label == LogLabel.DEBUG or label == LogLabel.RELEASE:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tag = label.value.upper()
        print(f"{ts} - [{tag}] {msg}")
