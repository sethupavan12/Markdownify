# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevel = Literal["quiet", "normal", "verbose", "debug"]

_LEVEL_MAP = {
    "quiet": logging.WARNING,
    "normal": logging.INFO,
    "verbose": logging.DEBUG,
    "debug": logging.DEBUG,
}

_configured_level: int = logging.INFO


def set_log_level(level: LogLevel) -> None:
    """Set the global log level for all llm_markdownify loggers."""
    global _configured_level
    _configured_level = _LEVEL_MAP.get(level, logging.INFO)

    # Update existing loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("llm_markdownify"):
            logger = logging.getLogger(name)
            logger.setLevel(_configured_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(_configured_level)
    handler = logging.StreamHandler(sys.stderr)  # stdout is reserved for program output
    handler.setLevel(logging.DEBUG)  # Handler allows all; logger controls level

    if _configured_level <= logging.DEBUG:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    else:
        fmt = "%(asctime)s | %(levelname)s | %(message)s"

    formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
