# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Logging for llm-markdownify.

As a library it is silent: log records go to the `llm_markdownify` logger, which only has a
`NullHandler`, so nothing is printed unless the application configures logging (for example
`logging.basicConfig(level=logging.INFO)`). The `markdownify` command, or a caller that asks for
it, turns on console output with `enable_console_logging()`.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevel = Literal["quiet", "normal", "verbose", "debug"]

PACKAGE_LOGGER = "llm_markdownify"

_LEVEL_MAP = {
    "quiet": logging.WARNING,
    "normal": logging.INFO,
    "verbose": logging.DEBUG,
    "debug": logging.DEBUG,
}

logging.getLogger(PACKAGE_LOGGER).addHandler(logging.NullHandler())


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever `sys.stderr` is at the time of each record, not the stream that existed
    when the handler was created (test runners and some apps replace sys.stderr)."""

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:  # StreamHandler.__init__ assigns it; ignore
        pass


def get_logger(name: str) -> logging.Logger:
    """Logger for a module of this package. Adds no handlers: the application decides output."""
    return logging.getLogger(name)


def enable_console_logging(level: LogLevel = "normal") -> None:
    """Print llm-markdownify's logs to stderr, as the command-line tool does.

    Safe to call more than once: it reuses its own handler and only updates the level. Stdout is
    left alone so it can carry Markdown.
    """
    package = logging.getLogger(PACKAGE_LOGGER)
    numeric = _LEVEL_MAP.get(level, logging.INFO)
    handler = next((h for h in package.handlers if getattr(h, "_markdownify_console", False)), None)
    if handler is None:
        handler = _StderrHandler()
        handler._markdownify_console = True  # type: ignore[attr-defined]
        package.addHandler(handler)
    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        if numeric <= logging.DEBUG
        else "%(asctime)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    package.setLevel(numeric)
    package.propagate = False  # avoid printing twice if the app also logs to the console
