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
import threading
from collections.abc import Iterator
from contextlib import contextmanager
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

    def setStream(self, stream):
        """Not supported: this handler always follows the current sys.stderr."""
        raise NotImplementedError("_StderrHandler always writes to the current sys.stderr")


def get_logger(name: str) -> logging.Logger:
    """Logger for a module of this package. Adds no handlers: the application decides output."""
    return logging.getLogger(name)


def _format_for(numeric: int) -> logging.Formatter:
    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        if numeric <= logging.DEBUG
        else "%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def enable_console_logging(level: LogLevel = "normal") -> None:
    """Print llm-markdownify's logs to stderr for the rest of the process, as the CLIs do.

    Meant for programs that own the process (the command-line tools). Libraries and apps should
    use their own logging config, or `console_logging()` for a single call. Safe to call more than
    once: it reuses its own handler and only updates the level. Stdout is left alone.
    """
    package = logging.getLogger(PACKAGE_LOGGER)
    numeric = _LEVEL_MAP.get(level, logging.INFO)
    handler = next((h for h in package.handlers if getattr(h, "_markdownify_console", False)), None)
    if handler is None:
        handler = _StderrHandler()
        handler._markdownify_console = True  # type: ignore[attr-defined]
        package.addHandler(handler)
    handler.setFormatter(_format_for(numeric))
    package.setLevel(numeric)
    package.propagate = False  # avoid printing twice if the app also logs to the console


_scoped_lock = threading.Lock()
_scoped_levels: list[int] = []
_scoped_state: dict = {}


@contextmanager
def console_logging(level: LogLevel = "normal") -> Iterator[None]:
    """Print llm-markdownify's logs to stderr only while the block runs, then restore the logger.

    Used by `convert(..., log_level=...)`. It does not stop records from reaching the app's own
    handlers. Overlapping calls (e.g. from several threads) share one stderr handler; while they
    overlap, the most verbose requested level applies.
    """
    numeric = _LEVEL_MAP.get(level, logging.INFO)
    package = logging.getLogger(PACKAGE_LOGGER)
    with _scoped_lock:
        if not _scoped_levels:
            handler = _StderrHandler()
            handler.setFormatter(_format_for(numeric))
            _scoped_state.update(level=package.level, handler=handler)
            package.addHandler(handler)
        _scoped_levels.append(numeric)
        package.setLevel(min(_scoped_levels))
    try:
        yield
    finally:
        with _scoped_lock:
            _scoped_levels.remove(numeric)
            if _scoped_levels:
                package.setLevel(min(_scoped_levels))
            else:
                package.removeHandler(_scoped_state.pop("handler"))
                package.setLevel(_scoped_state.pop("level"))
