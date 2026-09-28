# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import threading

from llm_markdownify.logging import (
    PACKAGE_LOGGER,
    console_logging,
    enable_console_logging,
    get_logger,
)


def _console_handlers():
    package = logging.getLogger(PACKAGE_LOGGER)
    return [h for h in package.handlers if getattr(h, "_markdownify_console", False)]


def test_library_is_silent_by_default(capsys):
    """Importing and using the library must not print anything the app did not ask for."""
    get_logger("llm_markdownify.core").warning("should not appear")
    out, err = capsys.readouterr()
    assert out == "" and err == ""


def test_app_logging_config_receives_library_records(caplog):
    with caplog.at_level(logging.INFO, logger=PACKAGE_LOGGER):
        get_logger("llm_markdownify.core").info("converted 3 pages")
    assert "converted 3 pages" in caplog.text


def test_console_logging_goes_to_stderr_never_stdout(capsys):
    enable_console_logging("normal")
    get_logger("llm_markdownify.core").info("hello")
    out, err = capsys.readouterr()
    assert out == ""
    assert "hello" in err


def test_console_logging_is_idempotent_and_updates_level():
    enable_console_logging("normal")
    enable_console_logging("quiet")
    assert len(_console_handlers()) == 1
    assert logging.getLogger(PACKAGE_LOGGER).level == logging.WARNING
    enable_console_logging("verbose")
    assert logging.getLogger(PACKAGE_LOGGER).level == logging.DEBUG


def test_get_logger_adds_no_handlers():
    logger = get_logger("llm_markdownify.some_module")
    assert logger.handlers == []
    assert logger.propagate is True


def test_console_logging_block_restores_the_logger(capsys):
    package = logging.getLogger(PACKAGE_LOGGER)
    before = (list(package.handlers), package.level, package.propagate)
    with console_logging("normal"):
        get_logger("llm_markdownify.core").info("inside")
    get_logger("llm_markdownify.core").warning("after")
    _, err = capsys.readouterr()
    assert "inside" in err and "after" not in err
    assert (list(package.handlers), package.level, package.propagate) == before


def test_console_logging_block_keeps_app_handlers_working(caplog):
    """A per-call opt-in must not hide records from the app's own logging setup."""
    with caplog.at_level(logging.INFO), console_logging("normal"):
        get_logger("llm_markdownify.core").info("to the app too")
    assert "to the app too" in caplog.text


def test_overlapping_console_logging_blocks_restore_cleanly():
    package = logging.getLogger(PACKAGE_LOGGER)
    before = (list(package.handlers), package.level)
    started = threading.Barrier(2)

    def run(level):
        with console_logging(level):
            started.wait()

    threads = [threading.Thread(target=run, args=(lvl,)) for lvl in ("quiet", "verbose")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (list(package.handlers), package.level) == before
