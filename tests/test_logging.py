# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from llm_markdownify.logging import PACKAGE_LOGGER, enable_console_logging, get_logger


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
