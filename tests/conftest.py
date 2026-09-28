# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest

from llm_markdownify.logging import PACKAGE_LOGGER


@pytest.fixture(autouse=True)
def _reset_package_logger():
    """CLI tests turn on console logging; undo it so every test starts from the silent default."""
    package = logging.getLogger(PACKAGE_LOGGER)
    saved = (list(package.handlers), package.level, package.propagate)
    yield
    package.handlers[:] = saved[0]
    package.setLevel(saved[1])
    package.propagate = saved[2]


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    """The response cache is on by default; keep tests out of the user's real cache."""
    monkeypatch.setenv("LLM_MARKDOWNIFY_CACHE_DIR", str(tmp_path / "llm-markdownify-cache"))
