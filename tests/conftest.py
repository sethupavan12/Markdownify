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
