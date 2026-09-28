# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from llm_markdownify.config import MarkdownifyConfig


def test_config_defaults(tmp_path: Path):
    """Config has sensible defaults."""
    input_pdf = tmp_path / "test.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    output_md = tmp_path / "out.md"

    cfg = MarkdownifyConfig(input_path=input_pdf, output_path=output_md)

    assert cfg.dpi == 200
    assert cfg.max_image_px == 2048
    assert cfg.image_format == "jpeg"
    assert cfg.max_group_pages == 3
    assert cfg.enable_grouping is True
    assert cfg.temperature is None
    assert cfg.concurrency == 4
    assert cfg.max_retries == 5
    assert cfg.retry_delay == 1.0
    assert cfg.rate_limit_rpm is None
    assert cfg.enable_cache is True  # decision A: on by default
    assert cfg.log_level is None  # deprecated field, no longer used


def test_config_custom_retry_settings(tmp_path: Path):
    """Config accepts custom retry settings."""
    input_pdf = tmp_path / "test.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    output_md = tmp_path / "out.md"

    cfg = MarkdownifyConfig(
        input_path=input_pdf,
        output_path=output_md,
        max_retries=5,
        retry_delay=2.5,
    )

    assert cfg.max_retries == 5
    assert cfg.retry_delay == 2.5


def test_config_rate_limit(tmp_path: Path):
    """Config accepts rate limit setting."""
    input_pdf = tmp_path / "test.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    output_md = tmp_path / "out.md"

    cfg = MarkdownifyConfig(
        input_path=input_pdf,
        output_path=output_md,
        rate_limit_rpm=60,
    )

    assert cfg.rate_limit_rpm == 60


def test_config_cache_settings(tmp_path: Path):
    """Config accepts cache settings."""
    input_pdf = tmp_path / "test.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    output_md = tmp_path / "out.md"
    cache_dir = tmp_path / "cache"

    cfg = MarkdownifyConfig(
        input_path=input_pdf,
        output_path=output_md,
        enable_cache=True,
        cache_dir=cache_dir,
    )

    assert cfg.enable_cache is True
    assert cfg.cache_dir == cache_dir


def test_config_log_level_is_deprecated(tmp_path: Path):
    """log_level on the config no longer does anything, so setting it warns."""
    input_pdf = tmp_path / "test.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    with pytest.warns(DeprecationWarning, match="no longer configures logging"):
        MarkdownifyConfig(input_path=input_pdf, output_path=tmp_path / "out.md", log_level="debug")


def test_config_invalid_retry_bounds(tmp_path: Path):
    """Config validates retry bounds."""
    input_pdf = tmp_path / "test.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    output_md = tmp_path / "out.md"

    with pytest.raises(ValueError):
        MarkdownifyConfig(
            input_path=input_pdf,
            output_path=output_md,
            max_retries=15,  # > 10
        )


def test_config_invalid_rate_limit(tmp_path: Path):
    """Config validates rate limit bounds."""
    input_pdf = tmp_path / "test.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    output_md = tmp_path / "out.md"

    with pytest.raises(ValueError):
        MarkdownifyConfig(
            input_path=input_pdf,
            output_path=output_md,
            rate_limit_rpm=0,  # < 1
        )


def test_config_input_not_found(tmp_path: Path):
    """Config raises for missing input file."""
    output_md = tmp_path / "out.md"

    with pytest.raises(ValueError, match="not found"):
        MarkdownifyConfig(
            input_path=tmp_path / "nonexistent.pdf",
            output_path=output_md,
        )


def test_config_invalid_input_extension(tmp_path: Path):
    """Config raises for invalid input extension."""
    input_txt = tmp_path / "test.txt"
    input_txt.write_text("hello")
    output_md = tmp_path / "out.md"

    with pytest.raises(ValueError, match="must be"):
        MarkdownifyConfig(
            input_path=input_txt,
            output_path=output_md,
        )


def test_config_invalid_output_extension(tmp_path: Path):
    """Config raises for invalid output extension."""
    input_pdf = tmp_path / "test.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")

    with pytest.raises(ValueError, match="must be"):
        MarkdownifyConfig(
            input_path=input_pdf,
            output_path=tmp_path / "out.txt",
        )
