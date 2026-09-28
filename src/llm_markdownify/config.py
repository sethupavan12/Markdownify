# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .pager import SUPPORTED_SUFFIXES


LogLevel = Literal["quiet", "normal", "verbose", "debug"]


class MarkdownifyConfig(BaseModel):
    """Configuration for the markdownification process."""

    input_path: Path = Field(
        ..., description="Path to input PDF/DOCX or image file (.png/.jpg/.jpeg)"
    )
    output_path: Path = Field(..., description="Path to output Markdown file")

    dpi: int = Field(
        200,
        ge=72,
        le=600,
        description="DPI used to render PDF pages (ignored for direct image inputs). "
        "The rendered image is still capped at max_image_px.",
    )
    max_image_px: int = Field(
        2048,
        ge=512,
        le=8192,
        description="Longest side, in pixels, of every page image sent to the model",
    )
    image_format: Literal["jpeg", "png"] = Field(
        "jpeg", description="Encoding for page images sent to the model"
    )
    max_group_pages: int = Field(3, ge=1, le=10, description="Max pages to group together")
    enable_grouping: bool = Field(True, description="Enable LLM-based grouping")

    # Prefer PDFs; DOCX allowed only with explicit opt-in
    allow_docx: bool = Field(
        False,
        description="Allow DOCX via Word/COM conversion (not recommended). Prefer PDFs.",
    )

    model: str = Field(
        default_factory=lambda: os.getenv("LLM_MARKDOWNIFY_MODEL", "gpt-4.1-mini"),
        description="LiteLLM model name (e.g., gpt-4.1-mini, azure/<deployment>, gemini/gemini-2.5-flash)",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. None uses the provider default "
        "(reasoning models reject custom temperatures)",
    )
    max_tokens: int = Field(16000, ge=256, le=128000)
    llm_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments passed to every LiteLLM completion call "
        "(e.g. api_base, api_key, reasoning_effort, timeout)",
    )

    concurrency: int = Field(
        4,
        ge=1,
        le=1000,
        description="Max concurrent LLM requests when processing page groups",
    )

    grouping_concurrency: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Max concurrent LLM requests for adjacent-page continuation checks (defaults to concurrency)",
    )

    # Retry configuration
    max_retries: int = Field(
        5,
        ge=0,
        le=10,
        description="Max retry attempts for failed LLM calls",
    )
    retry_delay: float = Field(
        1.0,
        ge=0.1,
        le=60.0,
        description="Initial delay between retries in seconds (exponential backoff)",
    )

    # Rate limiting
    rate_limit_rpm: Optional[int] = Field(
        None,
        ge=1,
        le=10000,
        description="Max requests per minute (None = no limit)",
    )

    # Caching
    enable_cache: bool = Field(
        False,
        description="Enable response caching to avoid redundant LLM calls",
    )
    cache_dir: Optional[Path] = Field(
        None,
        description="Directory for response cache (defaults to ~/.cache/llm-markdownify)",
    )

    # Logging
    log_level: Optional[LogLevel] = Field(
        None,
        description="Deprecated: no effect. Use convert(log_level=...) or "
        "llm_markdownify.logging.enable_console_logging()",
    )

    @field_validator("input_path")
    @classmethod
    def _validate_input(cls, path: Path) -> Path:
        if not path.exists():
            raise ValueError(f"Input file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"input_path must be one of: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        return path

    @field_validator("output_path")
    @classmethod
    def _validate_output(cls, path: Path) -> Path:
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError("output_path must be a .md or .markdown file")
        return path

    @field_validator("log_level")
    @classmethod
    def _warn_log_level(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            warnings.warn(
                "MarkdownifyConfig.log_level no longer configures logging and will be removed. "
                "Use convert(log_level=...) or llm_markdownify.logging.enable_console_logging().",
                DeprecationWarning,
                stacklevel=2,
            )
        return value

    @model_validator(mode="after")
    def _enforce_pdf_preference(self) -> "MarkdownifyConfig":
        if self.input_path.suffix.lower() == ".docx" and not self.allow_docx:
            raise ValueError(
                "DOCX input is not enabled. Prefer exporting to PDF, or rerun with --allow-docx (requires Word/COM)."
            )
        return self
