# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class MarkdownifyConfig(BaseModel):
    """Configuration for the markdownification process."""

    input_path: Path = Field(
        ..., description="Path to input PDF/DOCX or image file (.png/.jpg/.jpeg)"
    )
    output_path: Path = Field(..., description="Path to output Markdown file")

    dpi: int = Field(
        72,
        ge=72,
        le=600,
        description="DPI used to render PDF pages (ignored for direct image inputs)",
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
    temperature: float = Field(0.1, ge=0.0, le=1)
    max_tokens: int = Field(16000, ge=256, le=128000)

    concurrency: int = Field(
        4,
        ge=1,
        le=1000,
        description="Max concurrent LLM requests when processing page groups",
    )

    # Optional path where page images are cached for debugging
    cache_dir: Optional[Path] = Field(None)

    @field_validator("input_path")
    @classmethod
    def _validate_input(cls, path: Path) -> Path:
        if not path.exists():
            raise ValueError(f"Input file not found: {path}")
        if path.suffix.lower() not in {".pdf", ".docx", ".png", ".jpg", ".jpeg"}:
            raise ValueError("input_path must be a .pdf, .docx, .png, .jpg, or .jpeg file")
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

    @model_validator(mode="after")
    def _enforce_pdf_preference(self) -> "MarkdownifyConfig":
        if self.input_path.suffix.lower() == ".docx" and not self.allow_docx:
            raise ValueError(
                "DOCX input is not enabled. Prefer exporting to PDF, or rerun with --allow-docx (requires Word/COM)."
            )
        return self
