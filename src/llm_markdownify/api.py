# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import MarkdownifyConfig
from .markdownifier import Markdownifier


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model: Optional[str] = None,
    dpi: int = 200,
    max_group_pages: int = 3,
    enable_grouping: bool = True,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    concurrency: int = 4,
    profile: Optional[str] = None,
    allow_docx: bool = False,
) -> Path:
    """Convert a document to Markdown using the configured LLM via LiteLLM.

    Parameters
    - input_path: PDF path (preferred) or DOCX if `allow_docx=True`
    - output_path: Markdown file destination
    - model: LiteLLM model name (e.g., 'gpt-4.1-mini', 'azure/<deployment>', 'gemini/gemini-2.5-flash')
    - dpi: Render DPI for PDF pages (higher = slower, clearer)
    - max_group_pages: Max pages to merge when a table/chart spans pages
    - enable_grouping: Whether to use LLM to detect cross-page continuations
    - temperature: LLM temperature
    - max_tokens: LLM max tokens; if None, uses config default
    - concurrency: Max parallel LLM calls across page groups
    - profile: Prompt profile name ('contracts', 'generic') or path to a JSON profile
    - allow_docx: Enable DOCX via Word/COM conversion (not recommended; prefer PDFs)

    Returns
    - Path to the written Markdown file
    """
    cfg_kwargs = dict(
        input_path=Path(input_path),
        output_path=Path(output_path),
        dpi=dpi,
        max_group_pages=max_group_pages,
        enable_grouping=enable_grouping,
        temperature=temperature,
        concurrency=concurrency,
        allow_docx=allow_docx,
    )
    if model is not None:
        cfg_kwargs["model"] = model
    if max_tokens is not None:
        cfg_kwargs["max_tokens"] = max_tokens

    cfg = MarkdownifyConfig(**cfg_kwargs)
    return Markdownifier(cfg, profile=profile).run()
