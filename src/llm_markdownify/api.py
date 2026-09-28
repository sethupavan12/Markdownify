# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from .config import MarkdownifyConfig
from .logging import enable_console_logging
from .markdownifier import Markdownifier

LogLevel = Literal["quiet", "normal", "verbose", "debug"]


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model: Optional[str] = None,
    dpi: Optional[int] = None,
    max_image_px: Optional[int] = None,
    image_format: Optional[Literal["jpeg", "png"]] = None,
    max_group_pages: Optional[int] = None,
    enable_grouping: Optional[bool] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    concurrency: Optional[int] = None,
    grouping_concurrency: Optional[int] = None,
    profile: Optional[str] = None,
    allow_docx: Optional[bool] = None,
    llm_kwargs: Optional[dict[str, Any]] = None,
    # Retry options
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
    # Rate limiting
    rate_limit_rpm: Optional[int] = None,
    # Caching
    enable_cache: Optional[bool] = None,
    cache_dir: Optional[str | Path] = None,
    # Logging
    log_level: Optional[LogLevel] = None,
) -> Path:
    """Convert a document to Markdown using the configured LLM via LiteLLM.

    Any argument left as None uses the default from `MarkdownifyConfig`, so the Python API and
    the CLI always behave the same.

    Parameters
    - input_path: PDF, image (.png/.jpg/.jpeg/.webp/.tif/.tiff/.bmp/.gif), or DOCX if `allow_docx`
    - output_path: Markdown file destination
    - model: LiteLLM model name (e.g. 'gpt-5.4-mini', 'azure/<deployment>', 'gemini/gemini-2.5-flash')
    - dpi: Render DPI for PDF pages (default 200, ignored for image inputs)
    - max_image_px: Longest side in pixels of each page image sent to the model (default 2048)
    - image_format: 'jpeg' (default, smaller uploads) or 'png'
    - max_group_pages: Max pages to merge when a table/chart spans pages
    - enable_grouping: Whether to use the LLM to detect cross-page continuations
    - temperature: Sampling temperature; None uses the provider default
    - max_tokens: Max output tokens per call
    - concurrency: Max parallel LLM calls across page groups
    - grouping_concurrency: Max parallel continuation checks (defaults to `concurrency`)
    - profile: Prompt profile name ('generic' default, 'contracts') or path to a JSON profile
    - allow_docx: Enable DOCX via Word/COM conversion (not recommended; prefer PDFs)
    - llm_kwargs: Extra LiteLLM completion kwargs (api_base, api_key, reasoning_effort, timeout...)
    - max_retries: Max retries for transient LLM errors (rate limits, timeouts, 5xx)
    - retry_delay: Base delay for exponential backoff between retries, in seconds
    - rate_limit_rpm: Max requests per minute (None = no limit)
    - enable_cache: Enable response caching to avoid redundant LLM calls
    - cache_dir: Directory for response cache (defaults to ~/.cache/llm-markdownify)
    - log_level: None (default) keeps the library silent; your app's logging config decides.
      'quiet', 'normal', 'verbose' or 'debug' prints progress and logs to stderr like the CLI.

    Returns
    - Path to the written Markdown file
    """
    options: dict[str, Any] = dict(
        model=model,
        dpi=dpi,
        max_image_px=max_image_px,
        image_format=image_format,
        max_group_pages=max_group_pages,
        enable_grouping=enable_grouping,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
        grouping_concurrency=grouping_concurrency,
        allow_docx=allow_docx,
        llm_kwargs=llm_kwargs,
        max_retries=max_retries,
        retry_delay=retry_delay,
        rate_limit_rpm=rate_limit_rpm,
        enable_cache=enable_cache,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        log_level=log_level,
    )
    if log_level is not None:
        enable_console_logging(log_level)
    cfg = MarkdownifyConfig(
        input_path=Path(input_path),
        output_path=Path(output_path),
        **{k: v for k, v in options.items() if v is not None},
    )
    show_progress = log_level is not None and log_level != "quiet"
    return Markdownifier(cfg, profile=profile, show_progress=show_progress).run()
