# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal, Optional

from .config import MarkdownifyConfig
from .llm import RateLimiter
from .logging import LogLevel, console_logging, get_logger
from .markdownifier import Markdownifier, PageCallback
from .result import ConversionResult
from .sources import Source

logger = get_logger("llm_markdownify.api")

_OPTIONS_DOC = """
    Options (None means the default from `MarkdownifyConfig`, so Python and the CLI behave alike):
    - model: LiteLLM model name (e.g. 'gpt-5.4-mini', 'anthropic/claude-opus-5', 'openai/<local>')
    - pages: 1-based page selection such as "1-5,12,40-" (default: all pages)
    - strict: raise if any page fails, instead of marking it and returning the rest
    - profile: prompt profile name ('generic' default, 'contracts') or a path to a JSON profile
    - dpi, max_image_px, image_format: how pages are rendered (200 DPI, 2048 px, jpeg)
    - max_group_pages, enable_grouping: merging of tables/charts that continue across pages
    - temperature, max_tokens: generation settings (provider default, 16000)
    - llm_kwargs: extra LiteLLM arguments (api_base, api_key, reasoning_effort, timeout...)
    - concurrency, grouping_concurrency: parallel requests (4)
    - max_retries, retry_delay, rate_limit_rpm: retries for transient errors and request pacing
    - rate_limiter: a RateLimiter shared with other conversions, to pace them together
    - enable_cache, cache_dir: answers are cached on disk by default, so reruns only pay for pages
      that failed or changed ($LLM_MARKDOWNIFY_CACHE_DIR or ~/.cache/llm-markdownify)
    - allow_docx: accept .docx (converted through Microsoft Word)
    - allow_private_urls: allow URLs on private/local addresses (refused by default). Strings that
      are not URLs are read as local paths, so only pass trusted strings as `source`.
    - on_page: called with each PageResult as soon as that page is done
    - log_level: None keeps the library silent; 'normal'/'verbose'/'quiet' prints the CLI's
      progress and logs to stderr for this call only
"""


def _build(
    options: dict[str, Any],
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> MarkdownifyConfig:
    if options.get("cache_dir") is not None:
        options["cache_dir"] = Path(options["cache_dir"])
    return MarkdownifyConfig(
        input_path=input_path,
        output_path=output_path,
        **{k: v for k, v in options.items() if v is not None},
    )


def _run(
    cfg: MarkdownifyConfig,
    source: Source,
    profile: Optional[str],
    rate_limiter: Optional[RateLimiter],
    on_page: Optional[PageCallback],
    log_level: Optional[LogLevel],
    output_path: Optional[Path] = None,
) -> ConversionResult:
    def convert_and_write(show_progress: bool) -> ConversionResult:
        result = Markdownifier(
            cfg,
            profile=profile,
            rate_limiter=rate_limiter,
            show_progress=show_progress,
            on_page=on_page,
        ).convert(source)
        for warning in result.warnings:
            logger.warning(warning)
        if output_path is not None:
            output_path.write_text(result.markdown, encoding="utf-8")
            logger.info("Wrote Markdown to %s", output_path)
        return result

    if log_level is None:
        return convert_and_write(False)
    # Console output for this call only; the app's own logging config is left as it was.
    with console_logging(log_level):
        return convert_and_write(log_level != "quiet")


def markdownify(
    source: Source,
    *,
    model: Optional[str] = None,
    pages: Optional[str] = None,
    strict: Optional[bool] = None,
    profile: Optional[str] = None,
    dpi: Optional[int] = None,
    max_image_px: Optional[int] = None,
    image_format: Optional[Literal["jpeg", "png"]] = None,
    max_group_pages: Optional[int] = None,
    enable_grouping: Optional[bool] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    llm_kwargs: Optional[dict[str, Any]] = None,
    concurrency: Optional[int] = None,
    grouping_concurrency: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
    rate_limit_rpm: Optional[int] = None,
    rate_limiter: Optional[RateLimiter] = None,
    enable_cache: Optional[bool] = None,
    cache_dir: Optional[str | Path] = None,
    allow_docx: Optional[bool] = None,
    allow_private_urls: Optional[bool] = None,
    on_page: Optional[PageCallback] = None,
    log_level: Optional[LogLevel] = None,
) -> ConversionResult:
    options = dict(locals())
    for key in ("source", "profile", "rate_limiter", "on_page", "log_level"):
        options.pop(key)
    cfg = _build(options)
    return _run(cfg, source, profile, rate_limiter, on_page, log_level)


markdownify.__doc__ = (
    """Convert a document to Markdown and return it, with per-page results and token usage.

    `source` is a file path, bytes, a binary file object, or an http(s) URL (PDF or image).

        result = markdownify("report.pdf", model="gpt-5.4-mini")
        result.markdown          # the whole document
        result.page(7).markdown  # one page
        result.failed_pages      # [] when every page converted
        result.usage             # requests, tokens, estimated cost

    A page that still fails after retries is marked in the Markdown with an HTML comment and listed
    in `result.failed_pages`; the other pages are returned. It raises when nothing could be
    converted, on errors no page can avoid (bad key, unknown model), or with `strict=True`.
"""
    + _OPTIONS_DOC
)


async def amarkdownify(source: Source, **options: Any) -> ConversionResult:
    """Async version of `markdownify()`, for event loops such as web servers. Runs the same
    conversion in a worker thread; takes the same arguments."""
    return await asyncio.to_thread(markdownify, source, **options)


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model: Optional[str] = None,
    pages: Optional[str] = None,
    strict: Optional[bool] = None,
    profile: Optional[str] = None,
    dpi: Optional[int] = None,
    max_image_px: Optional[int] = None,
    image_format: Optional[Literal["jpeg", "png"]] = None,
    max_group_pages: Optional[int] = None,
    enable_grouping: Optional[bool] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    llm_kwargs: Optional[dict[str, Any]] = None,
    concurrency: Optional[int] = None,
    grouping_concurrency: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
    rate_limit_rpm: Optional[int] = None,
    rate_limiter: Optional[RateLimiter] = None,
    enable_cache: Optional[bool] = None,
    cache_dir: Optional[str | Path] = None,
    allow_docx: Optional[bool] = None,
    allow_private_urls: Optional[bool] = None,
    on_page: Optional[PageCallback] = None,
    log_level: Optional[LogLevel] = None,
) -> Path:
    options = dict(locals())
    for key in ("input_path", "output_path", "profile", "rate_limiter", "on_page", "log_level"):
        options.pop(key)
    cfg = _build(options, Path(input_path), Path(output_path))
    _run(cfg, cfg.input_path, profile, rate_limiter, on_page, log_level, cfg.output_path)
    return cfg.output_path  # type: ignore[return-value]


convert.__doc__ = (
    """Convert a file to Markdown and write it to `output_path`. Returns `output_path`.

    Same conversion as `markdownify()`; use that to get the Markdown, per-page results and usage in
    memory. Pages that fail after retries are marked in the file with an HTML comment.
"""
    + _OPTIONS_DOC
)
