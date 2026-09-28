# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, List, Optional

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .cache import ResponseCache
from .config import MarkdownifyConfig
from .grouping import group_pages
from .llm import LLMSettings, RateLimiter, generate_markdown, is_fatal
from .logging import PACKAGE_LOGGER, get_logger
from .pager import PageImage, render_document
from .prompt_profiles import DEFAULT_PROFILE, PromptProfile, load_prompt_profile
from .result import ConversionResult, PageResult, UsageCounter
from .sources import Source, load_source

logger = get_logger("llm_markdownify.core")

PageCallback = Callable[[PageResult], None]


def _pages_label(group: List[PageImage]) -> str:
    first, last = group[0].index + 1, group[-1].index + 1
    return f"page {first}" if first == last else f"pages {first}-{last}"


def failed_marker(group: List[PageImage], error: BaseException) -> str:
    """Placeholder left in the Markdown where a page could not be converted. Only the error type is
    included: messages can be long and may echo request details."""
    return (
        f"<!-- llm-markdownify: {_pages_label(group)} failed ({type(error).__name__}). "
        "Rerun to retry. -->"
    )


class Markdownifier:
    """Orchestrates the conversion of a document into Markdown using a Vision LLM."""

    def __init__(
        self,
        config: MarkdownifyConfig,
        profile: str | None = None,
        rate_limiter: RateLimiter | None = None,
        show_progress: bool = False,
        on_page: Optional[PageCallback] = None,
    ) -> None:
        """`rate_limiter` lets several conversions share one request budget (e.g. one API key);
        by default each conversion gets its own limiter from `config.rate_limit_rpm`.
        `show_progress` draws a progress bar on stderr (the CLI turns it on; libraries stay quiet).
        `on_page` is called with each page's PageResult as soon as it is done.
        """
        self.show_progress = show_progress
        self.on_page = on_page
        self.config = config
        self.profile: PromptProfile = load_prompt_profile(profile or DEFAULT_PROFILE)

        # This conversion's own settings; nothing here is shared with other conversions.
        if rate_limiter is None and config.rate_limit_rpm:
            rate_limiter = RateLimiter(config.rate_limit_rpm)
        self.settings = LLMSettings(
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            rate_limiter=rate_limiter,
            llm_kwargs=dict(config.llm_kwargs),
            cache=ResponseCache(cache_dir=config.cache_dir, enabled=config.enable_cache),
        )

    def _group_pages(self, pages: List[PageImage]) -> List[List[PageImage]]:
        return group_pages(
            pages=pages,
            model=self.config.model,
            max_group_pages=self.config.max_group_pages,
            enable_grouping=self.config.enable_grouping,
            profile=self.profile,
            settings=self.settings,
            grouping_concurrency=(
                self.config.grouping_concurrency
                if self.config.grouping_concurrency
                else self.config.concurrency
            ),
        )

    def _markdown_for_group(self, group: List[PageImage]) -> str:
        image_urls = [p.data_url for p in group]
        return generate_markdown(
            model=self.config.model,
            image_data_urls=image_urls,
            profile=self.profile,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            settings=self.settings,
        )

    def _page_results(
        self, group: List[PageImage], markdown: str, error: Optional[BaseException]
    ) -> List[PageResult]:
        first = group[0].index + 1
        error_text = f"{type(error).__name__}: {error}"[:300] if error else None
        results = [
            PageResult(number=first, markdown=markdown if error is None else "", error=error_text)
        ]
        for page in group[1:]:
            results.append(PageResult(number=page.index + 1, error=error_text, merged_into=first))
        return results

    def convert(self, source: Source) -> ConversionResult:
        """Convert a path, bytes, binary file object or URL, and return the Markdown in memory.

        A page that fails after retries is marked in the Markdown and in `result.pages`, and the
        rest of the document is still converted. It raises instead when nothing could be converted,
        on errors no page can avoid (bad key, unknown model), or when `config.strict` is set.
        """
        started = time.monotonic()
        self.settings.usage = UsageCounter()  # this conversion's totals only
        doc = load_source(source, allow_private_urls=self.config.allow_private_urls)
        total, pages = render_document(
            doc,
            dpi=self.config.dpi,
            max_side=self.config.max_image_px,
            image_format=self.config.image_format,
            allow_docx=self.config.allow_docx,
            pages=self.config.pages,
        )
        if not pages:
            raise ValueError(f"{doc.name} has no pages to convert")
        groups = self._group_pages(pages)
        logger.info(
            "Processing %d groups with concurrency=%d", len(groups), self.config.concurrency
        )

        markdown_by_group: dict[int, str] = {}
        errors: dict[int, BaseException] = {}
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            future_to_idx = {
                executor.submit(self._markdown_for_group, group): idx
                for idx, group in enumerate(groups)
            }
            # Log lines print above the bar instead of through it; the bar is always closed.
            redirect = (
                logging_redirect_tqdm(loggers=[logging.getLogger(PACKAGE_LOGGER)])
                if self.show_progress
                else nullcontext()
            )
            with (
                redirect,
                tqdm(
                    as_completed(future_to_idx),
                    total=len(groups),
                    desc="Pages",
                    disable=not self.show_progress,
                ) as progress,
            ):
                try:
                    for future in progress:
                        idx = future_to_idx[future]
                        group = groups[idx]
                        try:
                            markdown_by_group[idx] = future.result()
                        except Exception as e:
                            if self.config.strict or is_fatal(e):
                                logger.error("Conversion failed at %s", _pages_label(group))
                                raise
                            errors[idx] = e
                            logger.warning(
                                "Could not convert %s (%s); continuing with the rest",
                                _pages_label(group),
                                type(e).__name__,
                            )
                        if self.on_page:
                            for page_result in self._page_results(
                                group, markdown_by_group.get(idx, ""), errors.get(idx)
                            ):
                                self.on_page(page_result)
                except BaseException:
                    # Whatever stops us (a fatal error, strict mode, an on_page callback raising,
                    # Ctrl-C): don't keep paying for queued pages.
                    for pending in future_to_idx:
                        pending.cancel()
                    raise

        if len(errors) == len(groups):
            # Nothing usable: surface the real error instead of a document made of placeholders.
            raise errors[0]

        parts: List[str] = []
        page_results: List[PageResult] = []
        for idx, group in enumerate(groups):
            error = errors.get(idx)
            text = markdown_by_group.get(idx, "")
            parts.append(failed_marker(group, error) if error else text)
            page_results.extend(self._page_results(group, text, error))

        warnings = []
        if errors:
            failed = sorted(p.number for p in page_results if p.error)
            warnings.append(
                f"{len(failed)} page(s) failed: {', '.join(map(str, failed))}. "
                "Run again to retry them; finished pages come from the cache."
                if self.config.enable_cache
                else f"{len(failed)} page(s) failed: {', '.join(map(str, failed))}."
            )
        return ConversionResult(
            markdown="\n\n".join(p for p in parts if p).strip() + "\n",
            pages=page_results,
            usage=self.settings.usage.snapshot(),
            model=self.config.model,
            source=doc.name,
            total_pages=total,
            elapsed_s=time.monotonic() - started,
            warnings=warnings,
        )

    def run(self) -> Path:
        """Convert `config.input_path` and write `config.output_path` (what `convert()` does)."""
        if self.config.input_path is None or self.config.output_path is None:
            raise ValueError("run() needs config.input_path and config.output_path")
        result = self.convert(self.config.input_path)
        self.config.output_path.write_text(result.markdown, encoding="utf-8")
        for warning in result.warnings:
            logger.warning(warning)
        logger.info("Wrote Markdown to %s", self.config.output_path)
        return self.config.output_path
