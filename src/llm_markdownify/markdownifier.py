# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from .cache import configure_cache
from .config import MarkdownifyConfig
from .grouping import group_pages
from .llm import configure_llm, generate_markdown
from .logging import get_logger, set_log_level
from .pager import PageImage, load_document_pages
from .prompt_profiles import DEFAULT_PROFILE, PromptProfile, load_prompt_profile

logger = get_logger("llm_markdownify.core")


class Markdownifier:
    """Orchestrates the conversion of a document into Markdown using a Vision LLM."""

    def __init__(self, config: MarkdownifyConfig, profile: str | None = None) -> None:
        self.config = config
        self.profile: PromptProfile = load_prompt_profile(profile or DEFAULT_PROFILE)

        # Configure logging level
        set_log_level(config.log_level)

        # Configure LLM retry and rate limiting
        configure_llm(
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            rate_limit_rpm=config.rate_limit_rpm,
            llm_kwargs=config.llm_kwargs,
        )

        # Configure caching
        configure_cache(
            cache_dir=config.cache_dir,
            enabled=config.enable_cache,
        )

    def _render_pages(self) -> List[PageImage]:
        return load_document_pages(
            self.config.input_path,
            dpi=self.config.dpi,
            allow_docx=self.config.allow_docx,
            max_side=self.config.max_image_px,
            image_format=self.config.image_format,
        )

    def _group_pages(self, pages: List[PageImage]) -> List[List[PageImage]]:
        return group_pages(
            pages=pages,
            model=self.config.model,
            max_group_pages=self.config.max_group_pages,
            enable_grouping=self.config.enable_grouping,
            profile=self.profile,
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
        )

    def run(self) -> Path:
        pages = self._render_pages()
        groups = self._group_pages(pages)

        logger.info(
            "Processing %d groups with concurrency=%d", len(groups), self.config.concurrency
        )

        # Submit LLM work in parallel, but preserve order by collecting (idx, result)
        results: List[Tuple[int, str]] = []
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            future_to_idx = {
                executor.submit(self._markdown_for_group, group): idx
                for idx, group in enumerate(groups)
            }
            progress = tqdm(
                as_completed(future_to_idx),
                total=len(groups),
                desc="LLM groups",
                disable=self.config.log_level == "quiet",
            )
            for future in progress:
                idx = future_to_idx[future]
                try:
                    md = future.result()
                except Exception:
                    # Don't keep paying for the remaining groups when the document will fail.
                    for pending in future_to_idx:
                        pending.cancel()
                    first_page = groups[idx][0].index + 1
                    logger.error("Conversion failed for the group starting at page %d", first_page)
                    raise
                results.append((idx, md))

        ordered = [text for _, text in sorted(results, key=lambda t: t[0])]
        output = "\n\n".join(ordered).strip() + "\n"
        self.config.output_path.write_text(output, encoding="utf-8")
        logger.info("Wrote Markdown to %s", self.config.output_path)
        return self.config.output_path
