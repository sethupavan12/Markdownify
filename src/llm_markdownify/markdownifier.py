# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from .config import MarkdownifyConfig
from .grouping import group_pages
from .llm import generate_markdown
from .logging import get_logger
from .pager import PageImage, load_document_pages
from .prompt_profiles import load_prompt_profile, PromptProfile

logger = get_logger("llm_markdownify.core")


class Markdownifier:
    """Orchestrates the conversion of a document into Markdown using a Vision LLM."""

    def __init__(self, config: MarkdownifyConfig, profile: str | None = None) -> None:
        self.config = config
        self.profile: PromptProfile = load_prompt_profile(profile or "contracts")

    def _render_pages(self) -> List[PageImage]:
        return load_document_pages(
            self.config.input_path, dpi=self.config.dpi, allow_docx=self.config.allow_docx
        )

    def _group_pages(self, pages: List[PageImage]) -> List[List[PageImage]]:
        return group_pages(
            pages=pages,
            model=self.config.model,
            max_group_pages=self.config.max_group_pages,
            enable_grouping=self.config.enable_grouping,
            profile=self.profile,
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
            for future in tqdm(as_completed(future_to_idx), total=len(groups), desc="LLM groups"):
                idx = future_to_idx[future]
                md = future.result()
                results.append((idx, md))

        ordered = [text for _, text in sorted(results, key=lambda t: t[0])]
        output = "\n\n".join(ordered).strip() + "\n"
        self.config.output_path.write_text(output, encoding="utf-8")
        logger.info("Wrote Markdown to %s", self.config.output_path)
        return self.config.output_path
