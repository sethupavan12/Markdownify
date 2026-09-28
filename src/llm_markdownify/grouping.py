# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from .llm import LLMSettings, assess_continuation, is_fatal
from .pager import PageImage
from .logging import get_logger
from .prompt_profiles import PromptProfile

logger = get_logger("llm_markdownify.grouping")


def group_pages(
    pages: List[PageImage],
    model: str,
    max_group_pages: int,
    enable_grouping: bool,
    profile: PromptProfile,
    settings: LLMSettings | None = None,
    grouping_concurrency: int | None = None,
) -> List[List[PageImage]]:
    if not pages:
        return []

    if not enable_grouping:
        return [[p] for p in pages]

    # Compute continuation labels for adjacent pairs in parallel for speed
    num_pairs = max(0, len(pages) - 1)
    if num_pairs == 0:
        return [[p] for p in pages]

    workers = max(1, min(grouping_concurrency or 8, num_pairs))
    labels: List[str] = ["NONE"] * num_pairs

    def _get_continuation_url(page: PageImage) -> str:
        url = getattr(page, "continuation_data_url", None)
        return url or page.data_url

    def _assess_pair(i: int) -> tuple[int, str]:
        a = pages[i]
        b = pages[i + 1]
        if b.index != a.index + 1:
            return i, "NONE"  # not neighbours in the source (a page selection skipped some)
        try:
            label = assess_continuation(
                model=model,
                first_data_url=_get_continuation_url(a),
                second_data_url=_get_continuation_url(b),
                profile=profile,
                settings=settings,
            )
        except Exception as e:  # noqa: BLE001
            if is_fatal(e):
                raise
            # Not knowing whether a table continues is no reason to lose the document:
            # convert the pages separately.
            logger.warning(
                "Could not check whether page %d continues onto page %d (%s); converting them "
                "separately",
                a.index + 1,
                b.index + 1,
                type(e).__name__,
            )
            label = "NONE"
        return i, label

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_assess_pair, i): i for i in range(num_pairs)}
        try:
            for future in as_completed(futures):
                i, label = future.result()
                labels[i] = label
                a = pages[i]
                b = pages[i + 1]
                logger.info(
                    "Continuation assessment for pages %d->%d: %s", a.index + 1, b.index + 1, label
                )

        except BaseException:
            for pending in futures:  # a fatal error: don't pay for the remaining checks
                pending.cancel()
            raise

    groups: List[List[PageImage]] = []
    current_group: List[PageImage] = [pages[0]]
    for i in range(num_pairs):
        continues = labels[i] == "CONTINUE_NEXT"
        next_page = pages[i + 1]
        if continues and len(current_group) < max_group_pages:
            current_group.append(next_page)
        else:
            groups.append(current_group)
            current_group = [next_page]

    if current_group:
        groups.append(current_group)

    return groups
