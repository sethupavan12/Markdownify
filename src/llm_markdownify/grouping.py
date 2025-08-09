# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

from .llm import assess_continuation
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
) -> List[List[PageImage]]:
    if not pages:
        return []

    if not enable_grouping:
        return [[p] for p in pages]

    groups: List[List[PageImage]] = []
    current_group: List[PageImage] = [pages[0]]

    for i in range(len(pages)):
        if i == len(pages) - 1:
            groups.append(current_group)
            break
        a = pages[i]
        b = pages[i + 1]

        label = assess_continuation(
            model=model, first_data_url=a.data_url, second_data_url=b.data_url, profile=profile
        )
        logger.info("Continuation assessment for pages %d->%d: %s", a.index + 1, b.index + 1, label)

        continues = label == "CONTINUE_NEXT"
        if continues and len(current_group) < max_group_pages:
            current_group.append(b)
        else:
            groups.append(current_group)
            current_group = [b]

    return groups
