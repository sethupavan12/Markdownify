# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""What a conversion returns: the Markdown, per-page outcomes, and token usage."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Usage:
    """Model usage for one conversion. Answers served from the cache cost nothing and are not
    counted. `cost_usd` is estimated from LiteLLM's price list and is None when a model is missing
    from it."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = 0.0
    cached_responses: int = 0


class UsageCounter:
    """Thread-safe running totals for one conversion (pages are converted in parallel)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage = Usage()

    def add_response(
        self, input_tokens: int, output_tokens: int, cost_usd: Optional[float]
    ) -> None:
        with self._lock:
            u = self._usage
            u.requests += 1
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens
            # One unknown price makes the total unknown rather than silently too low.
            u.cost_usd = None if (u.cost_usd is None or cost_usd is None) else u.cost_usd + cost_usd

    def add_cache_hit(self) -> None:
        with self._lock:
            self._usage.cached_responses += 1

    def snapshot(self) -> Usage:
        with self._lock:
            return Usage(**asdict(self._usage))


@dataclass
class PageResult:
    """The outcome for one page of the source document.

    When a table or chart continues across pages, those pages are converted together: the first
    page of the group carries the Markdown, and the others have `merged_into` set to its number.
    """

    number: int  # 1-based page number in the source document
    markdown: str = ""
    error: Optional[str] = None
    merged_into: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ConversionResult:
    markdown: str
    pages: list[PageResult]
    usage: Usage
    model: str
    source: str
    total_pages: int  # pages in the document, including ones not selected
    elapsed_s: float
    warnings: list[str] = field(default_factory=list)

    @property
    def failed_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.error is not None]

    @property
    def ok(self) -> bool:
        return not self.failed_pages

    def page(self, number: int) -> PageResult:
        """The result for 1-based page `number` (it must have been converted)."""
        for p in self.pages:
            if p.number == number:
                return p
        raise KeyError(f"Page {number} was not converted")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failed_pages"] = self.failed_pages
        data["ok"] = self.ok
        return data
