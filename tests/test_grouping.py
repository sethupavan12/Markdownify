# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


from llm_markdownify.grouping import group_pages


@dataclass
class FakePage:
    index: int
    data_url: str


def test_grouping_simple(monkeypatch):
    calls = []

    def fake_assess(model, first_data_url, second_data_url):
        calls.append((first_data_url, second_data_url))
        # pages 0->1 continue, 1->2 stop
        if second_data_url is None:
            return "NONE"
        if len(calls) == 1:
            return "CONTINUE_NEXT"
        return "NONE"

    monkeypatch.setattr("llm_markdownify.grouping.assess_continuation", fake_assess)

    pages = [
        FakePage(0, "data:a"),
        FakePage(1, "data:b"),
        FakePage(2, "data:c"),
    ]

    groups = group_pages(pages, model="dummy", max_group_pages=3, enable_grouping=True)
    assert len(groups) == 2
    assert [p.index for p in groups[0]] == [0, 1]
    assert [p.index for p in groups[1]] == [2]


def test_grouping_disabled():
    pages = [
        FakePage(0, "data:a"),
        FakePage(1, "data:b"),
        FakePage(2, "data:c"),
    ]
    groups = group_pages(pages, model="dummy", max_group_pages=2, enable_grouping=False)
    assert len(groups) == 3
