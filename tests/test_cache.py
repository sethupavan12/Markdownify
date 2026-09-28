# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from llm_markdownify.cache import ResponseCache


def test_cache_disabled():
    """Cache returns None when disabled."""
    cache = ResponseCache(enabled=False)
    cache.set("model", "prompt", ["img1"], "response")
    assert cache.get("model", "prompt", ["img1"]) is None


def test_cache_set_and_get(tmp_path: Path):
    """Cache stores and retrieves responses correctly."""
    cache = ResponseCache(cache_dir=tmp_path, enabled=True)

    # Initially empty
    assert cache.get("gpt-4", "abc123", ["img_hash1"]) is None

    # Set a value
    cache.set("gpt-4", "abc123", ["img_hash1"], "# Hello World")

    # Retrieve it
    result = cache.get("gpt-4", "abc123", ["img_hash1"])
    assert result == "# Hello World"


def test_cache_different_keys(tmp_path: Path):
    """Different inputs produce different cache entries."""
    cache = ResponseCache(cache_dir=tmp_path, enabled=True)

    cache.set("gpt-4", "prompt1", ["img1"], "response1")
    cache.set("gpt-4", "prompt2", ["img1"], "response2")
    cache.set("gpt-4", "prompt1", ["img2"], "response3")

    assert cache.get("gpt-4", "prompt1", ["img1"]) == "response1"
    assert cache.get("gpt-4", "prompt2", ["img1"]) == "response2"
    assert cache.get("gpt-4", "prompt1", ["img2"]) == "response3"


def test_cache_clear(tmp_path: Path):
    """Clear removes all cache entries."""
    cache = ResponseCache(cache_dir=tmp_path, enabled=True)

    cache.set("model", "p1", ["i1"], "r1")
    cache.set("model", "p2", ["i2"], "r2")

    count = cache.clear()
    assert count == 2

    assert cache.get("model", "p1", ["i1"]) is None
    assert cache.get("model", "p2", ["i2"]) is None


def test_cache_multiple_images(tmp_path: Path):
    """Cache handles multiple image hashes correctly."""
    cache = ResponseCache(cache_dir=tmp_path, enabled=True)

    cache.set("model", "prompt", ["img1", "img2", "img3"], "multi-page response")

    assert cache.get("model", "prompt", ["img1", "img2", "img3"]) == "multi-page response"
    # Different order = different key
    assert cache.get("model", "prompt", ["img3", "img2", "img1"]) is None
