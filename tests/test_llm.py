# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

from llm_markdownify.llm import LLMSettings, RateLimiter, _hash_content


def test_hash_content_deterministic():
    """Same input produces same hash."""
    h1 = _hash_content("hello world")
    h2 = _hash_content("hello world")
    assert h1 == h2


def test_hash_content_different_inputs():
    """Different inputs produce different hashes."""
    h1 = _hash_content("hello")
    h2 = _hash_content("world")
    assert h1 != h2


def test_rate_limiter_no_limit():
    """RateLimiter with None rpm doesn't block."""
    limiter = RateLimiter(rpm=None)
    start = time.monotonic()
    for _ in range(10):
        limiter.acquire()
    elapsed = time.monotonic() - start
    # Should be nearly instant
    assert elapsed < 0.1


def test_rate_limiter_high_limit():
    """RateLimiter with high limit doesn't block much."""
    limiter = RateLimiter(rpm=6000)  # 100 per second
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    elapsed = time.monotonic() - start
    # Should be fast
    assert elapsed < 0.5


def test_rate_limiter_low_limit():
    """RateLimiter with low limit introduces delays."""
    limiter = RateLimiter(rpm=120)  # 2 per second
    start = time.monotonic()
    limiter.acquire()  # First one is instant
    limiter.acquire()  # Second one should be ~0.5s wait
    elapsed = time.monotonic() - start
    # Should take some time but not too long
    assert elapsed < 2.0


def test_default_settings():
    """Each conversion gets fresh settings: sensible retries, no rate limit, cache off."""
    settings = LLMSettings()
    assert settings.max_retries == 5
    assert settings.rate_limiter is None
    assert settings.cache.enabled is False
    assert LLMSettings().llm_kwargs is not settings.llm_kwargs  # never shared between instances
