# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Response caching for LLM calls to avoid redundant API costs."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from .logging import get_logger

logger = get_logger("llm_markdownify.cache")


def _hash_inputs(*args: str) -> str:
    """Create a stable hash from input strings."""
    combined = "||".join(args)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def default_cache_dir() -> Path:
    """`$LLM_MARKDOWNIFY_CACHE_DIR`, or `~/.cache/llm-markdownify`."""
    override = os.environ.get("LLM_MARKDOWNIFY_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "llm-markdownify"


class ResponseCache:
    """File-based cache for LLM responses."""

    def __init__(self, cache_dir: Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = default_cache_dir()

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, model: str, prompt_hash: str, image_hashes: list[str]) -> str | None:
        """Retrieve cached response if exists."""
        if not self.enabled:
            return None

        key = _hash_inputs(model, prompt_hash, *image_hashes)
        path = self._cache_path(key)

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                logger.info("Cache hit for key %s", key)
                return data.get("response")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Cache read error: %s", e)
                return None
        return None

    def set(self, model: str, prompt_hash: str, image_hashes: list[str], response: str) -> None:
        """Store response in cache."""
        if not self.enabled:
            return

        key = _hash_inputs(model, prompt_hash, *image_hashes)
        path = self._cache_path(key)

        try:
            data = {
                "model": model,
                "prompt_hash": prompt_hash,
                "image_hashes": image_hashes,
                "response": response,
            }
            # Write to a temp file then rename, so concurrent readers never see a partial file.
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, path)
            logger.debug("Cached response with key %s", key)
        except OSError as e:
            logger.warning("Cache write error: %s", e)

    def clear(self) -> int:
        """Clear all cached responses. Returns count of deleted entries."""
        if not self.cache_dir.exists():
            return 0

        count = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
        logger.info("Cleared %d cache entries", count)
        return count
