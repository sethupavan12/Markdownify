# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    wait_random_exponential,
)

from .cache import ResponseCache
from .logging import get_logger
from .prompt_profiles import PromptProfile

# Soften LiteLLM's heavy logging/cold-storage features which can import proxy/apscheduler
# and cause shutdown-time errors on some Python versions.
os.environ.setdefault("LITELLM_LOGGING", "false")
os.environ.setdefault("LITELLM_DISABLE_COLD_STORAGE", "1")
os.environ.setdefault("LITELLM_LOG_LEVEL", "ERROR")

# Aggressively silence noisy third-party loggers
for logger_name in ("LiteLLM", "litellm", "litellm.proxy", "apscheduler"):
    try:
        _log = logging.getLogger(logger_name)
        _log.setLevel(logging.CRITICAL)
        _log.propagate = False
    except Exception:
        pass

logger = get_logger("llm_markdownify.llm")


class EmptyResponseError(RuntimeError):
    """The model returned no content for no visible reason. Retried."""


class OutputBudgetExhaustedError(RuntimeError):
    """The model hit max_tokens before writing any answer (typically all spent on reasoning).

    Not retried: the same request fails the same way every time and each attempt is billed.
    """


class RateLimiter:
    """Token bucket limiting requests per minute.

    Each caller reserves a slot under the lock before sleeping, so concurrent waiters are spread
    out instead of all waking at once. The bucket holds about one second of requests.
    """

    def __init__(self, rpm: Optional[int] = None) -> None:
        self.rpm = rpm
        self.lock = threading.Lock()
        self.rate = (rpm / 60.0) if rpm else float("inf")
        self.capacity = max(1.0, self.rate) if rpm else float("inf")
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def acquire(self) -> None:
        """Block until a request slot is available."""
        if self.rpm is None:
            return
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.rate)
            self.last_refill = now
            self.tokens -= 1.0  # reserve; may go negative, which queues later callers further out
            wait_time = 0.0 if self.tokens >= 0 else -self.tokens / self.rate
        if wait_time > 0:
            logger.debug("Rate limit: waiting %.2fs", wait_time)
            time.sleep(wait_time)


# Rate limits clear with time, so they get a time budget instead of an attempt count. This keeps a
# page alive while another job saturates the same key's tokens-per-minute quota.
RATE_LIMIT_BUDGET_S = 180.0


@dataclass
class LLMSettings:
    """Everything one conversion needs to call the model: retries, rate limit, extra provider
    options and the response cache.

    Each conversion carries its own instance, so conversions running at the same time in one
    process never see each other's settings or API keys. To make several conversions share one
    request budget, give them the same `rate_limiter`.
    """

    max_retries: int = 5
    retry_delay: float = 1.0
    rate_limiter: Optional[RateLimiter] = None
    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    cache: ResponseCache = field(default_factory=lambda: ResponseCache(enabled=False))


def _hash_content(content: str) -> str:
    """Full sha256 of content, used for cache keys."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient failures only. Auth, bad-request and not-found errors fail fast."""
    if isinstance(exc, EmptyResponseError):
        return True
    import litellm  # type: ignore

    transient = (
        litellm.RateLimitError,
        litellm.APIConnectionError,
        litellm.Timeout,
        litellm.InternalServerError,
        litellm.ServiceUnavailableError,
    )
    return isinstance(exc, transient)


def _stop_after(max_retries: int):
    def _stop_retrying(state: RetryCallState) -> bool:
        import litellm  # type: ignore

        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, litellm.RateLimitError):
            return state.seconds_since_start >= RATE_LIMIT_BUDGET_S
        # Count other failures separately so waiting out rate limits doesn't use up their retries.
        other_failures = getattr(state, "other_failures", 0) + 1
        state.other_failures = other_failures  # type: ignore[attr-defined]
        return other_failures > max_retries  # first attempt isn't a retry

    return _stop_retrying


def _completion_with_retry(
    *,
    model: str,
    messages: list,
    temperature: float | None,
    max_tokens: int | None,
    settings: LLMSettings,
) -> tuple[str, str | None]:
    """Run one completion with retries. Returns (content, finish_reason)."""
    # Local import to allow env configuration above to take effect
    import litellm  # type: ignore
    from litellm import completion as _litellm_completion  # type: ignore

    # LiteLLM prints "If you need to debug this error..." to the terminal on every handled error,
    # which leaks into --quiet output. We report errors ourselves.
    litellm.suppress_debug_info = True

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=_stop_after(settings.max_retries),
        wait=wait_random_exponential(multiplier=settings.retry_delay, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _do_completion() -> tuple[str, str | None]:
        if settings.rate_limiter:
            # every attempt counts against the limit, retries included
            settings.rate_limiter.acquire()
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "drop_params": True}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs.update(settings.llm_kwargs)
        if (
            kwargs.get("api_base")
            and "api_key" not in kwargs
            and model.startswith("openai/")
            and not os.environ.get("OPENAI_API_KEY")
        ):
            # Local OpenAI-compatible servers (LM Studio, Ollama, vLLM, llama.cpp) need no key, but
            # the OpenAI client refuses to start without one.
            kwargs["api_key"] = "not-needed"
        resp = _litellm_completion(**kwargs)
        choice = resp["choices"][0]
        content = choice["message"]["content"]
        if not content or not str(content).strip():
            if choice.get("finish_reason") == "stop":
                # The model finished normally with nothing to say: a blank page, or one holding
                # only headers/footers we asked it to drop. That is a correct answer, not a failure.
                return "", "stop"
            if choice.get("finish_reason") == "length":
                raise OutputBudgetExhaustedError(
                    "The model used its whole output budget without answering (usually on "
                    "reasoning). Raise --max-tokens or lower --reasoning-effort."
                )
            raise EmptyResponseError(
                f"Model returned empty content (finish_reason={choice.get('finish_reason')})"
            )
        return str(content), choice.get("finish_reason")

    return _do_completion()


def _message_with_images(text: str, image_data_urls: List[str]) -> dict:
    """Build a message dict with text and images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": content}


_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?[ \t]*\n(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


_INNER_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)


def strip_markdown_fence(text: str) -> str:
    """Remove a single fence wrapping the whole response (models often add ```markdown ... ```).

    A bare ``` fence is only stripped when nothing inside it is fenced too; otherwise the text is
    a document that starts and ends with separate code blocks and must be left alone.
    """
    match = _FENCE_RE.match(text)
    if not match:
        return text
    tagged = text.lstrip()[3:].lower().startswith(("markdown", "md"))
    body = match.group(1)
    if not tagged and _INNER_FENCE_RE.search(body):
        return text
    return body


def parse_continuation_label(text: str) -> str:
    """Map a free-form model answer to CONTINUE_NEXT or NONE. Anything doubtful is NONE."""
    words = set(re.findall(r"[A-Z_]+", text.upper()))
    if words & {"NONE", "NOT", "NO"}:
        return "NONE"
    if "CONTINUE_NEXT" in words or {"CONTINUE", "NEXT"} <= words:
        return "CONTINUE_NEXT"
    return "NONE"


# kwargs that change how a request is sent, not what the model answers. Kept out of cache keys so
# rotating a key or changing a timeout does not invalidate the cache.
_TRANSPORT_ONLY_KWARGS = {"api_key", "timeout", "num_retries", "extra_headers"}


def _request_key(model: str, messages: list, llm_kwargs: dict[str, Any], **params: Any) -> str:
    """Stable hash of everything that affects the model's answer."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "params": params,
            "extra": {k: v for k, v in llm_kwargs.items() if k not in _TRANSPORT_ONLY_KWARGS},
        },
        sort_keys=True,
        default=str,
    )
    return _hash_content(payload)


def assess_continuation(
    model: str,
    first_data_url: str,
    second_data_url: str | None,
    profile: PromptProfile,
    settings: Optional[LLMSettings] = None,
) -> str:
    """Assess if pages should be merged (continuation detection)."""
    settings = settings or LLMSettings()
    images = [first_data_url] + ([second_data_url] if second_data_url else [])
    messages = [
        {"role": "system", "content": profile.continuation_system},
        _message_with_images(profile.continuation_user, images),
    ]

    cache = settings.cache
    key = _request_key(model, messages, settings.llm_kwargs, task="continuation")
    cached = cache.get(model, key, [])
    if cached is not None:
        return cached

    # No max_tokens cap: reasoning models spend tokens thinking before answering, and a tiny
    # cap leaves them with an empty answer.
    content, _ = _completion_with_retry(
        model=model, messages=messages, temperature=None, max_tokens=None, settings=settings
    )
    result = parse_continuation_label(content)
    cache.set(model, key, [], result)
    return result


def generate_markdown(
    model: str,
    image_data_urls: List[str],
    profile: PromptProfile,
    temperature: float | None = None,
    max_tokens: int | None = None,
    settings: Optional[LLMSettings] = None,
) -> str:
    """Generate markdown from page images."""
    settings = settings or LLMSettings()
    messages = [
        {"role": "system", "content": profile.markdown_system},
        _message_with_images(profile.markdown_user, image_data_urls),
    ]

    cache = settings.cache
    key = _request_key(
        model, messages, settings.llm_kwargs, temperature=temperature, max_tokens=max_tokens
    )
    cached = cache.get(model, key, [])
    if cached is not None:
        return cached

    content, finish_reason = _completion_with_retry(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        settings=settings,
    )
    if finish_reason == "length":
        logger.warning(
            "Model output was truncated at max_tokens; the Markdown for this group is incomplete. "
            "Raise --max-tokens or lower --max-group-pages."
        )
    result = strip_markdown_fence(content).strip()
    if finish_reason != "length":  # never cache a truncated answer
        cache.set(model, key, [], result)
    return result
