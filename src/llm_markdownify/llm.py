# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import logging
from typing import List

from .prompt_profiles import PromptProfile

# Soften LiteLLM's heavy logging/cold-storage features which can import proxy/apscheduler
# and cause shutdown-time errors on some Python versions. These are safe defaults and can
# be overridden by the user's environment.
os.environ.setdefault("LITELLM_LOGGING", "false")
os.environ.setdefault("LITELLM_DISABLE_COLD_STORAGE", "1")
os.environ.setdefault("LITELLM_LOG_LEVEL", "ERROR")

# Aggressively silence noisy third-party loggers that can emit shutdown-time traces
for logger_name in ("LiteLLM", "litellm", "litellm.proxy", "apscheduler"):
    try:
        _log = logging.getLogger(logger_name)
        _log.setLevel(logging.CRITICAL)
        _log.propagate = False
    except Exception:
        pass


def _completion(*, model: str, messages: list, temperature: float, max_tokens: int | None):
    # Local import to allow env configuration above to take effect and satisfy import-order linting
    import litellm  # type: ignore
    from litellm import completion as _litellm_completion  # type: ignore

    # Drop unsupported params for strict models (e.g., gpt-5-mini)
    try:
        litellm.drop_params = True  # type: ignore[attr-defined]
    except Exception:
        pass

    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return _litellm_completion(**kwargs)


def _message_with_images(text: str, image_data_urls: List[str]) -> dict:
    content = [{"type": "text", "text": text}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": content}


def assess_continuation(
    model: str,
    first_data_url: str,
    second_data_url: str | None,
    profile: PromptProfile,
) -> str:
    images = [first_data_url] + ([second_data_url] if second_data_url else [])
    messages = [
        {"role": "system", "content": profile.continuation_system},
        _message_with_images(profile.continuation_user, images),
    ]
    resp = _completion(model=model, messages=messages, temperature=0.0, max_tokens=4)
    return str(resp["choices"][0]["message"]["content"]).strip().upper()


def generate_markdown(
    model: str,
    image_data_urls: List[str],
    profile: PromptProfile,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> str:
    messages = [
        {"role": "system", "content": profile.markdown_system},
        _message_with_images(profile.markdown_user, image_data_urls),
    ]
    resp = _completion(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return str(resp["choices"][0]["message"]["content"]).strip()
