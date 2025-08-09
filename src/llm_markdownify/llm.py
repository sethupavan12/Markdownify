# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

import litellm  # type: ignore
from litellm import completion  # type: ignore

from .prompt_profiles import PromptProfile

# Drop unsupported params for strict models (e.g., gpt-5-mini)
litellm.drop_params = True  # type: ignore[attr-defined]


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
    resp = completion(model=model, messages=messages, temperature=0.0, max_tokens=4)
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
    resp = completion(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return str(resp["choices"][0]["message"]["content"]).strip()
