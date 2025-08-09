# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from . import prompts as default_prompts


@dataclass(frozen=True)
class PromptProfile:
    name: str
    continuation_system: str
    continuation_user: str
    markdown_system: str
    markdown_user: str


_BUILTIN_PROFILES: Dict[str, PromptProfile] = {
    "contracts": PromptProfile(
        name="contracts",
        continuation_system=default_prompts.CONTINUATION_SYSTEM_PROMPT,
        continuation_user=default_prompts.CONTINUATION_USER_PROMPT,
        markdown_system=default_prompts.MARKDOWN_SYSTEM_PROMPT,
        markdown_user=default_prompts.MARKDOWN_USER_PROMPT,
    ),
    "generic": PromptProfile(
        name="generic",
        continuation_system=(
            "You analyze page images to decide if content visually continues (tables/charts split). "
            "Respond only CONTINUE_NEXT or NONE."
        ),
        continuation_user=(
            "Look for split tables/charts at the end of page A and start of page B. If found, CONTINUE_NEXT; else NONE."
        ),
        markdown_system=(
            "Convert page images into clean Markdown. Use clear headings, lists, and tables; cover all pages; avoid page numbers."
        ),
        markdown_user=("Produce a single coherent Markdown output from the provided page images."),
    ),
}


def load_prompt_profile(name_or_path: str) -> PromptProfile:
    candidate = Path(name_or_path)
    if candidate.exists() and candidate.is_file():
        with candidate.open("r", encoding="utf-8") as f:
            data = json.load(f)
        required = {
            "name",
            "continuation_system",
            "continuation_user",
            "markdown_system",
            "markdown_user",
        }
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Prompt profile missing required fields {sorted(missing)} in {candidate}"
            )
        return PromptProfile(
            name=str(data["name"]),
            continuation_system=str(data["continuation_system"]),
            continuation_user=str(data["continuation_user"]),
            markdown_system=str(data["markdown_system"]),
            markdown_user=str(data["markdown_user"]),
        )

    key = name_or_path.strip().lower()
    if key in _BUILTIN_PROFILES:
        return _BUILTIN_PROFILES[key]

    raise ValueError(
        f"Unknown prompt profile '{name_or_path}'. Provide a built-in name ({', '.join(sorted(_BUILTIN_PROFILES))}) "
        f"or a path to a JSON file with the fields: name, continuation_system, continuation_user, markdown_system, markdown_user."
    )
