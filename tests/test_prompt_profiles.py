# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from llm_markdownify.prompt_profiles import load_prompt_profile


def test_load_builtin_contracts():
    """Load the contracts profile."""
    profile = load_prompt_profile("contracts")
    assert profile.name == "contracts"
    assert "document structure" in profile.continuation_system.lower()


def test_load_builtin_generic():
    """Load the generic profile."""
    profile = load_prompt_profile("generic")
    assert profile.name == "generic"


def test_load_builtin_case_insensitive():
    """Profile names are case-insensitive."""
    profile1 = load_prompt_profile("CONTRACTS")
    profile2 = load_prompt_profile("Contracts")
    profile3 = load_prompt_profile("contracts")

    assert profile1.name == profile2.name == profile3.name


def test_load_unknown_profile_raises():
    """Unknown profile name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown prompt profile"):
        load_prompt_profile("nonexistent_profile")


def test_load_custom_json_profile(tmp_path: Path):
    """Load a custom JSON profile."""
    profile_data = {
        "name": "custom",
        "continuation_system": "Custom continuation system",
        "continuation_user": "Custom continuation user",
        "markdown_system": "Custom markdown system",
        "markdown_user": "Custom markdown user",
    }
    profile_path = tmp_path / "custom.json"
    profile_path.write_text(json.dumps(profile_data))

    profile = load_prompt_profile(str(profile_path))

    assert profile.name == "custom"
    assert profile.continuation_system == "Custom continuation system"
    assert profile.markdown_user == "Custom markdown user"


def test_load_json_profile_missing_fields(tmp_path: Path):
    """JSON profile with missing fields raises ValueError."""
    profile_data = {
        "name": "incomplete",
        "continuation_system": "sys",
        # missing other fields
    }
    profile_path = tmp_path / "incomplete.json"
    profile_path.write_text(json.dumps(profile_data))

    with pytest.raises(ValueError, match="missing required fields"):
        load_prompt_profile(str(profile_path))


def test_prompt_profile_is_frozen():
    """PromptProfile is immutable (frozen dataclass)."""
    profile = load_prompt_profile("contracts")

    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.name = "modified"
