#!/usr/bin/env python3
# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from llm_markdownify import convert

"""
Minimal quickstart for the installed llm-markdownify library.

Prereqs:
- pip install llm-markdownify
- Set provider env vars (e.g., OPENAI_API_KEY) / any LiteLLM provider

Edit input_pdf/output_md below, then run:
  python examples/quickstart.py
"""


def main() -> int:
    input_pdf = Path("input.pdf")
    output_md = Path("output.md")

    # Optional: override model/profile, or rely on your LiteLLM defaults
    result = convert(
        input_path=input_pdf,
        output_path=output_md,
        model="gpt-5-mini",
        profile="contracts",
        dpi=72,
        concurrency=4,
    )

    print(f"Wrote: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
