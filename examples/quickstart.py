#!/usr/bin/env python3
# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

from llm_markdownify import convert


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PDF to Markdown using the llm-markdownify library (installed distribution).\n"
            "Ensure your provider env vars are set (e.g., OPENAI_API_KEY) before running."
        )
    )
    parser.add_argument("input_pdf", type=Path, help="Path to input PDF file")
    parser.add_argument("output_md", type=Path, help="Path to output Markdown file")
    parser.add_argument(
        "--model",
        default=None,
        help="LiteLLM model (e.g., gpt-4.1-mini, azure/<deployment>, gemini/gemini-2.5-flash)",
    )
    parser.add_argument("--profile", default="contracts", help="Prompt profile name or JSON path")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI (default 200)")
    parser.add_argument(
        "--no-grouping",
        action="store_true",
        help="Disable LLM-based grouping (debug per-page output)",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="LLM max tokens")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max parallel LLM calls across page groups",
    )

    args = parser.parse_args()

    path = convert(
        input_path=args.input_pdf,
        output_path=args.output_md,
        model=args.model,
        dpi=args.dpi,
        max_group_pages=3,
        enable_grouping=not args.no_grouping,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        profile=args.profile,
        allow_docx=False,
    )

    print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
