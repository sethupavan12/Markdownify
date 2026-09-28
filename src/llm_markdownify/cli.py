# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys

from pathlib import Path
from typing import Any, Optional

import typer
from pydantic import ValidationError

from . import __version__
from .config import MarkdownifyConfig
from .logging import enable_console_logging
from .logging import get_logger
from .markdownifier import Markdownifier
from .sources import PageSelectionError
from .sources import is_url as source_is_url

logger = get_logger("llm_markdownify.cli")

# Exit codes: 0 all pages converted, 1 nothing usable, 2 bad arguments, 3 some pages failed.
EXIT_PARTIAL = 3

app = typer.Typer(
    help=(
        "Convert documents (PDF/DOCX) or images (PNG/JPG/WEBP/TIFF...) to Markdown using "
        "Vision LLMs via LiteLLM."
    ),
    # Never print local variables on errors: they include API keys and base64 page images.
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


def _write_stdout(text: str) -> None:
    """Write as UTF-8 whatever the terminal's locale, and exit quietly if the reader has gone away
    (e.g. piped into `head`)."""
    try:
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            buffer.write(text.encode("utf-8"))
        else:
            sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        # Point stdout at devnull so Python's shutdown flush does not raise again.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        raise typer.Exit(code=0)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def run(
    input_path: str = typer.Argument(
        ...,
        help="Path to input .pdf (preferred), image, or .docx (with --allow-docx)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file. Omit (or use -) to print to stdout; logs always go to stderr.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print the full result as JSON: markdown, per-page results, usage, warnings",
    ),
    model: Optional[str] = typer.Option(
        None, help="LiteLLM model, e.g. gpt-5.4-mini, azure/<deployment>, gemini/gemini-2.5-flash"
    ),
    dpi: Optional[int] = typer.Option(
        None, help="DPI for rendering PDF pages [default: 200]. Ignored for image inputs."
    ),
    max_image_px: Optional[int] = typer.Option(
        None, help="Longest side in pixels of each page image sent to the model [default: 2048]"
    ),
    image_format: Optional[str] = typer.Option(
        None, help="Page image encoding: jpeg or png [default: jpeg]"
    ),
    pages: Optional[str] = typer.Option(
        None, "--pages", help='Pages to convert, e.g. "1-5,12,40-" [default: all]'
    ),
    allow_private_urls: bool = typer.Option(
        False,
        "--allow-private-urls",
        help="Allow URL input on private or local addresses (refused by default)",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail if any page fails, instead of writing the rest with the failed pages marked",
    ),
    max_group_pages: Optional[int] = typer.Option(
        None, help="Max pages to merge for continued content [default: 3]"
    ),
    grouping: Optional[bool] = typer.Option(
        None,
        "--grouping/--no-grouping",
        help="Enable LLM-based grouping of continued content [default: on]",
    ),
    temperature: Optional[float] = typer.Option(
        None, help="LLM temperature [default: provider default]"
    ),
    max_tokens: Optional[int] = typer.Option(None, help="LLM max output tokens [default: 16000]"),
    reasoning_effort: Optional[str] = typer.Option(
        None, help="Reasoning effort for reasoning models (e.g. none, low, medium, high)"
    ),
    api_base: Optional[str] = typer.Option(
        None, help="Custom API base URL (e.g. a local vLLM/Ollama OpenAI-compatible server)"
    ),
    concurrency: Optional[int] = typer.Option(
        None,
        help="Max concurrent LLM requests for page groups [default: 4]. Higher is faster but "
        "may hit provider rate limits.",
    ),
    grouping_concurrency: Optional[int] = typer.Option(
        None,
        help="Max concurrent continuation checks (defaults to --concurrency)",
    ),
    profile: Optional[str] = typer.Option(
        None, help="Prompt profile name ('generic', 'contracts') or path to a JSON profile"
    ),
    allow_docx: bool = typer.Option(
        False, help="Allow DOCX via Word/COM conversion (not recommended). Prefer PDFs."
    ),
    max_retries: Optional[int] = typer.Option(
        None, help="Max retries for transient LLM errors [default: 5]"
    ),
    retry_delay: Optional[float] = typer.Option(
        None, help="Base delay in seconds for exponential backoff [default: 1.0]"
    ),
    rate_limit: Optional[int] = typer.Option(
        None, "--rate-limit", help="Max requests per minute (default: no limit)"
    ),
    cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Cache answers on disk so a rerun only pays for pages that failed or changed",
    ),
    cache_dir: Optional[str] = typer.Option(
        None, help="Directory for response cache (defaults to ~/.cache/llm-markdownify)"
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging (debug level)"
    ),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress non-error output"),
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit"
    ),
) -> None:
    log_level = "quiet" if quiet else "verbose" if verbose else "normal"

    llm_kwargs: dict[str, Any] = {}
    if reasoning_effort:
        llm_kwargs["reasoning_effort"] = reasoning_effort
    if api_base:
        llm_kwargs["api_base"] = api_base

    options: dict[str, Any] = dict(
        model=model,
        dpi=dpi,
        max_image_px=max_image_px,
        image_format=image_format,
        max_group_pages=max_group_pages,
        enable_grouping=grouping,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
        grouping_concurrency=grouping_concurrency,
        allow_docx=allow_docx,
        llm_kwargs=llm_kwargs or None,
        max_retries=max_retries,
        retry_delay=retry_delay,
        rate_limit_rpm=rate_limit,
        enable_cache=cache,
        pages=pages,
        strict=strict,
        allow_private_urls=allow_private_urls,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    try:
        is_url = source_is_url(input_path)
        cfg = MarkdownifyConfig(
            input_path=None if is_url else Path(input_path),
            **{k: v for k, v in options.items() if v is not None},
        )
        to_stdout = output in (None, "-")
        if (
            not to_stdout
            and not as_json
            and Path(output).suffix.lower() not in (".md", ".markdown")
        ):
            raise typer.BadParameter(
                "must end in .md or .markdown (or use --json)", param_hint="-o"
            )
    except ValidationError as e:
        for err in e.errors():
            field = ".".join(str(p) for p in err["loc"]) or "input"
            typer.secho(f"Error: {field}: {err['msg']}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    enable_console_logging(log_level)
    try:
        result = Markdownifier(cfg, profile=profile, show_progress=not quiet).convert(
            input_path if is_url else cfg.input_path
        )
    except PageSelectionError as e:  # a bad argument, found once the page count is known
        typer.secho(f"Error: pages: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)
    except Exception as e:  # show one clean line; full tracebacks only with --verbose
        if verbose:
            raise
        typer.secho(f"Error: {type(e).__name__}: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    text = json.dumps(result.to_dict(), indent=2) + "\n" if as_json else result.markdown
    if to_stdout:
        _write_stdout(text)
    else:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        logger.info("Wrote %s to %s", "JSON" if as_json else "Markdown", out_path)
    if result.failed_pages:
        for warning in result.warnings:
            typer.secho(f"Warning: {warning}", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(code=EXIT_PARTIAL)


if __name__ == "__main__":  # pragma: no cover
    app()
