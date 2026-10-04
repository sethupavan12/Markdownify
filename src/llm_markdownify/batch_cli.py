# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""`markdownify-batch`: convert large document sets through the OpenAI Batch API."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer

from . import batch as batch_api
from .logging import enable_console_logging

app = typer.Typer(
    help=(
        "Convert many documents through the OpenAI Batch API or Anthropic Message Batches: about "
        "half the cost of normal requests, finished within 24 hours. Submit, then collect."
    ),
    pretty_exceptions_show_locals=False,  # locals hold API keys and page images
    add_completion=False,
    no_args_is_help=True,
)


def _report(result: batch_api.CollectResult) -> None:
    typer.echo(f"Wrote {len(result.written)} document(s).")
    if result.pending_batches:
        typer.echo(f"{result.pending_batches} batch(es) still running; run collect again later.")
    for batch_id, error in result.batch_errors.items():
        typer.echo(f"  batch {batch_id} was rejected: {error}", err=True)
    for path, error in list(result.document_errors.items())[:10]:
        typer.echo(f"  could not read {path}: {error}", err=True)
    page_gaps = {o: p for o, p in result.incomplete.items() if p}
    if page_gaps:
        missing = sum(len(p) for p in page_gaps.values())
        typer.echo(f"{len(page_gaps)} document(s) incomplete ({missing} page(s) missing).")
        for output, pages in list(page_gaps.items())[:10]:
            typer.echo(f"  {output}: pages {', '.join(map(str, pages[:20]))}")
    for custom_id, error in list(result.failed_pages.items())[:10]:
        typer.echo(f"  failed {custom_id}: {error}", err=True)
    if result.resubmitted:
        typer.echo(f"Resubmitted {result.resubmitted} failed page(s) as a new batch.")


def _exit_for(result: batch_api.CollectResult) -> None:
    """Exit 1 when the job has finished but some documents could not be written."""
    if not result.pending_batches and not result.resubmitted and result.incomplete:
        raise typer.Exit(code=1)


def _fail(e: Exception) -> NoReturn:
    typer.secho(f"Error: {type(e).__name__}: {e}", err=True, fg=typer.colors.RED)
    raise typer.Exit(code=1)


@app.command()
def submit(
    inputs: list[str] = typer.Argument(..., help="PDF/image files or directories"),
    out: str = typer.Option(..., "-o", "--out", help="Output directory (also holds job state)"),
    model: str = typer.Option(
        "gpt-5.4-mini", help="OpenAI (gpt-5.4-mini) or Anthropic (anthropic/claude-opus-5) model"
    ),
    profile: str | None = typer.Option(None, help="Prompt profile name or JSON file"),
    dpi: int = typer.Option(200, help="PDF render DPI"),
    max_image_px: int = typer.Option(2048, help="Longest side of each page image"),
    max_tokens: int = typer.Option(16000, help="Max output tokens per page"),
    reasoning_effort: str | None = typer.Option(None, help="e.g. none, low, medium, high"),
    api_base: str | None = typer.Option(None, help="Custom base URL for the provider API"),
    wait: bool = typer.Option(False, "--wait", help="Block until done and write the Markdown"),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Render every page and submit it as a batch job."""
    enable_console_logging("quiet" if quiet else "normal")
    try:
        batch_api.submit_batch(
            inputs,
            out,
            model=model,
            profile=profile,
            dpi=dpi,
            max_image_px=max_image_px,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            api_base=api_base,
        )
        if wait:
            result = batch_api.wait_batch(out, retry_failed=True)
        else:
            typer.echo(f"Submitted. Check with: markdownify-batch status {out}")
            return
    except Exception as e:  # noqa: BLE001 - one clean line, never a traceback with locals
        _fail(e)
    _report(result)
    _exit_for(result)


@app.command()
def status(out: str = typer.Argument(..., help="Output directory of a submitted job")) -> None:
    """Show progress of every batch in the job."""
    enable_console_logging("normal")
    try:
        st = batch_api.batch_status(out)
    except Exception as e:  # noqa: BLE001 - reported as one clean line
        _fail(e)
    for b in st.batches:
        typer.echo(
            f"{b['id']}  {b['status']:<11} {b['completed']}/{b['total']} done, {b['failed']} failed"
        )
    typer.echo(
        f"Total: {st.completed}/{st.total} pages done, {st.failed} failed"
        + (" (finished)" if st.done else "")
    )


@app.command()
def collect(
    out: str = typer.Argument(..., help="Output directory of a submitted job"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="Resubmit failed pages"),
) -> None:
    """Download finished pages and write every complete document. Safe to re-run."""
    enable_console_logging("normal")
    try:
        result = batch_api.collect_batch(out, retry_failed=retry_failed)
    except Exception as e:  # noqa: BLE001 - reported as one clean line
        _fail(e)
    _report(result)
    _exit_for(result)


@app.command("wait")
def wait_cmd(
    out: str = typer.Argument(..., help="Output directory of a submitted job"),
    poll_seconds: float = typer.Option(60.0, help="Seconds between status checks"),
    retry_failed: bool = typer.Option(True, "--retry-failed/--no-retry-failed"),
) -> None:
    """Block until the job finishes, then collect."""
    enable_console_logging("normal")
    try:
        result = batch_api.wait_batch(
            Path(out), poll_seconds=poll_seconds, retry_failed=retry_failed
        )
    except Exception as e:  # noqa: BLE001 - reported as one clean line
        _fail(e)
    _report(result)
    _exit_for(result)


if __name__ == "__main__":  # pragma: no cover
    app()
