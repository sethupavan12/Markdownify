# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""MCP server so agents (Claude Code, Claude Desktop, Cursor, ...) can convert documents as a tool.

    pip install "llm-markdownify[mcp]"
    markdownify-mcp --root ~/docs --model gpt-5.4-mini

Two tools: `document_info` (page count, no model call) and `convert_document` (converts at most
`--chunk-pages` pages per call and says which pages are left, so long documents do not flood the
agent's context). Local paths must be inside an allowed root (the working directory by default);
URLs to private or local addresses are refused unless `--allow-private-urls` is given.
"""

from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path
from typing import Any, Optional

from .api import markdownify
from .pager import count_pages
from .sources import load_source

INSTRUCTIONS = (
    "Converts PDFs and images (scans, photos, screenshots) to clean Markdown with a vision model. "
    "Call document_info first to learn the page count, then convert_document for a page range. "
    "Long documents are returned in chunks; the reply says which pages are left."
)


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


class _Sandbox:
    """Resolves agent-supplied sources against the allowed roots."""

    def __init__(self, roots: list[Path]) -> None:
        self.roots = [r.expanduser().resolve() for r in roots]

    def resolve(self, source: str) -> str | Path:
        if _is_url(source):
            return source
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = self.roots[0] / path
        path = path.resolve()  # follows symlinks, so a link cannot escape the roots
        if not any(path == root or path.is_relative_to(root) for root in self.roots):
            allowed = ", ".join(str(r) for r in self.roots)
            raise PermissionError(f"{source} is outside the allowed folders ({allowed})")
        return path


def _explain_errors(fn):
    """Report failures to the agent with their reason. The MCP SDK hides the message of any
    exception that is not a ToolError, which would leave the agent guessing."""
    from mcp.server.mcpserver.exceptions import ToolError

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as e:  # noqa: BLE001 - every failure becomes a readable tool error
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return wrapper


def build_server(
    roots: Optional[list[Path]] = None,
    model: Optional[str] = None,
    chunk_pages: int = 20,
    allow_private_urls: bool = False,
):
    from mcp.server.mcpserver import MCPServer

    sandbox = _Sandbox(roots or [Path.cwd()])
    server = MCPServer("llm-markdownify", instructions=INSTRUCTIONS)

    @server.tool()
    @_explain_errors
    def document_info(source: str) -> dict[str, Any]:
        """Page count and type of a document, without calling a model.

        source: a file path (relative to the allowed folder) or an http(s) URL.
        """
        doc = load_source(sandbox.resolve(source), allow_private_urls=allow_private_urls)
        pages = None
        if doc.kind in ("pdf", "image"):
            pages = count_pages(doc.kind, doc.data if doc.data is not None else doc.path)
        return {"source": doc.name, "type": doc.kind, "pages": pages}

    @server.tool()
    @_explain_errors
    def convert_document(
        source: str, pages: Optional[str] = None, model_name: Optional[str] = None
    ) -> str:
        """Convert a PDF or image to Markdown.

        source: a file path (relative to the allowed folder) or an http(s) URL.
        pages: 1-based selection like "1-5,12". Default: the first chunk of pages. The reply ends
            with a note saying which pages are left.
        model_name: override the server's default vision model.
        """
        resolved = sandbox.resolve(source)
        doc = load_source(resolved, allow_private_urls=allow_private_urls)
        total = (
            count_pages(doc.kind, doc.data if doc.data is not None else doc.path)
            if doc.kind in ("pdf", "image")
            else None
        )
        selection = pages or (f"1-{min(total, chunk_pages)}" if total else None)
        result = markdownify(
            doc.data if doc.data is not None else doc.path,
            model=model_name or model,
            pages=selection,
            allow_private_urls=allow_private_urls,
        )
        done = [p.number for p in result.pages]
        notes = [f"Converted pages {_ranges(done)} of {result.total_pages}."]
        if pages is None and done and max(done) < result.total_pages:
            nxt = max(done) + 1
            end = min(result.total_pages, nxt + chunk_pages - 1)
            left = _ranges(list(range(nxt, result.total_pages + 1)))
            nxt_chunk = _ranges(list(range(nxt, end + 1)))
            notes.append(
                f"{'Page' if nxt == result.total_pages else 'Pages'} {left} "
                f"{'is' if nxt == result.total_pages else 'are'} not converted yet; call "
                f'convert_document again with pages="{nxt_chunk}".'
            )
        if result.failed_pages:
            notes.append(
                f"Pages {_ranges(result.failed_pages)} failed and are marked in the text; "
                "calling again retries only those (finished pages are cached)."
            )
        return result.markdown.rstrip() + "\n\n---\n" + " ".join(notes) + "\n"

    return server


def _ranges(numbers: list[int]) -> str:
    """[1, 2, 3, 7] -> "1-3, 7"."""
    parts: list[str] = []
    for n in sorted(set(numbers)):
        if parts and n == int(parts[-1].split("-")[-1]) + 1:
            parts[-1] = f"{parts[-1].split('-')[0]}-{n}"
        else:
            parts.append(str(n))
    return ", ".join(parts)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="markdownify-mcp", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        help="Folder the agent may read documents from (repeatable). Default: the working folder.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MARKDOWNIFY_MODEL"),
        help="Vision model (LiteLLM name). Default: $LLM_MARKDOWNIFY_MODEL or the library default.",
    )
    parser.add_argument(
        "--chunk-pages", type=int, default=20, help="Max pages per convert_document call (20)"
    )
    parser.add_argument(
        "--allow-private-urls",
        action="store_true",
        help="Allow URLs on private or local addresses (refused by default)",
    )
    args = parser.parse_args(argv)
    try:
        server = build_server(args.root, args.model, args.chunk_pages, args.allow_private_urls)
    except ImportError as e:  # the mcp package is an optional extra
        raise SystemExit(
            f'markdownify-mcp needs the MCP extra: pip install "llm-markdownify[mcp]" ({e})'
        )
    server.run()  # stdio: stdout carries the protocol, so the library must not print


if __name__ == "__main__":  # pragma: no cover
    main()
