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
from typing import Any

from .api import markdownify
from .pager import count_pages
from .sources import Document, is_url, load_source

INSTRUCTIONS = (
    "Converts PDFs and images (scans, photos, screenshots) to clean Markdown with a vision model. "
    "Call document_info first to learn the page count, then convert_document for a page range. "
    "Long documents are returned in chunks; the reply says which pages are left."
)


class _Sandbox:
    """Resolves agent-supplied sources against the allowed roots."""

    def __init__(self, roots: list[Path]) -> None:
        self.roots = [r.expanduser().resolve() for r in roots]

    def resolve(self, source: str) -> str | Path:
        if is_url(source):
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
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return wrapper


def _default_roots() -> list[Path]:
    """The working folder, unless it is the filesystem root or the home folder (Claude Desktop, for
    one, starts servers in /): then an explicit --root is required."""
    cwd = Path.cwd().resolve()
    if cwd in (Path(cwd.anchor), Path.home().resolve()):
        raise SystemExit(
            f"markdownify-mcp is running in {cwd}; pass --root <folder> to choose which folder "
            "the agent may read documents from."
        )
    return [cwd]


def build_server(
    roots: list[Path] | None = None,
    model: str | None = None,
    chunk_pages: int = 20,
    allow_private_urls: bool = False,
    allow_model_override: bool = False,
):
    from mcp.server.mcpserver import MCPServer

    if chunk_pages < 1:
        raise ValueError("chunk_pages must be at least 1")
    sandbox = _Sandbox(roots or _default_roots())
    server = MCPServer("llm-markdownify", instructions=INSTRUCTIONS)
    default_model = model

    def _load(source: str) -> Document:
        """Check the source, then read it into memory once, so the file that was checked is the
        file that gets converted (no swap between the check and the read)."""
        doc = load_source(sandbox.resolve(source), allow_private_urls=allow_private_urls)
        if doc.data is None and doc.path is not None:
            doc.data = doc.path.read_bytes()
        return doc

    @server.tool()
    @_explain_errors
    def document_info(source: str) -> dict[str, Any]:
        """Page count and type of a document, without calling a model.

        source: a file path (relative to the allowed folder) or an http(s) URL.
        """
        doc = _load(source)
        pages = None
        if doc.kind in ("pdf", "image"):
            pages = count_pages(doc.kind, doc.data)
        return {"source": doc.name, "type": doc.kind, "pages": pages}

    @server.tool()
    @_explain_errors
    def convert_document(source: str, pages: str | None = None, model: str | None = None) -> str:
        """Convert a PDF or image to Markdown.

        source: a file path (relative to the allowed folder) or an http(s) URL.
        pages: 1-based selection like "1-5,12". Default: the first chunk of pages. The reply ends
            with a note saying which pages are left.
        model: a different vision model, only if the server was started with
            --allow-model-override.
        """
        if model and not allow_model_override:
            raise PermissionError(
                "This server does not let the agent choose the model (start it with "
                "--allow-model-override to allow that)"
            )
        doc = _load(source)
        total = count_pages(doc.kind, doc.data) if doc.kind in ("pdf", "image") else None
        selection = pages or (f"1-{min(total, chunk_pages)}" if total else None)
        result = markdownify(
            doc.data,
            model=model or default_model,
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


def main(argv: list[str] | None = None) -> None:
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
        "--allow-model-override",
        action="store_true",
        help="Let the agent pick a different model per call (off: the agent cannot run other, "
        "possibly more expensive models on your keys)",
    )
    parser.add_argument(
        "--allow-private-urls",
        action="store_true",
        help="Allow URLs on private or local addresses (refused by default)",
    )
    args = parser.parse_args(argv)
    if args.chunk_pages < 1:
        parser.error("--chunk-pages must be at least 1")
    try:
        server = build_server(
            args.root,
            args.model,
            args.chunk_pages,
            args.allow_private_urls,
            args.allow_model_override,
        )
    except ImportError as e:  # the mcp package is an optional extra
        raise SystemExit(
            f'markdownify-mcp needs the MCP extra: pip install "llm-markdownify[mcp]" ({e})'
        ) from e
    server.run()  # stdio: stdout carries the protocol, so the library must not print


if __name__ == "__main__":  # pragma: no cover
    main()
