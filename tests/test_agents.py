# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Agent-facing surfaces: stdout/JSON CLI output and the MCP server."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import litellm
import pytest
from PIL import Image
from typer.testing import CliRunner

from llm_markdownify.cli import app

mcp = pytest.importorskip("mcp")


def _pdf(path: Path, pages: int) -> Path:
    imgs = [Image.new("RGB", (300, 400), (255, 255 - i, 255)) for i in range(pages)]
    imgs[0].save(path, format="PDF", save_all=True, append_images=imgs[1:])
    return path


@pytest.fixture
def model(monkeypatch):
    """Fake vision model: numbers pages by the order they are first seen within one request."""
    from llm_markdownify import pager

    page_of_url: dict[str, int] = {}
    real = pager._page_from_pil

    def tracking(index, img, max_side, fmt):
        page = real(index, img, max_side, fmt)
        page_of_url[page.data_url] = index + 1
        return page

    def completion(**kwargs):
        if "CONTINUE_NEXT" in kwargs["messages"][0]["content"]:
            return {"choices": [{"message": {"content": "NONE"}, "finish_reason": "stop"}]}
        url = kwargs["messages"][1]["content"][1]["image_url"]["url"]
        text = f"# page {page_of_url[url]}"
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}

    monkeypatch.setattr(pager, "_page_from_pil", tracking)
    monkeypatch.setattr(litellm, "completion", completion)


# --- CLI output ------------------------------------------------------------------------------


def test_cli_prints_markdown_to_stdout_without_o(model, tmp_path: Path):
    run = CliRunner().invoke(app, [str(_pdf(tmp_path / "a.pdf", 2)), "--model", "m", "-q"])
    assert run.exit_code == 0
    assert run.stdout == "# page 1\n\n# page 2\n"


def test_cli_dash_means_stdout(model, tmp_path: Path):
    run = CliRunner().invoke(
        app, [str(_pdf(tmp_path / "a.pdf", 1)), "-o", "-", "--model", "m", "-q"]
    )
    assert run.exit_code == 0
    assert run.stdout == "# page 1\n"


def test_cli_json_output(model, tmp_path: Path):
    run = CliRunner().invoke(
        app, [str(_pdf(tmp_path / "a.pdf", 2)), "--model", "m", "--json", "-q"]
    )
    assert run.exit_code == 0
    data = json.loads(run.stdout)
    assert data["markdown"] == "# page 1\n\n# page 2\n"
    assert [p["number"] for p in data["pages"]] == [1, 2]
    assert data["ok"] is True and data["failed_pages"] == []
    assert data["usage"]["requests"] == 3


def test_cli_json_to_a_file(model, tmp_path: Path):
    out = tmp_path / "result.json"
    run = CliRunner().invoke(
        app, [str(_pdf(tmp_path / "a.pdf", 1)), "--model", "m", "--json", "-o", str(out)]
    )
    assert run.exit_code == 0 and run.stdout == ""
    assert json.loads(out.read_text())["markdown"] == "# page 1\n"


def test_cli_rejects_non_markdown_output_without_json(tmp_path: Path):
    run = CliRunner().invoke(app, [str(_pdf(tmp_path / "a.pdf", 1)), "-o", str(tmp_path / "x.txt")])
    assert run.exit_code == 2


def test_cli_accepts_urls(model, tmp_path: Path):
    import functools
    import http.server
    import threading

    _pdf(tmp_path / "doc.pdf", 1)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/doc.pdf"
        refused = CliRunner().invoke(app, [url, "--model", "m", "-q"])
        allowed = CliRunner().invoke(app, [url, "--model", "m", "-q", "--allow-private-urls"])
    finally:
        server.shutdown()
    assert refused.exit_code == 1 and "private, local or reserved" in refused.output
    assert allowed.exit_code == 0 and allowed.stdout == "# page 1\n"


# --- MCP server ------------------------------------------------------------------------------


def _call(server, tool: str, **arguments):
    result = asyncio.run(server.call_tool(tool, arguments))
    text = "".join(getattr(block, "text", "") for block in result.content)
    return result, text


def test_mcp_lists_both_tools(tmp_path: Path):
    from llm_markdownify.mcp_server import build_server

    tools = asyncio.run(build_server([tmp_path]).list_tools())
    assert sorted(t.name for t in tools) == ["convert_document", "document_info"]


def test_mcp_document_info_needs_no_model(tmp_path: Path, monkeypatch):
    from llm_markdownify.mcp_server import build_server

    monkeypatch.setattr(litellm, "completion", lambda **kw: pytest.fail("no model call expected"))
    _pdf(tmp_path / "a.pdf", 7)
    result, text = _call(build_server([tmp_path]), "document_info", source="a.pdf")
    assert not result.is_error
    assert json.loads(text) == {
        "source": str((tmp_path / "a.pdf").resolve()),
        "type": "pdf",
        "pages": 7,
    }


def test_mcp_converts_in_chunks_and_says_what_is_left(model, tmp_path: Path):
    from llm_markdownify.mcp_server import build_server

    _pdf(tmp_path / "long.pdf", 5)
    server = build_server([tmp_path], model="m", chunk_pages=2)
    _, first = _call(server, "convert_document", source="long.pdf")
    assert first.startswith("# page 1\n\n# page 2")
    assert "Converted pages 1-2 of 5." in first
    assert 'call convert_document again with pages="3-4"' in first
    _, rest = _call(server, "convert_document", source="long.pdf", pages="3-5")
    assert "# page 5" in rest and "not converted yet" not in rest


def test_mcp_refuses_paths_outside_the_allowed_folder(tmp_path: Path):
    from mcp.server.mcpserver.exceptions import ToolError

    from llm_markdownify.mcp_server import build_server

    allowed = tmp_path / "docs"
    allowed.mkdir()
    secret = _pdf(tmp_path / "secret.pdf", 1)
    (allowed / "link.pdf").symlink_to(secret)  # a symlink cannot escape either
    server = build_server([allowed])
    for source in (str(secret), "../secret.pdf", "link.pdf"):
        with pytest.raises(ToolError, match="outside the allowed folders"):
            _call(server, "document_info", source=source)


def test_mcp_server_over_real_stdio(tmp_path: Path):
    """Start markdownify-mcp as a subprocess and talk to it the way Claude Code would."""
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client

    _pdf(tmp_path / "a.pdf", 3)

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "llm_markdownify.mcp_server", "--root", str(tmp_path)],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                info = await session.call_tool("document_info", {"source": "a.pdf"})
                denied = await session.call_tool("document_info", {"source": "/etc/hosts"})
                return [t.name for t in tools.tools], info, denied

    names, info, denied = asyncio.run(run())
    assert sorted(names) == ["convert_document", "document_info"]
    assert json.loads(info.content[0].text)["pages"] == 3
    # The agent is told why, not just "Error executing tool".
    assert denied.is_error and "outside the allowed folders" in denied.content[0].text


def test_ranges_helper():
    from llm_markdownify.mcp_server import _ranges

    assert _ranges([1, 2, 3, 7, 9, 10]) == "1-3, 7, 9-10"
