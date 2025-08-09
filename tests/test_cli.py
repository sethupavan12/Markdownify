# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from llm_markdownify.cli import app


def test_cli_invokes_markdownifier(monkeypatch, tmp_path: Path):
    input_pdf = tmp_path / "in.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")  # minimal placeholder; pager isn't invoked here
    output_md = tmp_path / "out.md"

    called = {}

    class FakeMarkdownifier:
        def __init__(self, cfg):
            called["cfg"] = cfg

        def run(self):
            output_md.write_text("# ok\n")
            return output_md

    monkeypatch.setattr("llm_markdownify.cli.Markdownifier", FakeMarkdownifier)

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(input_pdf), "-o", str(output_md), "--dpi", "150"])  # noqa: S607

    assert result.exit_code == 0
    assert output_md.read_text() == "# ok\n"
    assert called["cfg"].dpi == 150
