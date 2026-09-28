# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 API: markdownify() results, inputs, page selection and partial results."""

from __future__ import annotations

import asyncio
import functools
import http.server
import io
import threading
from pathlib import Path

import litellm
import pytest
from PIL import Image
from typer.testing import CliRunner

from llm_markdownify import amarkdownify, convert, markdownify
from llm_markdownify.cli import app
from llm_markdownify.sources import parse_pages, sniff_kind


def _is_continuation_check(kwargs) -> bool:
    return "CONTINUE_NEXT" in kwargs["messages"][0]["content"]


def _pdf_bytes(pages: int) -> bytes:
    buf = io.BytesIO()
    # A different shade per page, so each page image (and cache key) is distinct.
    imgs = [Image.new("RGB", (300, 400), (255, 255 - i, 255)) for i in range(pages)]
    imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def _pdf(path: Path, pages: int) -> Path:
    path.write_bytes(_pdf_bytes(pages))
    return path


class FakeModel:
    """Stands in for litellm.completion: answers '# page N' per image, fails chosen pages."""

    def __init__(self, fail_pages: set[int] | None = None):
        self.fail_pages = set(fail_pages or ())
        self.markdown_calls: list[int] = []
        self.lock = threading.Lock()
        self.page_of_url: dict[str, int] = {}

    def __call__(self, **kwargs):
        content = kwargs["messages"][1]["content"]
        urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
        if _is_continuation_check(kwargs):
            return {"choices": [{"message": {"content": "NONE"}, "finish_reason": "stop"}]}
        page = self.page_of_url[urls[0]]
        with self.lock:
            self.markdown_calls.append(page)
        if page in self.fail_pages:
            raise litellm.BadRequestError("image rejected", model="m", llm_provider="openai")
        return {
            "choices": [{"message": {"content": f"# page {page}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }


@pytest.fixture
def model(monkeypatch):
    """Install FakeModel and map each rendered page image to its page number."""
    from llm_markdownify import pager

    fake = FakeModel()
    real = pager._page_from_pil

    def tracking(index, img, max_side, fmt):
        page = real(index, img, max_side, fmt)
        fake.page_of_url[page.data_url] = index + 1
        return page

    monkeypatch.setattr(pager, "_page_from_pil", tracking)
    monkeypatch.setattr(litellm, "completion", fake)
    return fake


def test_markdownify_returns_markdown_pages_and_usage(model, tmp_path: Path):
    result = markdownify(_pdf(tmp_path / "a.pdf", 3), model="m")
    assert result.markdown == "# page 1\n\n# page 2\n\n# page 3\n"
    assert [p.number for p in result.pages] == [1, 2, 3]
    assert result.page(2).markdown == "# page 2"
    assert result.ok and result.failed_pages == []
    assert result.total_pages == 3
    assert result.usage.requests == 5  # 2 continuation checks + 3 pages
    assert result.usage.input_tokens == 300 and result.usage.output_tokens == 30
    assert result.to_dict()["ok"] is True


@pytest.mark.parametrize("kind", ["bytes", "file"])
def test_markdownify_accepts_bytes_and_file_objects(model, kind):
    data = _pdf_bytes(2)
    source = data if kind == "bytes" else io.BytesIO(data)
    result = markdownify(source, model="m")
    assert result.markdown == "# page 1\n\n# page 2\n"
    assert result.source in ("<bytes>", "<file>")


def test_markdownify_accepts_a_url(model, tmp_path: Path):
    _pdf(tmp_path / "doc.pdf", 2)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/doc.pdf"
        result = markdownify(url, model="m", allow_private_urls=True)
        with pytest.raises(PermissionError, match="private, local or reserved"):
            markdownify(url, model="m")  # refused by default
    finally:
        server.shutdown()
    assert result.markdown == "# page 1\n\n# page 2\n"
    assert result.source == url


def test_page_selection_only_pays_for_chosen_pages(model, tmp_path: Path):
    result = markdownify(_pdf(tmp_path / "a.pdf", 6), model="m", pages="2,5-")
    assert [p.number for p in result.pages] == [2, 5, 6]  # real page numbers are kept
    assert sorted(model.markdown_calls) == [2, 5, 6]
    assert result.total_pages == 6


def test_one_failed_page_no_longer_sinks_the_document(model, tmp_path: Path):
    model.fail_pages = {2}
    result = markdownify(_pdf(tmp_path / "a.pdf", 3), model="m")
    assert result.failed_pages == [2]
    assert not result.ok
    assert "# page 1" in result.markdown and "# page 3" in result.markdown
    assert "<!-- llm-markdownify: page 2 failed (BadRequestError). Rerun to retry. -->" in (
        result.markdown
    )
    assert "BadRequestError" in result.page(2).error
    assert any("1 page(s) failed: 2" in w for w in result.warnings)


def test_rerun_only_resends_failed_pages(model, tmp_path: Path):
    """The cache is on by default, so a rerun pays only for what failed."""
    pdf = _pdf(tmp_path / "a.pdf", 3)
    model.fail_pages = {2}
    markdownify(pdf, model="m", enable_grouping=False)
    model.fail_pages = set()
    model.markdown_calls.clear()
    result = markdownify(pdf, model="m", enable_grouping=False)
    assert model.markdown_calls == [2]
    assert result.ok
    assert result.usage.requests == 1 and result.usage.cached_responses == 2


def test_strict_mode_raises_on_a_failed_page(model, tmp_path: Path):
    model.fail_pages = {2}
    with pytest.raises(litellm.BadRequestError):
        markdownify(_pdf(tmp_path / "a.pdf", 3), model="m", strict=True)


def test_nothing_usable_raises_the_real_error(model, tmp_path: Path):
    model.fail_pages = {1, 2}
    with pytest.raises(litellm.BadRequestError, match="image rejected"):
        markdownify(_pdf(tmp_path / "a.pdf", 2), model="m")


def test_on_page_reports_each_page_as_it_finishes(model, tmp_path: Path):
    seen = []
    markdownify(_pdf(tmp_path / "a.pdf", 3), model="m", on_page=seen.append)
    assert sorted(p.number for p in seen) == [1, 2, 3]


def test_amarkdownify(model, tmp_path: Path):
    result = asyncio.run(amarkdownify(_pdf(tmp_path / "a.pdf", 1), model="m"))
    assert result.markdown == "# page 1\n"


def test_continuation_check_failure_does_not_lose_the_document(monkeypatch, model, tmp_path):
    def flaky(**kwargs):
        if _is_continuation_check(kwargs):
            raise litellm.BadRequestError("nope", model="m", llm_provider="openai")
        return model(**kwargs)

    monkeypatch.setattr(litellm, "completion", flaky)
    result = markdownify(_pdf(tmp_path / "a.pdf", 2), model="m", max_retries=0)
    assert result.ok and result.markdown == "# page 1\n\n# page 2\n"


def test_merged_pages_point_at_their_group(monkeypatch, model, tmp_path):
    def merging(**kwargs):
        if _is_continuation_check(kwargs):
            return {"choices": [{"message": {"content": "CONTINUE_NEXT"}, "finish_reason": "stop"}]}
        return {"choices": [{"message": {"content": "# merged table"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(litellm, "completion", merging)
    result = markdownify(_pdf(tmp_path / "a.pdf", 2), model="m")
    assert result.page(1).markdown == "# merged table"
    assert result.page(2).merged_into == 1 and result.page(2).markdown == ""


def test_convert_writes_partial_output_and_cli_exits_3(model, tmp_path: Path):
    pdf = _pdf(tmp_path / "a.pdf", 3)
    model.fail_pages = {3}
    out = convert(pdf, tmp_path / "a.md", model="m")
    assert "page 3 failed" in out.read_text()
    run = CliRunner().invoke(
        app, [str(pdf), "-o", str(tmp_path / "b.md"), "--model", "m", "--no-cache"]
    )
    assert run.exit_code == 3
    assert "page(s) failed: 3" in run.output
    assert "page 3 failed" in (tmp_path / "b.md").read_text()


def test_cli_pages_option(model, tmp_path: Path):
    pdf = _pdf(tmp_path / "a.pdf", 4)
    run = CliRunner().invoke(
        app, [str(pdf), "-o", str(tmp_path / "b.md"), "--model", "m", "--pages", "3-"]
    )
    assert run.exit_code == 0
    assert (tmp_path / "b.md").read_text() == "# page 3\n\n# page 4\n"


@pytest.mark.parametrize(
    "spec,expected",
    [("1", [0]), ("1-3", [0, 1, 2]), ("2,4-", [1, 3, 4]), ("-2", [0, 1]), (" 5 - 5 ", [4])],
)
def test_parse_pages(spec, expected):
    assert parse_pages(spec, total=5) == expected


@pytest.mark.parametrize("spec", ["0", "3-1", "6", "2-9", "", "a-b", "1,,2", "1 2", "-"])
def test_parse_pages_rejects_bad_input(spec):
    with pytest.raises(ValueError):
        parse_pages(spec, total=5)


def test_bad_page_selection_is_rejected_before_any_work(tmp_path: Path):
    run = CliRunner().invoke(
        app, [str(_pdf(tmp_path / "a.pdf", 1)), "-o", str(tmp_path / "b.md"), "--pages", "3-1"]
    )
    assert run.exit_code == 2


@pytest.mark.parametrize(
    "head,kind",
    [
        (b"%PDF-1.7", "pdf"),
        (b"\x89PNG\r\n\x1a\n", "image"),
        (b"\xff\xd8\xff\xe0", "image"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image"),
        (b"II*\x00", "image"),
        (b"hello world", None),
        (b"BM just some text", None),  # "BM" alone is not a BMP
    ],
)
def test_sniff_kind(head, kind):
    assert sniff_kind(head + b"\x00" * 16) == kind


def test_open_ended_page_selection_is_validated_instantly(tmp_path: Path):
    """Validating "40-" used to expand it against a huge placeholder page count and hang."""
    import time

    from llm_markdownify.config import MarkdownifyConfig

    started = time.monotonic()
    MarkdownifyConfig(pages="40-")
    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize(
    "url",
    ["http://169.254.169.254/latest/meta-data", "http://10.0.0.5/a.pdf", "http://localhost/a.pdf"],
)
def test_private_and_metadata_addresses_are_refused(url):
    from llm_markdownify.sources import _refuse_private_network

    with pytest.raises(PermissionError):
        _refuse_private_network(url)


def test_urls_are_redacted_in_results_and_logs():
    from llm_markdownify.sources import _redact

    url = "https://user:secret@bucket.s3.amazonaws.com/doc.pdf?X-Amz-Signature=abc#frag"
    assert _redact(url) == "https://bucket.s3.amazonaws.com/doc.pdf"


def test_html_served_at_a_pdf_url_is_refused():
    from llm_markdownify.sources import _kind_from_content_type

    assert _kind_from_content_type("text/html; charset=utf-8", "https://x.com/doc.pdf") is None
    assert _kind_from_content_type("image/svg+xml", "https://x.com/a.svg") is None
    assert _kind_from_content_type("application/octet-stream", "https://x.com/doc.pdf") == "pdf"


def test_zip_that_is_not_a_word_document_is_not_docx():
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", "<x/>")
    assert sniff_kind(buf.getvalue()) is None


def test_on_page_error_cancels_queued_pages(model, tmp_path: Path):
    def explode(page):
        raise RuntimeError("callback bug")

    with pytest.raises(RuntimeError, match="callback bug"):
        markdownify(
            _pdf(tmp_path / "a.pdf", 6),
            model="m",
            enable_grouping=False,
            concurrency=1,
            on_page=explode,
        )
    assert len(model.markdown_calls) <= 2  # the finished page plus at most one in flight


def test_cli_page_past_the_end_is_a_usage_error(model, tmp_path: Path):
    pdf = _pdf(tmp_path / "a.pdf", 2)
    run = CliRunner().invoke(
        app, [str(pdf), "-o", str(tmp_path / "b.md"), "--model", "m", "--pages", "5"]
    )
    assert run.exit_code == 2
    assert "past the end" in run.output


def test_word_document_zip_is_docx():
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
    assert sniff_kind(buf.getvalue()) == "docx"
