# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bugs found in the 2026 audit. Each test names the failure it guards."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
import pytest
from PIL import Image
from typer.testing import CliRunner

from llm_markdownify import llm
from llm_markdownify.api import convert
from llm_markdownify.cache import configure_cache
from llm_markdownify.cli import app
from llm_markdownify.config import MarkdownifyConfig
from llm_markdownify.pager import load_document_pages
from llm_markdownify.prompt_profiles import load_prompt_profile


def _response(content, finish_reason="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


@pytest.fixture(autouse=True)
def _isolated_llm_state():
    # llm/cache settings are module globals; reset them so tests cannot leak into each other.
    configure_cache(enabled=False)
    llm.configure_llm(max_retries=2, retry_delay=0.1)
    yield
    llm.configure_llm()
    configure_cache(enabled=False)


def _make_pdf(path: Path, pages: int = 3) -> None:
    imgs = [Image.new("RGB", (612, 792), "white") for _ in range(pages)]
    imgs[0].save(path, format="PDF", save_all=True, append_images=imgs[1:])


def test_concurrent_pdf_rendering_does_not_crash(tmp_path: Path):
    """PDFium is not thread-safe; concurrent convert() calls used to segfault the interpreter."""
    pdfs = []
    for i in range(8):
        pdfs.append(tmp_path / f"d{i}.pdf")
        _make_pdf(pdfs[-1])
    with ThreadPoolExecutor(8) as ex:
        counts = list(ex.map(lambda p: len(load_document_pages(p, dpi=150)), pdfs * 4))
    assert counts == [3] * 32


def test_large_pages_are_capped_to_max_side(tmp_path: Path):
    """Large scans at high DPI exceeded provider image limits and failed the whole page."""
    pdf = tmp_path / "big.pdf"
    Image.new("RGB", (3000, 4000), "white").save(pdf, format="PDF", resolution=72)
    page = load_document_pages(pdf, dpi=300, max_side=2048)[0]
    assert max(page.width, page.height) == 2048


def test_transparent_png_is_flattened_on_white(tmp_path: Path):
    """convert('RGB') turned transparent backgrounds black, hiding black text."""
    path = tmp_path / "t.png"
    Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(path)
    page = load_document_pages(path, dpi=72, image_format="png")[0]
    with Image.open(__import__("io").BytesIO(page.content)) as img:
        assert img.convert("RGB").getpixel((5, 5)) == (255, 255, 255)


def test_multipage_tiff_yields_one_page_per_frame(tmp_path: Path):
    path = tmp_path / "scan.tiff"
    frames = [Image.new("RGB", (50, 50), c) for c in ("white", "gray", "black")]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    assert [p.index for p in load_document_pages(path, dpi=72)] == [0, 1, 2]


def test_markdown_fence_is_stripped():
    assert llm.strip_markdown_fence("```markdown\n# Title\n\ntext\n```") == "# Title\n\ntext"
    assert llm.strip_markdown_fence("# Title\n\n```py\nx\n```") == "# Title\n\n```py\nx\n```"


@pytest.mark.parametrize(
    "answer,label",
    [
        ("CONTINUE_NEXT", "CONTINUE_NEXT"),
        ("continue next.", "CONTINUE_NEXT"),
        ("NONE", "NONE"),
        ("", "NONE"),
    ],
)
def test_continuation_label_parsing(answer, label):
    assert llm.parse_continuation_label(answer) == label


def test_continuation_check_does_not_cap_tokens(monkeypatch):
    """max_tokens=4 left reasoning models with an empty answer, so grouping never triggered."""
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _response("CONTINUE_NEXT")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    profile = load_prompt_profile("generic")
    assert llm.assess_continuation("m", "data:a", "data:b", profile) == "CONTINUE_NEXT"
    assert "max_tokens" not in seen
    assert "temperature" not in seen


def test_empty_content_is_retried_then_fails(monkeypatch):
    """None content used to be written into the document as the literal string 'None'."""
    calls = []

    def fake_completion(**kwargs):
        calls.append(1)
        return _response(None, None)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    with pytest.raises(llm.EmptyResponseError):
        llm.generate_markdown("m", ["data:a"], load_prompt_profile("generic"))
    assert len(calls) == 3  # first attempt + 2 retries


def test_empty_content_recovers_on_retry(monkeypatch):
    answers = iter([_response("", None), _response("```markdown\n# ok\n```")])
    monkeypatch.setattr(litellm, "completion", lambda **kw: next(answers))
    assert llm.generate_markdown("m", ["data:a"], load_prompt_profile("generic")) == "# ok"


def test_auth_errors_are_not_retried(monkeypatch):
    """Every exception used to be retried, so a bad key cost several slow attempts."""
    calls = []

    def fake_completion(**kwargs):
        calls.append(1)
        raise litellm.AuthenticationError("bad key", llm_provider="openai", model="m")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    with pytest.raises(litellm.AuthenticationError):
        llm.generate_markdown("m", ["data:a"], load_prompt_profile("generic"))
    assert len(calls) == 1


def test_rate_limit_errors_are_retried(monkeypatch):
    answers = iter(["rate", "ok"])

    def fake_completion(**kwargs):
        if next(answers) == "rate":
            raise litellm.RateLimitError("slow down", llm_provider="openai", model="m")
        return _response("# fine")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    assert llm.generate_markdown("m", ["data:a"], load_prompt_profile("generic")) == "# fine"


def test_rate_limiter_spreads_concurrent_waiters():
    """Waiting threads used to all wake at the same moment and burst past the limit."""
    limiter = llm.RateLimiter(rpm=600)  # 10/s, bucket of 10
    for _ in range(10):
        limiter.acquire()  # drain the bucket
    stamps = []
    lock = threading.Lock()

    def worker():
        limiter.acquire()
        with lock:
            stamps.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stamps.sort()
    assert stamps[-1] - stamps[0] >= 0.3  # 5 requests at 10/s need ~0.4s, not a burst


def test_api_and_cli_share_config_defaults(monkeypatch, tmp_path: Path):
    """api.convert used dpi=200/temperature=0.2 while the CLI used 72/0.1."""
    captured = []

    class Fake:
        def __init__(self, cfg, profile=None):
            captured.append(cfg)

        def run(self):
            return captured[-1].output_path

    monkeypatch.setattr("llm_markdownify.api.Markdownifier", Fake)
    monkeypatch.setattr("llm_markdownify.cli.Markdownifier", Fake)
    pdf = tmp_path / "in.pdf"
    _make_pdf(pdf, 1)
    convert(pdf, tmp_path / "a.md")
    CliRunner().invoke(app, ["-o", str(tmp_path / "b.md"), str(pdf)])
    api_cfg, cli_cfg = captured
    fields = set(MarkdownifyConfig.model_fields) - {"output_path", "log_level", "enable_cache"}
    assert {f: getattr(api_cfg, f) for f in fields} == {f: getattr(cli_cfg, f) for f in fields}


def test_cli_errors_do_not_leak_secrets(monkeypatch, tmp_path: Path):
    """Typer's default pretty tracebacks printed local variables, including the API key."""
    secret = "sk-test-SECRET-should-never-print"

    def boom(**kwargs):
        api_key = secret  # noqa: F841 - a local that a locals dump would show
        raise litellm.BadRequestError("nope", model="m", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", boom)
    img = tmp_path / "in.png"
    Image.new("RGB", (20, 20), "white").save(img)
    result = CliRunner().invoke(app, ["-o", str(tmp_path / "o.md"), "--no-grouping", str(img)])
    assert result.exit_code == 1
    assert secret not in result.output
    assert "BadRequestError" in result.output


def test_cli_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __import__("llm_markdownify").__version__


def test_failed_group_cancels_pending_work(monkeypatch, tmp_path: Path):
    """One failing group used to wait for (and pay for) every other group before raising."""
    from llm_markdownify.markdownifier import Markdownifier

    started = []

    def fake_generate(**kwargs):
        started.append(1)
        time.sleep(0.2)  # a real LLM call takes a while; cancellation happens meanwhile
        raise litellm.BadRequestError("bad", model="m", llm_provider="openai")

    monkeypatch.setattr("llm_markdownify.markdownifier.generate_markdown", fake_generate)
    pdf = tmp_path / "in.pdf"
    _make_pdf(pdf, 6)
    cfg = MarkdownifyConfig(
        input_path=pdf, output_path=tmp_path / "o.md", enable_grouping=False, concurrency=1
    )
    with pytest.raises(litellm.BadRequestError):
        Markdownifier(cfg).run()
    assert len(started) <= 2  # the failed call plus at most the one already in flight
    assert not (tmp_path / "o.md").exists()


def test_default_profile_is_generic():
    from llm_markdownify.prompt_profiles import DEFAULT_PROFILE

    assert DEFAULT_PROFILE == "generic"
    assert "LaTeX" in load_prompt_profile(DEFAULT_PROFILE).markdown_system


def test_rate_limits_retry_on_a_time_budget_not_attempts(monkeypatch):
    """A shared key's TPM quota can stay saturated past max_retries; keep trying within the budget."""
    calls = []

    def fake_completion(**kwargs):
        calls.append(1)
        if len(calls) < 6:  # more failures than max_retries=2
            raise litellm.RateLimitError("slow down", llm_provider="openai", model="m")
        return _response("# eventually")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(llm, "RATE_LIMIT_BUDGET_S", 30.0)
    llm.configure_llm(max_retries=2, retry_delay=0.01)
    assert llm.generate_markdown("m", ["data:a"], load_prompt_profile("generic")) == "# eventually"


def test_rate_limit_budget_is_enforced(monkeypatch):
    def fake_completion(**kwargs):
        raise litellm.RateLimitError("slow down", llm_provider="openai", model="m")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(llm, "RATE_LIMIT_BUDGET_S", 0.2)
    llm.configure_llm(max_retries=2, retry_delay=0.01)
    with pytest.raises(litellm.RateLimitError):
        llm.generate_markdown("m", ["data:a"], load_prompt_profile("generic"))


def test_fence_stripping_keeps_documents_that_start_and_end_with_code():
    doc = "```\ncode\n```\n\ntext\n\n```\nmore\n```"
    assert llm.strip_markdown_fence(doc) == doc
    wrapped = "```markdown\n# T\n\n```py\nx\n```\n```"
    assert llm.strip_markdown_fence(wrapped) == "# T\n\n```py\nx\n```"


def test_exhausted_output_budget_fails_fast(monkeypatch):
    """Reasoning models that spend max_tokens thinking fail identically on every retry."""
    calls = []

    def fake_completion(**kwargs):
        calls.append(1)
        return _response(None, "length")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    with pytest.raises(llm.OutputBudgetExhaustedError, match="max-tokens"):
        llm.generate_markdown("m", ["data:a"], load_prompt_profile("generic"))
    assert len(calls) == 1


def test_rate_limits_do_not_use_up_other_retries(monkeypatch):
    errors = [litellm.RateLimitError("rl", llm_provider="openai", model="m")] * 4 + [
        litellm.Timeout("t", model="m", llm_provider="openai")
    ]

    def fake_completion(**kwargs):
        if errors:
            raise errors.pop(0)
        return _response("# ok")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    llm.configure_llm(max_retries=1, retry_delay=0.01)
    assert llm.generate_markdown("m", ["data:a"], load_prompt_profile("generic")) == "# ok"


def test_cache_key_ignores_transport_only_kwargs():
    llm.configure_llm(llm_kwargs={"api_key": "a", "timeout": 5})
    key_a = llm._request_key("m", [])
    llm.configure_llm(llm_kwargs={"api_key": "b", "timeout": 9})
    assert llm._request_key("m", []) == key_a
    llm.configure_llm(llm_kwargs={"reasoning_effort": "high"})
    assert llm._request_key("m", []) != key_a


def test_blank_page_returns_empty_markdown_without_retrying(monkeypatch):
    """A page with only a running header and page number correctly yields nothing. It used to be
    retried 5 times and then fail the whole document (seen on olmOCR-Bench headers_footers)."""
    calls = []

    def fake_completion(**kwargs):
        calls.append(1)
        return _response("", "stop")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    assert llm.generate_markdown("m", ["data:a"], load_prompt_profile("generic")) == ""
    assert len(calls) == 1


def test_local_server_works_without_an_api_key(monkeypatch):
    """LM Studio / Ollama need no key, but the OpenAI client refused to start without one."""
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _response("# local")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm.configure_llm(llm_kwargs={"api_base": "http://localhost:1234/v1"})
    llm.generate_markdown("openai/qwen3.5-9b", ["data:a"], load_prompt_profile("generic"))
    assert seen["api_key"] == "not-needed"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    seen.clear()
    llm.generate_markdown("openai/qwen3.5-9b", ["data:b"], load_prompt_profile("generic"))
    assert "api_key" not in seen  # a real key is left for LiteLLM to pick up
