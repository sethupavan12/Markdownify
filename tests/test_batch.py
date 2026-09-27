# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from llm_markdownify import batch


class FakeOpenAI:
    """In-memory stand-in for the OpenAI files + batches APIs."""

    def __init__(self, fail_ids: set[str] | None = None, blank_ids: set[str] | None = None):
        self.uploads: dict[str, list[dict]] = {}
        self.batch_store: dict[str, SimpleNamespace] = {}
        self.fail_ids = set(fail_ids or ())
        self.blank_ids = set(blank_ids or ())
        self.reject_next = False  # next finish_all rejects the batch as a whole
        self.crash_on_create = False
        self.contents: dict[str, str] = {}
        self.files = SimpleNamespace(create=self._create_file, content=self._content)
        self.batches = SimpleNamespace(
            create=self._create_batch, retrieve=self._retrieve, list=self._list
        )

    def _create_file(self, file, purpose):
        assert purpose == "batch"
        file_id = f"file-{len(self.uploads)}"
        self.uploads[file_id] = [json.loads(line) for line in file.read().decode().splitlines()]
        return SimpleNamespace(id=file_id)

    def _create_batch(self, input_file_id, endpoint, completion_window, metadata):
        assert endpoint == "/v1/chat/completions" and completion_window == "24h"
        batch_id = f"batch-{len(self.batch_store)}"
        self.batch_store[batch_id] = SimpleNamespace(
            id=batch_id,
            status="in_progress",
            input_file_id=input_file_id,
            metadata=metadata,
            errors=None,
            output_file_id=None,
            error_file_id=None,
            request_counts=SimpleNamespace(
                total=len(self.uploads[input_file_id]), completed=0, failed=0
            ),
        )
        if self.crash_on_create:  # the batch exists server-side, but the caller never hears back
            self.crash_on_create = False
            raise KeyboardInterrupt
        return self.batch_store[batch_id]

    def _list(self, limit=20):
        return list(self.batch_store.values())

    def finish_all(self):
        for b in self.batch_store.values():
            if b.status != "in_progress":
                continue
            ok, err = [], []
            if self.reject_next:
                self.reject_next = False
                b.status = "failed"
                b.errors = SimpleNamespace(data=[SimpleNamespace(message="invalid model")])
                continue
            for req in self.uploads[b.input_file_id]:
                cid = req["custom_id"]
                if cid in self.fail_ids:
                    err.append({"custom_id": cid, "error": {"message": "boom"}})
                    self.fail_ids.discard(cid)  # succeeds when retried
                else:
                    text = "" if cid in self.blank_ids else f"```markdown\n# {cid}\n```"
                    ok.append(
                        {
                            "custom_id": cid,
                            "response": {
                                "status_code": 200,
                                "body": {
                                    "choices": [
                                        {
                                            "message": {"content": text},
                                            "finish_reason": "stop",
                                        }
                                    ]
                                },
                            },
                        }
                    )
            b.output_file_id, b.error_file_id = f"out-{b.id}", f"err-{b.id}"
            self.contents[b.output_file_id] = "\n".join(json.dumps(r) for r in ok)
            self.contents[b.error_file_id] = "\n".join(json.dumps(r) for r in err)
            b.status = "completed"
            b.request_counts.completed, b.request_counts.failed = len(ok), len(err)

    def _retrieve(self, batch_id):
        return self.batch_store[batch_id]

    def _content(self, file_id):
        return SimpleNamespace(text=self.contents.get(file_id, ""))


def _pdf(path: Path, pages: int) -> Path:
    imgs = [Image.new("RGB", (300, 400), "white") for _ in range(pages)]
    imgs[0].save(path, format="PDF", save_all=True, append_images=imgs[1:])
    return path


def test_submit_collect_roundtrip(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    _pdf(docs / "a.pdf", 2)
    _pdf(docs / "b.pdf", 1)
    client = FakeOpenAI()
    out = tmp_path / "out"

    batch.submit_batch([docs], out, model="openai/gpt-5.4-mini", client=client)
    [upload] = client.uploads.values()
    assert [r["custom_id"] for r in upload] == ["d0-p0", "d0-p1", "d1-p0"]
    assert upload[0]["body"]["model"] == "gpt-5.4-mini"
    assert upload[0]["body"]["max_completion_tokens"] == 16000
    assert "temperature" not in upload[0]["body"]

    pending = batch.collect_batch(out, client=client)
    assert pending.pending_batches == 1 and not pending.written

    client.finish_all()
    result = batch.collect_batch(out, client=client)
    assert sorted(p.name for p in result.written) == ["a.md", "b.md"]
    assert (out / "a.md").read_text() == "# d0-p0\n\n# d0-p1\n"  # fences stripped, pages in order


def test_failed_pages_are_reported_and_retried(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 3)
    client = FakeOpenAI(fail_ids={"d0-p1"})
    out = tmp_path / "out"
    batch.submit_batch([tmp_path / "a.pdf"], out, client=client)
    client.finish_all()

    first = batch.collect_batch(out, client=client, retry_failed=True)
    assert "d0-p1" in first.failed_pages
    assert first.incomplete == {str((out / "a.md").resolve()): [2]}
    assert first.resubmitted == 1
    assert [r["custom_id"] for r in list(client.uploads.values())[-1]] == ["d0-p1"]

    client.finish_all()
    second = batch.collect_batch(out, client=client)
    assert [p.name for p in second.written] == ["a.md"]
    assert (out / "a.md").read_text().count("# d0-p") == 3


def test_requests_are_sharded_under_the_size_cap(tmp_path: Path, monkeypatch):
    _pdf(tmp_path / "a.pdf", 5)
    monkeypatch.setattr(batch, "MAX_SHARD_REQUESTS", 2)
    client = FakeOpenAI()
    batch.submit_batch([tmp_path / "a.pdf"], tmp_path / "out", client=client)
    assert [len(u) for u in client.uploads.values()] == [2, 2, 1]
    manifest = json.loads((tmp_path / "out" / batch.STATE_DIR / batch.MANIFEST).read_text())
    assert [b["requests"] for b in manifest["batches"]] == [2, 2, 1]


def test_duplicate_stems_get_distinct_outputs(tmp_path: Path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    files = [_pdf(tmp_path / "x" / "report.pdf", 1), _pdf(tmp_path / "y" / "report.pdf", 1)]
    outs = batch._output_paths(files, tmp_path / "out")
    assert [o.name for o in outs] == ["report.md", "report-2.md"]


def test_non_openai_models_are_rejected(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 1)
    with pytest.raises(ValueError, match="OpenAI .* and Anthropic"):
        batch.submit_batch(
            [tmp_path / "a.pdf"],
            tmp_path / "out",
            model="gemini/gemini-2.5-flash",
            client=FakeOpenAI(),
        )


def test_existing_job_is_not_overwritten(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 1)
    client = FakeOpenAI()
    batch.submit_batch([tmp_path / "a.pdf"], tmp_path / "out", client=client)
    with pytest.raises(FileExistsError):
        batch.submit_batch([tmp_path / "a.pdf"], tmp_path / "out", client=client)


def test_blank_pages_complete_the_document(tmp_path: Path):
    """A blank page returns empty text with finish_reason=stop; it must not block the document."""
    _pdf(tmp_path / "a.pdf", 3)
    client = FakeOpenAI(blank_ids={"d0-p1"})
    out = tmp_path / "out"
    batch.submit_batch([tmp_path / "a.pdf"], out, client=client)
    client.finish_all()
    result = batch.collect_batch(out, client=client)
    assert result.complete
    assert (out / "a.md").read_text() == "# d0-p0\n\n# d0-p2\n"


def test_unreadable_document_is_reported_not_written_empty(tmp_path: Path):
    """A corrupt file mid-job used to leave later documents with no page count, which collect
    then wrote out as empty Markdown files."""
    _pdf(tmp_path / "a.pdf", 1)
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    _pdf(tmp_path / "c.pdf", 2)
    client = FakeOpenAI()
    out = tmp_path / "out"
    batch.submit_batch(
        [tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"], out, client=client
    )
    client.finish_all()
    result = batch.collect_batch(out, client=client)
    assert sorted(p.name for p in result.written) == ["a.md", "c.md"]
    assert not (out / "b.md").exists()
    assert str((tmp_path / "b.pdf").resolve()) in result.document_errors


def test_crash_after_batch_creation_is_recovered_without_double_billing(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 2)
    client = FakeOpenAI()
    client.crash_on_create = True
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        batch.submit_batch([tmp_path / "a.pdf"], out, client=client)
    assert len(client.batch_store) == 1  # created server-side, id never saved locally
    client.finish_all()
    result = batch.collect_batch(out, client=client)  # adopts the orphaned batch
    assert len(client.batch_store) == 1  # nothing was paid for twice
    assert [p.name for p in result.written] == ["a.md"]


def test_rejected_batch_reports_reason_and_wait_does_not_retry(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 1)
    client = FakeOpenAI()
    client.reject_next = True
    out = tmp_path / "out"
    batch.submit_batch([tmp_path / "a.pdf"], out, client=client)
    client.finish_all()
    result = batch.wait_batch(out, poll_seconds=0, retry_failed=True, client=client)
    assert list(result.batch_errors.values()) == ["invalid model"]
    assert result.resubmitted == 0 and len(client.batch_store) == 1


def test_error_rows_with_null_body_do_not_crash(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 1)
    client = FakeOpenAI()
    out = tmp_path / "out"
    batch.submit_batch([tmp_path / "a.pdf"], out, client=client)
    b = client.batch_store["batch-0"]
    b.status, b.output_file_id, b.error_file_id = "completed", None, "err"
    client.contents["err"] = json.dumps({"custom_id": "d0-p0", "response": {"body": None}})
    result = batch.collect_batch(out, client=client)
    assert "d0-p0" in result.failed_pages
    # the reason survives a second collect
    assert "d0-p0" in batch.collect_batch(out, client=client).failed_pages


def test_output_names_never_collide(tmp_path: Path):
    files = [Path("x/report.pdf"), Path("y/report.pdf"), Path("report-2.pdf"), Path("Report.pdf")]
    names = [o.name for o in batch._output_paths(files, tmp_path)]
    assert len({n.casefold() for n in names}) == 4


def test_cli_collect_exits_nonzero_when_documents_are_incomplete(tmp_path: Path, monkeypatch):
    from typer.testing import CliRunner

    from llm_markdownify.batch_cli import app

    _pdf(tmp_path / "a.pdf", 2)
    client = FakeOpenAI(fail_ids={"d0-p1"})
    out = tmp_path / "out"
    batch.submit_batch([tmp_path / "a.pdf"], out, client=client)
    client.finish_all()
    monkeypatch.setattr(
        batch, "_backend", lambda manifest, c=None: batch._OpenAIBackend(client=client)
    )
    result = CliRunner().invoke(app, ["collect", str(out)])
    assert result.exit_code == 1
    assert "pages 2" in result.output


# ---------------------------------------------------------------------------------------------
# Anthropic Message Batches
# ---------------------------------------------------------------------------------------------


class FakeAnthropic:
    """In-memory stand-in for client.messages.batches (create/retrieve/list/results)."""

    def __init__(self, outcomes: dict[str, tuple] | None = None):
        # custom_id -> ("text", str) | ("blank",) | ("refusal",) | ("errored", msg)
        self.outcomes = outcomes or {}
        self.store: dict[str, SimpleNamespace] = {}
        self.requests: dict[str, list[dict]] = {}
        self.crash_on_create = False
        batches = SimpleNamespace(
            create=self._create, retrieve=self._retrieve, list=self._list, results=self._results
        )
        self.messages = SimpleNamespace(batches=batches)

    @staticmethod
    def _counts(processing=0, succeeded=0, errored=0):
        return SimpleNamespace(
            processing=processing, succeeded=succeeded, errored=errored, canceled=0, expired=0
        )

    def _create(self, requests):
        from datetime import datetime, timezone

        batch_id = f"msgbatch_{len(self.store)}"
        self.requests[batch_id] = requests
        self.store[batch_id] = SimpleNamespace(
            id=batch_id,
            processing_status="in_progress",
            created_at=datetime.now(timezone.utc),
            request_counts=self._counts(processing=len(requests)),
        )
        if self.crash_on_create:
            self.crash_on_create = False
            raise KeyboardInterrupt
        return self.store[batch_id]

    def _retrieve(self, batch_id):
        return self.store[batch_id]

    def _list(self, limit=20):
        return list(self.store.values())

    def finish_all(self):
        for b in self.store.values():
            b.processing_status = "ended"
            n = len(self.requests[b.id])
            b.request_counts = self._counts(succeeded=n)

    def _results(self, batch_id):
        for req in self.requests[batch_id]:
            cid = req["custom_id"]
            kind = self.outcomes.get(cid, ("text", f"```markdown\n# {cid}\n```"))
            if kind[0] == "errored":
                error = SimpleNamespace(error=SimpleNamespace(message=kind[1]))
                yield SimpleNamespace(
                    custom_id=cid, result=SimpleNamespace(type="errored", error=error)
                )
                continue
            text = kind[1] if kind[0] == "text" else ""
            stop = {"text": "end_turn", "blank": "end_turn", "refusal": "refusal"}[kind[0]]
            content = [SimpleNamespace(type="text", text=text)] if text else []
            msg = SimpleNamespace(content=content, stop_reason=stop)
            yield SimpleNamespace(
                custom_id=cid, result=SimpleNamespace(type="succeeded", message=msg)
            )


def test_anthropic_request_shape(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 1)
    client = FakeAnthropic()
    batch.submit_batch(
        [tmp_path / "a.pdf"],
        tmp_path / "out",
        model="anthropic/claude-opus-5",
        reasoning_effort="low",
        client=client,
    )
    [req] = client.requests["msgbatch_0"]
    params = req["params"]
    assert req["custom_id"] == "d0-p0"
    assert params["model"] == "claude-opus-5" and params["max_tokens"] == 16000
    assert "LaTeX" in params["system"]
    image, text = params["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/jpeg"
    assert image["source"]["data"] and not image["source"]["data"].startswith("data:")
    assert text["type"] == "text"
    assert params["output_config"] == {"effort": "low"}
    assert "temperature" not in params  # current Claude models reject it


def test_anthropic_roundtrip_with_blank_refusal_and_error(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 3)
    _pdf(tmp_path / "b.pdf", 1)
    client = FakeAnthropic(
        outcomes={"d0-p1": ("blank",), "d1-p0": ("refusal",), "d0-p2": ("text", "tail")}
    )
    out = tmp_path / "out"
    batch.submit_batch(
        [tmp_path / "a.pdf", tmp_path / "b.pdf"], out, model="claude-opus-5", client=client
    )
    assert batch.batch_status(out, client=client).done is False
    client.finish_all()
    assert batch.batch_status(out, client=client).done is True
    result = batch.collect_batch(out, client=client)
    assert [p.name for p in result.written] == ["a.md"]
    assert (out / "a.md").read_text() == "# d0-p0\n\ntail\n"  # blank page contributes nothing
    assert result.failed_pages == {"d1-p0": "model refused this page"}


def test_anthropic_errored_rows_are_retried(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 2)
    client = FakeAnthropic(outcomes={"d0-p1": ("errored", "overloaded")})
    out = tmp_path / "out"
    batch.submit_batch([tmp_path / "a.pdf"], out, model="claude-opus-5", client=client)
    client.finish_all()
    first = batch.collect_batch(out, client=client, retry_failed=True)
    assert first.failed_pages == {"d0-p1": "overloaded"} and first.resubmitted == 1
    assert [r["custom_id"] for r in client.requests["msgbatch_1"]] == ["d0-p1"]
    client.outcomes.clear()
    client.finish_all()
    assert [p.name for p in batch.collect_batch(out, client=client).written] == ["a.md"]


def test_anthropic_crash_after_create_is_adopted(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 2)
    client = FakeAnthropic()
    client.crash_on_create = True
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        batch.submit_batch([tmp_path / "a.pdf"], out, model="claude-opus-5", client=client)
    client.finish_all()
    result = batch.collect_batch(out, client=client)
    assert len(client.store) == 1  # adopted by creation time + request count, not resubmitted
    assert [p.name for p in result.written] == ["a.md"]
