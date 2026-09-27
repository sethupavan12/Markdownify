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

    def __init__(self, fail_ids: set[str] | None = None):
        self.uploads: dict[str, list[dict]] = {}
        self.batch_store: dict[str, SimpleNamespace] = {}
        self.fail_ids = set(fail_ids or ())
        self.contents: dict[str, str] = {}
        self.files = SimpleNamespace(create=self._create_file, content=self._content)
        self.batches = SimpleNamespace(create=self._create_batch, retrieve=self._retrieve)

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
            input=input_file_id,
            output_file_id=None,
            error_file_id=None,
            request_counts=SimpleNamespace(
                total=len(self.uploads[input_file_id]), completed=0, failed=0
            ),
        )
        return self.batch_store[batch_id]

    def finish_all(self):
        for b in self.batch_store.values():
            if b.status != "in_progress":
                continue
            ok, err = [], []
            for req in self.uploads[b.input]:
                cid = req["custom_id"]
                if cid in self.fail_ids:
                    err.append({"custom_id": cid, "error": {"message": "boom"}})
                    self.fail_ids.discard(cid)  # succeeds when retried
                else:
                    ok.append(
                        {
                            "custom_id": cid,
                            "response": {
                                "status_code": 200,
                                "body": {
                                    "choices": [
                                        {
                                            "message": {"content": f"```markdown\n# {cid}\n```"},
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
    with pytest.raises(ValueError, match="OpenAI"):
        batch.submit_batch(
            [tmp_path / "a.pdf"],
            tmp_path / "out",
            model="anthropic/claude-sonnet-5",
            client=FakeOpenAI(),
        )


def test_existing_job_is_not_overwritten(tmp_path: Path):
    _pdf(tmp_path / "a.pdf", 1)
    client = FakeOpenAI()
    batch.submit_batch([tmp_path / "a.pdf"], tmp_path / "out", client=client)
    with pytest.raises(FileExistsError):
        batch.submit_batch([tmp_path / "a.pdf"], tmp_path / "out", client=client)
