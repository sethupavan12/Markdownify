# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Convert large document sets through the OpenAI Batch API: about half the price, up to 24h.

Workflow: `submit_batch` renders every page and uploads the requests, `batch_status` reports
progress, and `collect_batch` downloads finished pages and writes one Markdown file per input.
All state lives in `<out_dir>/.markdownify-batch/`, so each step can run in a different process,
hours apart. Every page is its own request; cross-page table merging is not applied in batch mode.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .llm import _message_with_images, strip_markdown_fence
from .logging import get_logger
from .pager import IMAGE_SUFFIXES, ImageFormat, PageImage, iter_image_pages, iter_pdf_pages
from .prompt_profiles import DEFAULT_PROFILE, load_prompt_profile

logger = get_logger("llm_markdownify.batch")

STATE_DIR = ".markdownify-batch"
MANIFEST = "manifest.json"
TERMINAL = {"completed", "failed", "expired", "cancelled"}

# OpenAI limits per batch input file: 200 MB and 50,000 requests. Stay under both.
MAX_SHARD_BYTES = 190 * 1024 * 1024
MAX_SHARD_REQUESTS = 50_000

_BATCHABLE_SUFFIXES = {".pdf"} | IMAGE_SUFFIXES


@dataclass
class BatchStatus:
    batches: list[dict[str, Any]]
    total: int
    completed: int
    failed: int

    @property
    def done(self) -> bool:
        return all(b["status"] in TERMINAL for b in self.batches)


@dataclass
class CollectResult:
    written: list[Path] = field(default_factory=list)
    incomplete: dict[str, list[int]] = field(default_factory=dict)  # output path -> missing pages
    failed_pages: dict[str, str] = field(default_factory=dict)  # custom_id -> last error
    document_errors: dict[str, str] = field(default_factory=dict)  # input path -> render error
    batch_errors: dict[str, str] = field(default_factory=dict)  # batch id -> batch-level error
    resubmitted: int = 0
    pending_batches: int = 0

    @property
    def complete(self) -> bool:
        return not self.incomplete and not self.pending_batches


def _openai_model_name(model: str) -> str:
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    if "/" in model:
        raise ValueError(
            f"Batch mode supports OpenAI models (e.g. gpt-5.4-mini), got '{model}'. "
            "Use the regular `markdownify` command for other providers."
        )
    return model


def _client(api_base: Optional[str] = None):
    from openai import OpenAI

    return OpenAI(base_url=api_base) if api_base else OpenAI()


def _state(out_dir: Path) -> Path:
    return out_dir / STATE_DIR


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    path = _state(out_dir) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"No batch job found in {out_dir} (missing {path})")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    path = _state(out_dir) / MANIFEST
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _remove_stale_shards(out_dir: Path) -> None:
    """Delete request files left behind by a process that was killed mid-submit."""
    for leftover in _state(out_dir).glob("*.jsonl"):
        leftover.unlink(missing_ok=True)


def expand_inputs(inputs: Iterable[str | Path]) -> list[Path]:
    """Files as given; directories expanded recursively to supported documents, sorted."""
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files += sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in _BATCHABLE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in _BATCHABLE_SUFFIXES:
            files.append(path)
        else:
            raise ValueError(f"Not a supported file or directory: {path}")
    return files


def _output_paths(files: list[Path], out_dir: Path) -> list[Path]:
    """One .md per input, named after the input. Clashes (including case-only ones, which collide
    on macOS and Windows) get a numeric suffix that is itself checked for clashes."""
    used: set[str] = set()
    outputs = []
    for f in files:
        name, n = f.stem, 1
        while name.casefold() in used:
            n += 1
            name = f"{f.stem}-{n}"
        used.add(name.casefold())
        outputs.append(out_dir / f"{name}.md")
    return outputs


def _custom_id(doc: int, page: int) -> str:
    return f"d{doc}-p{page}"


def _iter_pages(path: Path, opts: dict[str, Any]) -> Iterator[PageImage]:
    if path.suffix.lower() == ".pdf":
        return iter_pdf_pages(path, opts["dpi"], opts["max_image_px"], opts["image_format"])
    return iter_image_pages(path, opts["max_image_px"], opts["image_format"])


def _request_lines(
    manifest: dict[str, Any], wanted: Optional[set[str]] = None
) -> Iterator[tuple[str, str]]:
    """Yield (custom_id, jsonl line) for every page, or only the custom_ids in `wanted`.

    Pages are rendered one at a time, so memory stays flat however long the documents are. A
    document that fails to render is recorded in the manifest and skipped; the rest continue.
    """
    opts = manifest["options"]
    profile = load_prompt_profile(manifest["profile"])
    body_extra: dict[str, Any] = {"max_completion_tokens": opts["max_tokens"]}
    if opts.get("temperature") is not None:
        body_extra["temperature"] = opts["temperature"]
    if opts.get("reasoning_effort"):
        body_extra["reasoning_effort"] = opts["reasoning_effort"]
    wanted_docs = {int(c.split("-")[0][1:]) for c in wanted} if wanted is not None else None

    for doc_index, doc in enumerate(manifest["documents"]):
        if wanted_docs is not None and doc_index not in wanted_docs:
            continue
        count = 0
        try:
            for page in _iter_pages(Path(doc["input"]), opts):
                count += 1
                custom_id = _custom_id(doc_index, page.index)
                if wanted is not None and custom_id not in wanted:
                    continue
                body = {
                    "model": manifest["model"],
                    "messages": [
                        {"role": "system", "content": profile.markdown_system},
                        _message_with_images(profile.markdown_user, [page.data_url]),
                    ],
                    **body_extra,
                }
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                yield custom_id, json.dumps(request)
        except Exception as e:  # noqa: BLE001 - one bad file must not sink the whole job
            doc["error"] = f"{type(e).__name__}: {e}"[:300]
            logger.error("Could not render %s: %s", doc["input"], doc["error"])
            continue
        doc["pages"] = count
        doc.pop("error", None)


def _start_batch(client, manifest: dict[str, Any], input_file_id: str):
    return client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"tool": "llm-markdownify", "job_id": manifest["job_id"]},
    )


def _submit_lines(
    client, out_dir: Path, manifest: dict[str, Any], lines: Iterator[tuple[str, str]]
) -> int:
    """Write requests into size-capped shards, upload each and start one batch per shard."""
    submitted = 0
    shard_ids: list[str] = []
    shard_bytes = 0
    shard = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", dir=_state(out_dir), delete=False, encoding="utf-8"
    )

    def flush() -> None:
        nonlocal shard, shard_ids, shard_bytes
        shard.close()
        if shard_ids:
            with open(shard.name, "rb") as fh:
                uploaded = client.files.create(file=fh, purpose="batch")
            # Record the upload before starting the batch: if we crash in between, collect finds
            # the entry without an id and starts (or adopts) the batch instead of losing pages.
            entry = {"id": None, "input_file_id": uploaded.id, "requests": len(shard_ids)}
            manifest["batches"].append(entry)
            _save_manifest(out_dir, manifest)
            entry["id"] = _start_batch(client, manifest, uploaded.id).id
            _save_manifest(out_dir, manifest)
            logger.info("Started batch %s with %d page requests", entry["id"], len(shard_ids))
        os.unlink(shard.name)
        shard = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", dir=_state(out_dir), delete=False, encoding="utf-8"
        )
        shard_ids, shard_bytes = [], 0

    try:
        for custom_id, line in lines:
            size = len(line.encode("utf-8")) + 1
            if shard_ids and (
                shard_bytes + size > MAX_SHARD_BYTES or len(shard_ids) >= MAX_SHARD_REQUESTS
            ):
                flush()
            shard.write(line + "\n")
            shard_ids.append(custom_id)
            shard_bytes += size
            submitted += 1
        flush()
    finally:
        shard.close()
        if os.path.exists(shard.name):
            os.unlink(shard.name)
    return submitted


def _recover_unstarted(client, out_dir: Path, manifest: dict[str, Any]) -> None:
    """Give every uploaded shard a batch id: adopt the batch if it was created before a crash,
    otherwise start it now."""
    unstarted = [b for b in manifest["batches"] if not b.get("id")]
    if not unstarted:
        return
    existing = {
        b.input_file_id: b.id
        for b in client.batches.list(limit=100)
        if (getattr(b, "metadata", None) or {}).get("job_id") == manifest["job_id"]
    }
    for entry in unstarted:
        entry["id"] = existing.get(entry["input_file_id"]) or (
            _start_batch(client, manifest, entry["input_file_id"]).id
        )
        logger.info("Recovered batch %s for upload %s", entry["id"], entry["input_file_id"])
    _save_manifest(out_dir, manifest)


def submit_batch(
    inputs: Iterable[str | Path],
    out_dir: str | Path,
    *,
    model: str = "gpt-5.4-mini",
    profile: Optional[str] = None,
    dpi: int = 200,
    max_image_px: int = 2048,
    image_format: ImageFormat = "jpeg",
    max_tokens: int = 16000,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    api_base: Optional[str] = None,
    client=None,
) -> Path:
    """Render every page of `inputs` and submit them as OpenAI batch jobs. Returns out_dir."""
    out = Path(out_dir)
    model_name = _openai_model_name(model)  # validate before creating any state
    files = expand_inputs(inputs)
    if not files:
        raise ValueError("No input documents found")
    state = _state(out)
    if (state / MANIFEST).exists():
        raise FileExistsError(f"{out} already holds a batch job; use another --out directory")
    state.mkdir(parents=True, exist_ok=True)
    _remove_stale_shards(out)

    manifest: dict[str, Any] = {
        "version": 1,
        "job_id": uuid.uuid4().hex,
        "model": model_name,
        "profile": profile or DEFAULT_PROFILE,
        "api_base": api_base,
        "created_at": int(time.time()),
        "options": {
            "dpi": dpi,
            "max_image_px": max_image_px,
            "image_format": image_format,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
        },
        "documents": [
            {"input": str(f.resolve()), "output": str(o.resolve()), "pages": None}
            for f, o in zip(files, _output_paths(files, out))
        ],
        "batches": [],
        "page_errors": {},
    }
    _save_manifest(out, manifest)
    client = client or _client(api_base)
    count = _submit_lines(client, out, manifest, _request_lines(manifest))
    _save_manifest(out, manifest)  # page counts and render errors were filled in while rendering
    failed_docs = sum(1 for d in manifest["documents"] if d.get("error"))
    logger.info(
        "Submitted %d pages from %d documents in %d batch(es)%s",
        count,
        len(files) - failed_docs,
        len(manifest["batches"]),
        f"; {failed_docs} document(s) could not be rendered" if failed_docs else "",
    )
    return out


def batch_status(out_dir: str | Path, client=None) -> BatchStatus:
    out = Path(out_dir)
    manifest = _load_manifest(out)
    client = client or _client(manifest.get("api_base"))
    _recover_unstarted(client, out, manifest)
    rows = []
    total = completed = failed = 0
    for entry in manifest["batches"]:
        batch = client.batches.retrieve(entry["id"])
        counts = batch.request_counts
        rows.append(
            {
                "id": batch.id,
                "status": batch.status,
                "total": counts.total if counts else entry["requests"],
                "completed": counts.completed if counts else 0,
                "failed": counts.failed if counts else 0,
            }
        )
        total += rows[-1]["total"]
        completed += rows[-1]["completed"]
        failed += rows[-1]["failed"]
    return BatchStatus(batches=rows, total=total, completed=completed, failed=failed)


def _page_path(out: Path, custom_id: str) -> Path:
    return _state(out) / "pages" / f"{custom_id}.md"


def _read_jsonl(client, file_id: Optional[str]) -> list[dict[str, Any]]:
    if not file_id:
        return []
    text = client.files.content(file_id).text
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _batch_error_text(batch) -> str:
    errors = getattr(batch, "errors", None)
    data = getattr(errors, "data", None) or []
    messages = [getattr(e, "message", None) or str(e) for e in data]
    return "; ".join(m for m in messages if m) or f"batch {batch.status}"


def collect_batch(out_dir: str | Path, *, retry_failed: bool = False, client=None) -> CollectResult:
    """Download finished pages and write every document whose pages are all done.

    Safe to call repeatedly. With `retry_failed`, pages that failed in finished batches are
    submitted again as a new batch.
    """
    out = Path(out_dir)
    manifest = _load_manifest(out)
    manifest.setdefault("page_errors", {})
    client = client or _client(manifest.get("api_base"))
    _remove_stale_shards(out)
    _recover_unstarted(client, out, manifest)
    result = CollectResult()
    (_state(out) / "pages").mkdir(parents=True, exist_ok=True)

    for entry in manifest["batches"]:
        if entry.get("collected"):
            if entry.get("error"):
                result.batch_errors[entry["id"]] = entry["error"]
            continue
        batch = client.batches.retrieve(entry["id"])
        if batch.status not in TERMINAL:
            result.pending_batches += 1
            continue
        for row in _read_jsonl(client, batch.output_file_id):
            custom_id = row["custom_id"]
            response = row.get("response") or {}
            body = response.get("body") or {}
            choices = body.get("choices") or []
            choice = choices[0] if choices else {}
            content = (choice.get("message") or {}).get("content")
            finish_reason = choice.get("finish_reason")
            has_text = bool(content and content.strip())
            # Empty text that finished normally is a blank page (or one holding only the headers
            # and footers we asked the model to drop): a correct answer, not a failure.
            if response.get("status_code") == 200 and (has_text or finish_reason == "stop"):
                if finish_reason == "length":
                    logger.warning("Page %s was truncated at max_tokens", custom_id)
                text = strip_markdown_fence(content).strip() if has_text else ""
                _page_path(out, custom_id).write_text(text, encoding="utf-8")
                manifest["page_errors"].pop(custom_id, None)
            else:
                error = row.get("error") or body.get("error") or f"empty ({finish_reason})"
                manifest["page_errors"][custom_id] = json.dumps(error)[:300]
        for row in _read_jsonl(client, batch.error_file_id):
            response_body = (row.get("response") or {}).get("body") or {}
            error = row.get("error") or response_body.get("error") or "unknown error"
            manifest["page_errors"][row["custom_id"]] = json.dumps(error)[:300]
        if batch.status == "failed":  # rejected as a whole (e.g. validation); no per-page rows
            entry["error"] = _batch_error_text(batch)
            result.batch_errors[entry["id"]] = entry["error"]
        entry["collected"] = True
        entry["status"] = batch.status
        _save_manifest(out, manifest)

    missing_ids: set[str] = set()
    for doc_index, doc in enumerate(manifest["documents"]):
        output = Path(doc["output"])
        if doc.get("error") or doc.get("pages") is None:
            result.document_errors[doc["input"]] = doc.get("error") or "never rendered"
            result.incomplete[str(output)] = []
            continue
        texts, missing = [], []
        for page in range(doc["pages"]):
            custom_id = _custom_id(doc_index, page)
            path = _page_path(out, custom_id)
            if path.exists():
                texts.append(path.read_text(encoding="utf-8"))
            else:
                missing.append(page + 1)
                missing_ids.add(custom_id)
                if custom_id in manifest["page_errors"]:
                    result.failed_pages[custom_id] = manifest["page_errors"][custom_id]
        if missing:
            result.incomplete[str(output)] = missing
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n\n".join(t for t in texts if t).strip() + "\n", encoding="utf-8")
        result.written.append(output)

    # Only retry pages that are not still queued in an unfinished batch.
    if retry_failed and missing_ids and result.pending_batches == 0:
        result.resubmitted = _submit_lines(
            client, out, manifest, _request_lines(manifest, wanted=missing_ids)
        )
        _save_manifest(out, manifest)
    return result


def wait_batch(
    out_dir: str | Path, *, poll_seconds: float = 60.0, retry_failed: bool = False, client=None
) -> CollectResult:
    """Poll until every batch finishes, then collect.

    With `retry_failed`, failed pages are resubmitted once, unless a batch was rejected as a
    whole: that usually fails the same way again, so its error is reported instead.
    """
    out = Path(out_dir)
    manifest = _load_manifest(out)
    client = client or _client(manifest.get("api_base"))
    retried = False
    while True:
        status = batch_status(out, client=client)
        logger.info(
            "Batch progress: %d/%d pages done, %d failed",
            status.completed,
            status.total,
            status.failed,
        )
        if status.done:
            batch_rejected = any(b["status"] == "failed" for b in status.batches)
            may_retry = retry_failed and not retried and not batch_rejected
            result = collect_batch(out, retry_failed=may_retry, client=client)
            if result.resubmitted:
                retried = True
                continue
            return result
        time.sleep(poll_seconds)
