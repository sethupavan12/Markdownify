# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Convert large document sets through a provider batch API: about half the price, up to 24h.

Supported: the OpenAI Batch API (`gpt-5.4-mini`, `openai/...`) and Anthropic Message Batches
(`anthropic/claude-...` or `claude-...`).

Workflow: `submit_batch` renders every page and submits the requests, `batch_status` reports
progress, and `collect_batch` downloads finished pages and writes one Markdown file per input.
All state lives in `<out_dir>/.markdownify-batch/`, so each step can run in a different process,
hours apart. Every page is its own request; cross-page table merging is not applied in batch mode.
"""

from __future__ import annotations

import base64
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
from .prompt_profiles import DEFAULT_PROFILE, PromptProfile, load_prompt_profile

logger = get_logger("llm_markdownify.batch")

STATE_DIR = ".markdownify-batch"
MANIFEST = "manifest.json"

# Per-batch limits: OpenAI 200 MB / 50,000 requests per input file; Anthropic 256 MB / 100,000
# requests per batch. One shard size under both keeps the code simple.
MAX_SHARD_BYTES = 190 * 1024 * 1024
MAX_SHARD_REQUESTS = 50_000

_BATCHABLE_SUFFIXES = {".pdf"} | IMAGE_SUFFIXES
_ANTHROPIC_EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass
class PageResult:
    custom_id: str
    text: Optional[str]  # None when the page failed
    error: Optional[str] = None
    truncated: bool = False


@dataclass
class BatchInfo:
    id: str
    status: str  # the provider's own status string, for display
    done: bool
    total: int
    completed: int
    failed: int
    error: Optional[str] = None  # set when the whole batch was rejected


@dataclass
class BatchStatus:
    batches: list[dict[str, Any]]
    total: int
    completed: int
    failed: int

    @property
    def done(self) -> bool:
        return all(b["done"] for b in self.batches)


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


# --------------------------------------------------------------------------------------------
# Provider backends. Each turns a page into one request line, starts batches from a shard file,
# reports status, and yields per-page results in a common shape.
# --------------------------------------------------------------------------------------------


class _OpenAIBackend:
    name = "openai"

    def __init__(self, api_base: Optional[str] = None, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=api_base) if api_base else OpenAI()
        self.client = client

    def request_line(
        self, custom_id: str, model: str, profile: PromptProfile, page: PageImage, opts: dict
    ) -> str:
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": profile.markdown_system},
                _message_with_images(profile.markdown_user, [page.data_url]),
            ],
            "max_completion_tokens": opts["max_tokens"],
        }
        if opts.get("temperature") is not None:
            body["temperature"] = opts["temperature"]
        if opts.get("reasoning_effort"):
            body["reasoning_effort"] = opts["reasoning_effort"]
        return json.dumps(
            {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}
        )

    def upload(self, shard: Path) -> Optional[str]:
        with open(shard, "rb") as fh:
            return self.client.files.create(file=fh, purpose="batch").id

    def create(self, shard: Path, input_ref: Optional[str], job_id: str) -> str:
        # One attempt only: an SDK retry after a client-side timeout could start a second batch
        # that we never record but still pay for. A failed create is left for recovery.
        return (
            self.client.with_options(max_retries=0)
            .batches.create(
                input_file_id=input_ref,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"tool": "llm-markdownify", "job_id": job_id},
            )
            .id
        )

    def find_orphan(self, entry: dict, job_id: str, known: set[str]) -> Optional[str]:
        for b in self.client.batches.list(limit=100):
            meta = getattr(b, "metadata", None) or {}
            if meta.get("job_id") == job_id and b.input_file_id == entry["input_file_id"]:
                return b.id
        return None

    def info(self, batch_id: str) -> BatchInfo:
        b = self.client.batches.retrieve(batch_id)
        counts = b.request_counts
        error = None
        if b.status == "failed":  # rejected as a whole (e.g. validation); no per-page rows
            data = getattr(getattr(b, "errors", None), "data", None) or []
            error = "; ".join(getattr(e, "message", None) or str(e) for e in data) or "failed"
        return BatchInfo(
            id=b.id,
            status=b.status,
            done=b.status in {"completed", "failed", "expired", "cancelled"},
            total=counts.total if counts else 0,
            completed=counts.completed if counts else 0,
            failed=counts.failed if counts else 0,
            error=error,
        )

    def _rows(self, file_id: Optional[str]) -> list[dict[str, Any]]:
        if not file_id:
            return []
        text = self.client.files.content(file_id).text
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def results(self, batch_id: str) -> Iterator[PageResult]:
        b = self.client.batches.retrieve(batch_id)
        for row in self._rows(b.output_file_id):
            response = row.get("response") or {}
            body = response.get("body") or {}
            choices = body.get("choices") or []
            choice = choices[0] if choices else {}
            content = (choice.get("message") or {}).get("content")
            finish = choice.get("finish_reason")
            has_text = bool(content and content.strip())
            # Empty text that finished normally is a blank page (or one holding only the headers
            # and footers we asked the model to drop): a correct answer, not a failure.
            if response.get("status_code") == 200 and (has_text or finish == "stop"):
                yield PageResult(row["custom_id"], content or "", truncated=finish == "length")
            else:
                error = row.get("error") or body.get("error") or f"empty ({finish})"
                yield PageResult(row["custom_id"], None, error=json.dumps(error)[:300])
        for row in self._rows(b.error_file_id):
            response_body = (row.get("response") or {}).get("body") or {}
            error = row.get("error") or response_body.get("error") or "unknown error"
            yield PageResult(row["custom_id"], None, error=json.dumps(error)[:300])


class _AnthropicBackend:
    name = "anthropic"

    def __init__(self, api_base: Optional[str] = None, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(base_url=api_base) if api_base else anthropic.Anthropic()
        self.client = client

    def request_line(
        self, custom_id: str, model: str, profile: PromptProfile, page: PageImage, opts: dict
    ) -> str:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": opts["max_tokens"],
            "system": profile.markdown_system,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": page.mime,
                                "data": base64.standard_b64encode(page.content).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": profile.markdown_user},
                    ],
                }
            ],
        }
        if opts.get("temperature") is not None:  # current Claude models reject it; only if asked
            params["temperature"] = opts["temperature"]
        if opts.get("reasoning_effort"):
            params["output_config"] = {"effort": opts["reasoning_effort"]}
        return json.dumps({"custom_id": custom_id, "params": params})

    def upload(self, shard: Path) -> Optional[str]:
        return None  # requests go inline in the create call

    def create(self, shard: Path, input_ref: Optional[str], job_id: str) -> str:
        with open(shard, encoding="utf-8") as fh:
            requests = [json.loads(line) for line in fh if line.strip()]
        # One attempt only (see the OpenAI backend): a retried 190 MB POST could double-bill.
        return self.client.with_options(max_retries=0).messages.batches.create(requests=requests).id

    def find_orphan(self, entry: dict, job_id: str, known: set[str]) -> Optional[str]:
        """Anthropic batches carry no metadata, so match on creation time and request count.
        Adopt only an unambiguous match; otherwise the pages show as missing and
        `--retry-failed` resubmits them."""
        started = entry.get("started_at") or 0
        matches = []
        for b in self.client.messages.batches.list(limit=50):  # newest first, auto-paginates
            created = b.created_at.timestamp()
            if created < started - 60:
                break  # everything further back is older than our submit
            c = b.request_counts
            total = c.processing + c.succeeded + c.errored + c.canceled + c.expired
            if b.id not in known and created <= started + 900 and total == entry["requests"]:
                matches.append(b.id)
        return matches[0] if len(matches) == 1 else None

    def info(self, batch_id: str) -> BatchInfo:
        b = self.client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        return BatchInfo(
            id=b.id,
            status=b.processing_status,
            done=b.processing_status == "ended",
            total=c.processing + c.succeeded + c.errored + c.canceled + c.expired,
            completed=c.succeeded,
            failed=c.errored + c.canceled + c.expired,
        )

    def results(self, batch_id: str) -> Iterator[PageResult]:
        for r in self.client.messages.batches.results(batch_id):
            result = r.result
            if result.type == "succeeded":
                msg = result.message
                text = "".join(b.text for b in msg.content if b.type == "text")
                if msg.stop_reason == "refusal":
                    yield PageResult(r.custom_id, None, error="model refused this page")
                elif text.strip() or msg.stop_reason == "end_turn":
                    truncated = msg.stop_reason in {"max_tokens", "model_context_window_exceeded"}
                    yield PageResult(r.custom_id, text, truncated=truncated)
                else:
                    yield PageResult(r.custom_id, None, error=f"empty ({msg.stop_reason})")
            elif result.type == "errored":
                err = result.error
                detail = getattr(getattr(err, "error", None), "message", None) or str(err)
                yield PageResult(r.custom_id, None, error=detail[:300])
            else:  # canceled / expired
                yield PageResult(r.custom_id, None, error=result.type)


def _provider_and_model(model: str) -> tuple[str, str]:
    if model.startswith("anthropic/"):
        return "anthropic", model.split("/", 1)[1]
    if model.startswith("claude"):
        return "anthropic", model
    if model.startswith("openai/"):
        return "openai", model.split("/", 1)[1]
    if "/" in model:
        raise ValueError(
            f"Batch mode supports OpenAI (e.g. gpt-5.4-mini) and Anthropic (e.g. "
            f"anthropic/claude-opus-5) models, got '{model}'. Use the regular `markdownify` "
            "command for other providers."
        )
    return "openai", model


def _backend(manifest: dict[str, Any], client=None):
    cls = _AnthropicBackend if manifest["provider"] == "anthropic" else _OpenAIBackend
    return cls(api_base=manifest.get("api_base"), client=client)


# --------------------------------------------------------------------------------------------
# Job state
# --------------------------------------------------------------------------------------------


def _state(out_dir: Path) -> Path:
    return out_dir / STATE_DIR


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    path = _state(out_dir) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"No batch job found in {out_dir} (missing {path})")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("provider", "openai")
    manifest.setdefault("page_errors", {})
    return manifest


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


def _custom_id(job_id: str, doc: int, page: int) -> str:
    """Request id for one page. The job prefix means results from any other job's batch can never
    be mistaken for ours, even if crash recovery adopted the wrong batch."""
    return f"j{job_id[:8]}-d{doc}-p{page}"


def _doc_index(custom_id: str) -> int:
    return int(custom_id.split("-")[1][1:])


def _iter_pages(path: Path, opts: dict[str, Any]) -> Iterator[PageImage]:
    if path.suffix.lower() == ".pdf":
        return iter_pdf_pages(path, opts["dpi"], opts["max_image_px"], opts["image_format"])
    return iter_image_pages(path, opts["max_image_px"], opts["image_format"])


def _request_lines(
    backend, manifest: dict[str, Any], wanted: Optional[set[str]] = None
) -> Iterator[tuple[str, str]]:
    """Yield (custom_id, jsonl line) for every page, or only the custom_ids in `wanted`.

    Pages are rendered one at a time, so memory stays flat however long the documents are. A
    document that fails to render is recorded in the manifest and skipped; the rest continue.
    """
    opts = manifest["options"]
    profile = load_prompt_profile(manifest["profile"])
    wanted_docs = {_doc_index(c) for c in wanted} if wanted is not None else None

    for doc_index, doc in enumerate(manifest["documents"]):
        if wanted_docs is not None and doc_index not in wanted_docs:
            continue
        count = 0
        try:
            for page in _iter_pages(Path(doc["input"]), opts):
                count += 1
                custom_id = _custom_id(manifest["job_id"], doc_index, page.index)
                if wanted is not None and custom_id not in wanted:
                    continue
                yield (
                    custom_id,
                    backend.request_line(custom_id, manifest["model"], profile, page, opts),
                )
        except Exception as e:  # noqa: BLE001 - one bad file must not sink the whole job
            doc["error"] = f"{type(e).__name__}: {e}"[:300]
            logger.error("Could not render %s: %s", doc["input"], doc["error"])
            continue
        doc["pages"] = count
        doc.pop("error", None)


def _submit_lines(
    backend, out_dir: Path, manifest: dict[str, Any], lines: Iterator[tuple[str, str]]
) -> int:
    """Write requests into size-capped shards and start one batch per shard."""
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
            input_ref = backend.upload(Path(shard.name))
            # Record the shard before starting its batch: if we crash in between, collect finds
            # the entry without an id and adopts (or restarts) the batch instead of losing pages.
            entry = {
                "id": None,
                "input_file_id": input_ref,
                "requests": len(shard_ids),
                "started_at": time.time(),
            }
            manifest["batches"].append(entry)
            _save_manifest(out_dir, manifest)
            entry["id"] = backend.create(Path(shard.name), input_ref, manifest["job_id"])
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


def _recover_unstarted(backend, out_dir: Path, manifest: dict[str, Any]) -> None:
    """Resolve shards recorded without a batch id (the process died between the two steps)."""
    unstarted = [b for b in manifest["batches"] if not b.get("id")]
    if not unstarted:
        return
    known = {b["id"] for b in manifest["batches"] if b.get("id")}
    for entry in unstarted:
        found = backend.find_orphan(entry, manifest["job_id"], known)
        if found:
            entry["id"] = found
            known.add(found)
            logger.info("Recovered batch %s after an interrupted submit", found)
        elif entry.get("input_file_id"):  # uploaded but never started: start it now
            entry["id"] = backend.create(Path(), entry["input_file_id"], manifest["job_id"])
            known.add(entry["id"])
            logger.info("Started batch %s for an upload an earlier run left behind", entry["id"])
        else:
            # Nothing to adopt and nothing uploaded to restart from: forget the shard. Its pages
            # show as missing and `--retry-failed` resubmits them.
            logger.warning("Could not recover an interrupted batch; its pages will be retried")
            entry["dropped"] = True
    manifest["batches"] = [b for b in manifest["batches"] if not b.get("dropped")]
    _save_manifest(out_dir, manifest)


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


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
    """Render every page of `inputs` and submit them as batch jobs. Returns out_dir.

    `model` picks the provider: `gpt-...`/`openai/...` use the OpenAI Batch API,
    `claude-...`/`anthropic/...` use Anthropic Message Batches.
    """
    out = Path(out_dir)
    provider, model_name = _provider_and_model(model)  # validate before creating any state
    if provider == "anthropic" and reasoning_effort and reasoning_effort not in _ANTHROPIC_EFFORTS:
        raise ValueError(
            f"Anthropic effort must be one of {', '.join(_ANTHROPIC_EFFORTS)}, got '{reasoning_effort}'"
        )
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
        "provider": provider,
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
    backend = _backend(manifest, client)
    count = _submit_lines(backend, out, manifest, _request_lines(backend, manifest))
    _save_manifest(out, manifest)  # page counts and render errors were filled in while rendering
    failed_docs = sum(1 for d in manifest["documents"] if d.get("error"))
    logger.info(
        "Submitted %d pages from %d documents in %d %s batch(es)%s",
        count,
        len(files) - failed_docs,
        len(manifest["batches"]),
        provider,
        f"; {failed_docs} document(s) could not be rendered" if failed_docs else "",
    )
    return out


def batch_status(out_dir: str | Path, client=None) -> BatchStatus:
    out = Path(out_dir)
    manifest = _load_manifest(out)
    backend = _backend(manifest, client)
    _recover_unstarted(backend, out, manifest)
    rows = []
    for entry in manifest["batches"]:
        info = backend.info(entry["id"])
        rows.append(
            {
                "id": info.id,
                "status": info.status,
                "done": info.done,
                "total": info.total or entry["requests"],
                "completed": info.completed,
                "failed": info.failed,
                "error": info.error,
            }
        )
    return BatchStatus(
        batches=rows,
        total=sum(r["total"] for r in rows),
        completed=sum(r["completed"] for r in rows),
        failed=sum(r["failed"] for r in rows),
    )


def _page_path(out: Path, custom_id: str) -> Path:
    return _state(out) / "pages" / f"{custom_id}.md"


def collect_batch(out_dir: str | Path, *, retry_failed: bool = False, client=None) -> CollectResult:
    """Download finished pages and write every document whose pages are all done.

    Safe to call repeatedly. With `retry_failed`, pages that failed in finished batches are
    submitted again as a new batch.
    """
    out = Path(out_dir)
    manifest = _load_manifest(out)
    backend = _backend(manifest, client)
    _remove_stale_shards(out)
    _recover_unstarted(backend, out, manifest)
    result = CollectResult()
    (_state(out) / "pages").mkdir(parents=True, exist_ok=True)

    for entry in manifest["batches"]:
        if entry.get("collected"):
            if entry.get("error"):
                result.batch_errors[entry["id"]] = entry["error"]
            continue
        info = backend.info(entry["id"])
        if not info.done:
            result.pending_batches += 1
            continue
        prefix = f"j{manifest['job_id'][:8]}-"
        try:
            pages = list(backend.results(entry["id"]))
        except Exception as e:  # noqa: BLE001 - e.g. results expired; keep collecting the rest
            entry["error"] = f"could not download results: {type(e).__name__}: {e}"[:300]
            entry["collected"] = True
            result.batch_errors[entry["id"]] = entry["error"]
            _save_manifest(out, manifest)
            continue
        for page in pages:
            if not page.custom_id.startswith(prefix):
                logger.warning("Ignoring result %s from another job", page.custom_id)
                continue
            if page.text is None:
                manifest["page_errors"][page.custom_id] = page.error or "failed"
                continue
            if page.truncated:
                logger.warning("Page %s was truncated at max_tokens", page.custom_id)
            text = strip_markdown_fence(page.text).strip() if page.text.strip() else ""
            _page_path(out, page.custom_id).write_text(text, encoding="utf-8")
            manifest["page_errors"].pop(page.custom_id, None)
        if info.error:
            entry["error"] = info.error
            result.batch_errors[entry["id"]] = info.error
        entry["collected"] = True
        entry["status"] = info.status
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
            custom_id = _custom_id(manifest["job_id"], doc_index, page)
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
            backend, out, manifest, _request_lines(backend, manifest, wanted=missing_ids)
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
            batch_rejected = any(b["error"] for b in status.batches)
            may_retry = retry_failed and not retried and not batch_rejected
            result = collect_batch(out, retry_failed=may_retry, client=client)
            if result.resubmitted:
                retried = True
                continue
            return result
        time.sleep(poll_seconds)
