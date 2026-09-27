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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .llm import _message_with_images, strip_markdown_fence
from .logging import get_logger
from .pager import IMAGE_SUFFIXES, ImageFormat, load_document_pages
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
    failed_pages: dict[str, str] = field(default_factory=dict)  # custom_id -> error
    resubmitted: int = 0
    pending_batches: int = 0


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
    """One .md per input, named after the input; duplicate stems get a numeric suffix."""
    seen: dict[str, int] = {}
    outputs = []
    for f in files:
        count = seen.get(f.stem, 0)
        seen[f.stem] = count + 1
        name = f.stem if count == 0 else f"{f.stem}-{count + 1}"
        outputs.append(out_dir / f"{name}.md")
    return outputs


def _custom_id(doc: int, page: int) -> str:
    return f"d{doc}-p{page}"


def _parse_custom_id(custom_id: str) -> tuple[int, int]:
    doc, page = custom_id.split("-")
    return int(doc[1:]), int(page[1:])


def _request_lines(
    manifest: dict[str, Any], wanted: Optional[set[str]] = None
) -> Iterator[tuple[str, str]]:
    """Yield (custom_id, jsonl line) for every page, or only the custom_ids in `wanted`."""
    opts = manifest["options"]
    profile = load_prompt_profile(manifest["profile"])
    body_extra: dict[str, Any] = {"max_completion_tokens": opts["max_tokens"]}
    if opts.get("temperature") is not None:
        body_extra["temperature"] = opts["temperature"]
    if opts.get("reasoning_effort"):
        body_extra["reasoning_effort"] = opts["reasoning_effort"]

    for doc_index, doc in enumerate(manifest["documents"]):
        if wanted is not None and not any(c.startswith(f"d{doc_index}-") for c in wanted):
            continue
        pages = load_document_pages(
            Path(doc["input"]),
            dpi=opts["dpi"],
            max_side=opts["max_image_px"],
            image_format=opts["image_format"],
        )
        doc["pages"] = len(pages)
        for page in pages:
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
            line = json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
            yield custom_id, line


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
            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"tool": "llm-markdownify"},
            )
            manifest["batches"].append(
                {"id": batch.id, "input_file_id": uploaded.id, "requests": len(shard_ids)}
            )
            _save_manifest(out_dir, manifest)  # persist each batch id as soon as it exists
            logger.info("Started batch %s with %d page requests", batch.id, len(shard_ids))
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

    manifest: dict[str, Any] = {
        "version": 1,
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
    }
    _save_manifest(out, manifest)
    client = client or _client(api_base)
    count = _submit_lines(client, out, manifest, _request_lines(manifest))
    _save_manifest(out, manifest)  # page counts were filled in while rendering
    logger.info(
        "Submitted %d pages from %d documents in %d batch(es)",
        count,
        len(files),
        len(manifest["batches"]),
    )
    return out


def batch_status(out_dir: str | Path, client=None) -> BatchStatus:
    out = Path(out_dir)
    manifest = _load_manifest(out)
    client = client or _client(manifest.get("api_base"))
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


def collect_batch(out_dir: str | Path, *, retry_failed: bool = False, client=None) -> CollectResult:
    """Download finished pages and write every document whose pages are all done.

    Safe to call repeatedly. With `retry_failed`, pages that failed in finished batches are
    submitted again as a new batch.
    """
    out = Path(out_dir)
    manifest = _load_manifest(out)
    client = client or _client(manifest.get("api_base"))
    result = CollectResult()
    (_state(out) / "pages").mkdir(parents=True, exist_ok=True)

    for entry in manifest["batches"]:
        if entry.get("collected"):
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
            content = (choices[0].get("message") or {}).get("content") if choices else None
            if response.get("status_code") == 200 and content and content.strip():
                if choices[0].get("finish_reason") == "length":
                    logger.warning("Page %s was truncated at max_tokens", custom_id)
                _page_path(out, custom_id).write_text(
                    strip_markdown_fence(content).strip(), encoding="utf-8"
                )
            else:
                error = row.get("error") or body.get("error") or "empty response"
                result.failed_pages[custom_id] = json.dumps(error)[:300]
        for row in _read_jsonl(client, batch.error_file_id):
            error = row.get("error") or (row.get("response") or {}).get("body", {}).get("error")
            result.failed_pages[row["custom_id"]] = json.dumps(error)[:300]
        entry["collected"] = True
        entry["status"] = batch.status
        _save_manifest(out, manifest)

    missing_ids: set[str] = set()
    for doc_index, doc in enumerate(manifest["documents"]):
        n_pages = doc.get("pages") or 0
        texts, missing = [], []
        for page in range(n_pages):
            path = _page_path(out, _custom_id(doc_index, page))
            if path.exists():
                texts.append(path.read_text(encoding="utf-8"))
            else:
                missing.append(page + 1)
                missing_ids.add(_custom_id(doc_index, page))
        output = Path(doc["output"])
        if missing:
            result.incomplete[str(output)] = missing
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n\n".join(texts).strip() + "\n", encoding="utf-8")
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
    """Poll until every batch finishes, then collect (resubmitting failures once if asked)."""
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
            result = collect_batch(out, retry_failed=retry_failed and not retried, client=client)
            if result.resubmitted and not retried:
                retried = True
                continue
            return result
        time.sleep(poll_seconds)
