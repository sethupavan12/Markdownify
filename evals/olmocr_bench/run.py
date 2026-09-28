# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Run llm-markdownify's public convert() over an olmOCR-Bench (subset) directory.

Writes <data>/<candidate>/<category>/<stem>_pg1_repeat1.md, the layout olmOCR's scorer expects, plus
<data>/<candidate>.stats.json with per-page latency, errors and token usage.

    python evals/olmocr_bench/run.py evals/olmocr_bench/data/subset15 my_run --model gpt-6-luna \
        --opt max_image_px=1536 --opt profile=generic

A page that fails to convert is written as an empty file and scored as failed, which is what a
user would get. Pages run in separate processes so this also benchmarks old releases.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import litellm

from llm_markdownify import convert

_usage = {"prompt": 0, "completion": 0, "calls": 0}
_lock = threading.Lock()


def _track_usage(kwargs, response, start, end) -> None:  # LiteLLM success callback
    usage = getattr(response, "usage", None)
    if usage:
        with _lock:
            _usage["prompt"] += usage.prompt_tokens or 0
            _usage["completion"] += usage.completion_tokens or 0
            _usage["calls"] += 1


def _parse_opt(raw: str) -> tuple[str, Any]:
    key, value = raw.split("=", 1)
    if value in ("true", "false"):
        return key, value == "true"
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def _convert_one(job: tuple[Path, Path, str, dict]) -> tuple[str, float, str | None, dict]:
    pdf, dst, model, options = job
    litellm.success_callback = [_track_usage]
    for key in _usage:
        _usage[key] = 0
    if dst.exists() and dst.stat().st_size > 0:  # empty files are failed pages: retry them
        return str(dst), 0.0, None, dict(_usage)
    dst.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    error = None
    try:
        # Cache off unless asked: a repeat run must call the model again to measure run-to-run noise.
        options = {"enable_cache": False, **options}
        convert(pdf, dst, model=model, log_level="quiet", **options)
    except Exception as e:  # noqa: BLE001 - record and score as a failed page
        error = repr(e)[:300]
        dst.write_text("")
    return str(dst), time.time() - start, error, dict(_usage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="subset dir containing pdfs/ and *.jsonl")
    parser.add_argument("candidate", help="output folder name, e.g. v1_gpt6luna")
    parser.add_argument("--model", required=True)
    parser.add_argument("--opt", action="append", default=[], help="convert() kwarg as key=value")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    options = dict(_parse_opt(o) for o in args.opt)
    pdf_root = args.data / "pdfs"
    jobs = [
        (
            pdf,
            args.data
            / args.candidate
            / f"{pdf.relative_to(pdf_root).with_suffix('')}_pg1_repeat1.md",
            args.model,
            options,
        )
        for pdf in sorted(pdf_root.rglob("*.pdf"))
    ]

    stats: dict[str, Any] = {"errors": {}, "latency": {}}
    totals = dict.fromkeys(_usage, 0)
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for dst, latency, error, usage in pool.map(_convert_one, jobs):
            stats["latency"][dst] = latency
            if error:
                stats["errors"][dst] = error
            for key in totals:
                totals[key] += usage[key]
    stats.update(
        usage=totals, wall_s=time.time() - started, n=len(jobs), model=args.model, options=options
    )
    (args.data / f"{args.candidate}.stats.json").write_text(json.dumps(stats, indent=1))

    latencies = sorted(stats["latency"].values())
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    print(
        f"{args.candidate}: pages={len(jobs)} errors={len(stats['errors'])} "
        f"wall={stats['wall_s']:.0f}s p50={p50:.1f}s tokens={totals}"
    )
    for dst, error in list(stats["errors"].items())[:5]:
        print(f"  error {dst}: {error}")


if __name__ == "__main__":
    main()
