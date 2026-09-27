# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Build a fixed, stratified subset of olmOCR-Bench for fast iteration.

Picks N PDFs per category (seeded), keeps all of their tests, and downloads only those PDFs.

    python evals/olmocr_bench/make_subset.py --per-category 15 --out evals/olmocr_bench/data/subset15
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/allenai/olmOCR-bench/resolve/main/bench_data"
CATEGORIES = [
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
]


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out: Path = args.out
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    needed: list[str] = []
    for category in CATEGORIES:
        raw = out / ".source" / f"{category}.jsonl"
        _download(f"{HF_BASE}/{category}.jsonl", raw)
        by_pdf: dict[str, list[dict]] = defaultdict(list)
        for line in raw.read_text().splitlines():
            if line.strip():
                test = json.loads(line)
                by_pdf[test["pdf"]].append(test)
        pdfs = sorted(by_pdf)
        random.Random(args.seed).shuffle(pdfs)
        picked = sorted(pdfs[: args.per_category])
        tests = [t for pdf in picked for t in by_pdf[pdf]]
        (out / f"{category}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tests))
        needed += picked
        print(f"{category}: {len(picked)} pdfs, {len(tests)} tests")

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda p: _download(f"{HF_BASE}/pdfs/{p}", out / "pdfs" / p), needed))
    print(f"downloaded {len(needed)} pdfs to {out / 'pdfs'}")


if __name__ == "__main__":
    main()
