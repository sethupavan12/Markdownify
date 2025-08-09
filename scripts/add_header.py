#!/usr/bin/env python3
# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

HEADER_TEMPLATE = (
    "# Copyright (c) {year} {owner}\n"
    "# Licensed under the Apache License, Version 2.0. See LICENSE file for details.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)

SUPPORTED_EXTS = {".py", ".toml", ".yml", ".yaml"}


@dataclass
class Config:
    owner: str
    year: int


def should_process(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in SUPPORTED_EXTS:
        return False
    # skip files in virtualenvs or build dirs
    parts = {p.name for p in path.parents}
    if any(x in parts for x in {".venv", "venv", "build", "dist", ".git"}):
        return False
    return True


def has_header(text: str, owner: str) -> bool:
    head = text.splitlines()[:10]
    blob = "\n".join(head)
    return (
        "SPDX-License-Identifier: Apache-2.0" in blob or "Copyright (c)" in blob and owner in blob
    )


def find_insert_index(text: str, suffix: str) -> int:
    lines = text.splitlines()
    idx = 0
    if suffix == ".py":
        # Keep shebang and encoding cookie at top
        if lines and lines[0].startswith("#!"):
            idx = 1
        # encoding cookie in first two lines
        if len(lines) > idx and re.search(r"coding[:=]", lines[idx]):
            idx += 1
    return idx


def insert_header(text: str, header: str, idx: int) -> str:
    lines = text.splitlines()
    before = lines[:idx]
    after = lines[idx:]
    out_lines: List[str] = []
    out_lines.extend(before)
    # ensure blank line separation when needed
    if before and before[-1] and not before[-1].endswith("\n"):
        pass
    out_lines.extend(header.rstrip("\n").splitlines())
    if after and after[0] != "":
        out_lines.append("")
    out_lines.extend(after)
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "\n")


def process_file(path: Path, cfg: Config) -> bool:
    text = path.read_text(encoding="utf-8")
    if has_header(text, cfg.owner):
        return False
    header = HEADER_TEMPLATE.format(year=cfg.year, owner=cfg.owner)
    idx = find_insert_index(text, path.suffix.lower())
    updated = insert_header(text, header, idx)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Insert Apache-2.0 header into source files.")
    parser.add_argument("files", nargs="*", help="Files to process; if none, scans src/ and tests/")
    parser.add_argument(
        "--owner", default="Sethu Pavan Venkata Reddy Pastula", help="Owner name for header"
    )
    parser.add_argument("--year", type=int, default=datetime.now().year, help="Year for header")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = Config(owner=args.owner, year=args.year)

    paths: List[Path]
    if args.files:
        paths = [Path(p) for p in args.files]
    else:
        roots = [Path("src"), Path("tests"), Path("scripts")]
        paths = [p for root in roots if root.exists() for p in root.rglob("*")]

    changed = 0
    for p in paths:
        if should_process(p):
            try:
                if process_file(p, cfg):
                    changed += 1
            except Exception:
                # best-effort; skip problematic files
                pass

    print(f"Header inserted/updated in {changed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
