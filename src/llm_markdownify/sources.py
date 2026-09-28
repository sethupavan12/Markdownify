# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Turn whatever the caller gives us (a path, bytes, a file object or a URL) into one Document, and
parse page selections like "1-5,12,40-"."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Optional, Union
from urllib.parse import urlparse

Kind = Literal["pdf", "image", "docx"]
Source = Union[str, Path, bytes, bytearray, BinaryIO]

# Bigger downloads are refused; documents that large should be fetched and checked by the caller.
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
DOWNLOAD_TIMEOUT_S = 60.0

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}


@dataclass
class Document:
    """A document ready to render: either a file on disk or bytes in memory."""

    kind: Kind
    name: str  # for messages and the result; a file name or URL
    path: Optional[Path] = None
    data: Optional[bytes] = None


def _kind_from_suffix(suffix: str) -> Optional[Kind]:
    suffix = suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    return None


def sniff_kind(data: bytes) -> Optional[Kind]:
    """Recognise a file type from its first bytes (every format we read has a fixed signature)."""
    head = data[:16]
    if head.startswith(b"%PDF"):
        return "pdf"
    if (
        head.startswith(b"\x89PNG")
        or head.startswith(b"\xff\xd8\xff")  # JPEG
        or head.startswith(b"GIF8")
        or head.startswith((b"II*\x00", b"MM\x00*"))  # TIFF
        or (head.startswith(b"RIFF") and head[8:12] == b"WEBP")
        or head.startswith(b"BM")
    ):
        return "image"
    if head.startswith(b"PK\x03\x04"):  # zip container; the only one we read is .docx
        return "docx"
    return None


def _download(url: str) -> tuple[bytes, Optional[str]]:
    import httpx  # installed with the openai and anthropic SDKs

    chunks: list[bytes] = []
    size = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_S) as response:
        response.raise_for_status()
        declared = int(response.headers.get("content-length") or 0)
        if declared > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"{url} is {declared} bytes; the limit is {MAX_DOWNLOAD_BYTES}")
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"{url} is larger than the {MAX_DOWNLOAD_BYTES}-byte limit")
            chunks.append(chunk)
        content_type = response.headers.get("content-type")
    return b"".join(chunks), content_type


def load_source(source: Source) -> Document:
    """Accept a path, bytes, a binary file object, or an http(s) URL."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        kind = sniff_kind(data)
        if kind is None:
            raise ValueError("Unrecognised document bytes: expected a PDF, an image, or a .docx")
        return Document(kind=kind, name="<bytes>", data=data)

    if hasattr(source, "read"):
        data = source.read()  # type: ignore[union-attr]
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("File objects must be opened in binary mode ('rb')")
        name = str(getattr(source, "name", "<file>"))
        kind = sniff_kind(bytes(data)) or _kind_from_suffix(Path(name).suffix)
        if kind is None:
            raise ValueError(f"Unrecognised document: {name}")
        return Document(kind=kind, name=name, data=bytes(data))

    text = str(source)
    if text.startswith(("http://", "https://")):
        data, content_type = _download(text)
        kind = sniff_kind(data)
        if kind is None and content_type:
            kind = "pdf" if "pdf" in content_type else "image" if "image" in content_type else None
        kind = kind or _kind_from_suffix(Path(urlparse(text).path).suffix)
        if kind is None:
            raise ValueError(f"{text} did not return a PDF or an image ({content_type})")
        return Document(kind=kind, name=text, data=data)

    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    kind = _kind_from_suffix(path.suffix)
    if kind is None:
        raise ValueError(
            f"Unsupported input type: {path.suffix}. Expected a PDF, an image "
            f"({', '.join(sorted(_IMAGE_SUFFIXES))}) or .docx"
        )
    return Document(kind=kind, name=str(path), path=path)


_RANGE = re.compile(r"^\s*(\d+)?\s*(-)?\s*(\d+)?\s*$")


def _page_ranges(spec: str) -> list[tuple[int, Optional[int]]]:
    """Split "1-5,12,40-" into 1-based (start, end) pairs; end None means "to the last page".

    Checks syntax only, so it is cheap even for open-ended ranges.
    """
    ranges: list[tuple[int, Optional[int]]] = []
    for part in spec.split(","):
        match = _RANGE.match(part)
        if not part.strip() or not match or not (match.group(1) or match.group(3)):
            raise ValueError(f"Invalid page selection {part!r} in {spec!r}")
        first, dash, last = match.groups()
        start = int(first) if first else 1
        end: Optional[int] = (int(last) if last else None) if dash else start
        if start < 1 or (end is not None and end < 1):
            raise ValueError(f"Page numbers start at 1: {part!r}")
        if end is not None and start > end:
            raise ValueError(f"Page range {part!r} runs backwards")
        ranges.append((start, end))
    return ranges


def validate_pages(spec: str) -> None:
    """Raise ValueError if `spec` is not a valid page selection (does not need the page count)."""
    _page_ranges(spec)


def parse_pages(spec: str, total: int) -> list[int]:
    """Parse a 1-based page selection into sorted 0-based indices.

    Accepts "3", "1-5", "40-" (to the end), "-3" (from the start), comma-separated in any mix:
    "1-5,12,40-". Page numbers beyond the document are an error, except in open-ended ranges.
    """
    selected: set[int] = set()
    for start, end in _page_ranges(spec):
        if start > total or (end is not None and end > total):
            raise ValueError(
                f"Page {end if end and end > total else start} is past the end of the document "
                f"({total} pages)"
            )
        selected.update(range(start - 1, end if end is not None else total))
    return sorted(selected)
