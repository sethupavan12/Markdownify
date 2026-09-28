# Copyright (c) 2026 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

"""Turn whatever the caller gives us (a path, bytes, a file object or a URL) into one Document, and
parse page selections like "1-5,12,40-"."""

from __future__ import annotations

import io
import ipaddress
import os
import re
import socket
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Optional, Union
from urllib.parse import urlparse

Kind = Literal["pdf", "image", "docx"]
Source = Union[str, Path, bytes, bytearray, BinaryIO]

# Bigger downloads are refused; documents that large should be fetched and checked by the caller.
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
DOWNLOAD_TIMEOUT_S = 60.0  # per network read
DOWNLOAD_DEADLINE_S = 300.0  # whole download

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
_BMP_DIB_SIZES = {12, 40, 52, 56, 64, 108, 124}


@dataclass
class Document:
    """A document ready to render: either a file on disk or bytes in memory."""

    kind: Kind
    name: str  # for messages and the result; a file name or URL
    path: Optional[Path] = None
    data: Optional[bytes] = None


def is_url(source: object) -> bool:
    """True for http(s) URLs, whatever the case of the scheme."""
    return isinstance(source, str) and urlparse(source.strip()).scheme.lower() in ("http", "https")


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
        # BMP: "BM" alone is too common, so also check the DIB header size at offset 14.
        or (head.startswith(b"BM") and int.from_bytes(data[14:18], "little") in _BMP_DIB_SIZES)
    ):
        return "image"
    if head.startswith(b"PK\x03\x04"):  # a zip; only a Word document counts
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if "word/document.xml" in archive.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
    return None


def _redact(url: str) -> str:
    """URL for logs and results, without credentials or query string (presigned URLs carry
    secrets there)."""
    parts = urlparse(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}{parts.path}"


def _refuse_private_network(url: str) -> None:
    """Refuse addresses inside the machine or the local network (loopback, private ranges, cloud
    metadata endpoints). Checked for every request, including redirects."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise PermissionError(f"Only http(s) URLs can be downloaded: {_redact(url)}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    for info in socket.getaddrinfo(parts.hostname, port, proto=socket.IPPROTO_TCP):
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise PermissionError(
                f"Refusing to download from {parts.hostname} ({address}): it is a private, local or "
                "reserved address. Pass allow_private_urls=True (or set "
                "LLM_MARKDOWNIFY_ALLOW_PRIVATE_URLS=1) if this is intended."
            )


def _download(url: str, allow_private_urls: bool) -> tuple[bytes, Optional[str]]:
    import httpx  # installed with the openai and anthropic SDKs

    allow_private_urls = allow_private_urls or os.environ.get(
        "LLM_MARKDOWNIFY_ALLOW_PRIVATE_URLS"
    ) in ("1", "true", "yes")

    def check(request: "httpx.Request") -> None:
        if not allow_private_urls:
            _refuse_private_network(str(request.url))

    name = _redact(url)
    chunks: list[bytes] = []
    size = 0
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
    with httpx.Client(
        follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_S, event_hooks={"request": [check]}
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            declared = int(response.headers.get("content-length") or 0)
            if declared > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"{name} is {declared} bytes; the limit is {MAX_DOWNLOAD_BYTES}")
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"{name} is larger than the {MAX_DOWNLOAD_BYTES}-byte limit")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Downloading {name} took longer than {DOWNLOAD_DEADLINE_S}s"
                    )
                chunks.append(chunk)
            content_type = response.headers.get("content-type")
    return b"".join(chunks), content_type


def _kind_from_content_type(content_type: Optional[str], url: str) -> Optional[Kind]:
    main = (content_type or "").split(";")[0].strip().lower()
    if main == "application/pdf":
        return "pdf"
    if main.startswith("image/") and main != "image/svg+xml":
        return "image"
    if main in ("", "application/octet-stream", "binary/octet-stream"):
        return _kind_from_suffix(Path(urlparse(url).path).suffix)
    return None  # e.g. an HTML login or error page served at a .pdf URL


def load_source(source: Source, allow_private_urls: bool = False) -> Document:
    """Accept a path, bytes, a binary file object, or an http(s) URL.

    Strings are trusted input: a path is read from disk as given. URLs to private, local or
    reserved addresses are refused unless `allow_private_urls` is set.
    """
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
    if is_url(text):
        data, content_type = _download(text, allow_private_urls)
        kind = sniff_kind(data) or _kind_from_content_type(content_type, text)
        if kind is None:
            raise ValueError(f"{_redact(text)} did not return a PDF or an image ({content_type})")
        return Document(kind=kind, name=_redact(text), data=data)

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


class PageSelectionError(ValueError):
    """A --pages / pages= selection that is malformed or outside the document."""


_RANGE = re.compile(r"^\s*(\d*)\s*(?:(-)\s*(\d*))?\s*$")


def _page_ranges(spec: str) -> list[tuple[int, Optional[int]]]:
    """Split "1-5,12,40-" into 1-based (start, end) pairs; end None means "to the last page".

    Checks syntax only, so it is cheap even for open-ended ranges.
    """
    ranges: list[tuple[int, Optional[int]]] = []
    for part in spec.split(","):
        match = _RANGE.match(part)
        if not part.strip() or not match or not (match.group(1) or match.group(3)):
            raise PageSelectionError(f"Invalid page selection {part!r} in {spec!r}")
        first, dash, last = match.groups()
        start = int(first) if first else 1
        end: Optional[int] = (int(last) if last else None) if dash else start
        if start < 1 or (end is not None and end < 1):
            raise PageSelectionError(f"Page numbers start at 1: {part!r}")
        if end is not None and start > end:
            raise PageSelectionError(f"Page range {part!r} runs backwards")
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
            raise PageSelectionError(
                f"Page {end if end and end > total else start} is past the end of the document "
                f"({total} pages)"
            )
        selected.update(range(start - 1, end if end is not None else total))
    return sorted(selected)
