# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from llm_markdownify.pager import PageImage, load_document_pages


def _create_test_image(path: Path, width: int = 100, height: int = 100, fmt: str = "PNG"):
    """Create a simple test image."""
    img = Image.new("RGB", (width, height), color="white")
    img.save(path, format=fmt)


def test_page_image_data_url():
    """PageImage generates correct data URL."""
    # Create minimal PNG bytes
    img = Image.new("RGB", (10, 10), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    content = buf.getvalue()

    page = PageImage(index=0, width=10, height=10, content=content)

    url = page.data_url
    assert url.startswith("data:image/png;base64,")
    # Calling again returns cached value
    assert page.data_url is url


def test_page_image_continuation_url():
    """PageImage generates smaller JPEG for continuation checks."""
    # Create a larger image
    img = Image.new("RGB", (2000, 2000), color="blue")
    buf = BytesIO()
    img.save(buf, format="PNG")
    content = buf.getvalue()

    page = PageImage(index=0, width=2000, height=2000, content=content)

    cont_url = page.continuation_data_url
    assert cont_url.startswith("data:image/jpeg;base64,")
    # Should be smaller than original due to resize and JPEG compression
    assert len(cont_url) < len(page.data_url)


def test_load_document_pages_png(tmp_path: Path):
    """Load a PNG image as single page."""
    img_path = tmp_path / "test.png"
    _create_test_image(img_path, 200, 300)

    pages = load_document_pages(img_path, dpi=72)

    assert len(pages) == 1
    assert pages[0].index == 0
    assert pages[0].width == 200
    assert pages[0].height == 300


def test_load_document_pages_jpg(tmp_path: Path):
    """Load a JPG image as single page."""
    img_path = tmp_path / "test.jpg"
    _create_test_image(img_path, 150, 200, fmt="JPEG")

    pages = load_document_pages(img_path, dpi=72)

    assert len(pages) == 1
    assert pages[0].index == 0


def test_load_document_pages_jpeg(tmp_path: Path):
    """Load a JPEG image (alternate extension)."""
    img_path = tmp_path / "test.jpeg"
    _create_test_image(img_path, 150, 200, fmt="JPEG")

    pages = load_document_pages(img_path, dpi=72)

    assert len(pages) == 1


def test_load_document_pages_unsupported(tmp_path: Path):
    """Unsupported file type raises ValueError."""
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("hello")

    with pytest.raises(ValueError, match="Unsupported input type"):
        load_document_pages(txt_path, dpi=72)


def test_load_document_pages_docx_not_allowed(tmp_path: Path):
    """DOCX without allow_docx raises ValueError."""
    docx_path = tmp_path / "test.docx"
    docx_path.write_bytes(b"PK")  # Minimal zip-like header

    with pytest.raises(ValueError, match="DOCX not allowed"):
        load_document_pages(docx_path, dpi=72, allow_docx=False)
