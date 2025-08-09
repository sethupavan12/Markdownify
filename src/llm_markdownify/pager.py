# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List

import pypdfium2 as pdfium
from PIL import Image

from .logging import get_logger

try:
    from docx2pdf import convert as docx2pdf_convert  # type: ignore
except Exception:  # pragma: no cover - optional
    docx2pdf_convert = None  # type: ignore

logger = get_logger("llm_markdownify.pager")


@dataclass
class PageImage:
    index: int
    width: int
    height: int
    content: bytes  # PNG bytes

    @property
    def data_url(self) -> str:
        b64 = base64.b64encode(self.content).decode("ascii")
        return f"data:image/png;base64,{b64}"


def _docx_to_pdf(input_path: Path) -> Path:
    if docx2pdf_convert is None:
        raise RuntimeError(
            "DOCX support requires 'docx2pdf' and platform support for Word/COM. Prefer PDFs."
        )
    temp_pdf = input_path.with_suffix(".converted.pdf")
    logger.info("Converting DOCX to PDF: %s -> %s", input_path, temp_pdf)
    docx2pdf_convert(str(input_path), str(temp_pdf))
    return temp_pdf


def iter_pdf_pages_as_images(pdf_path: Path, dpi: int) -> Iterable[PageImage]:
    logger.info("Rendering PDF pages to images at %s DPI", dpi)
    pdf = pdfium.PdfDocument(str(pdf_path))
    num_pages = len(pdf)
    scale = dpi / 72.0
    for i in range(num_pages):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        with BytesIO() as buf:
            pil_image.save(buf, format="PNG")
            data = buf.getvalue()
        yield PageImage(index=i, width=pil_image.width, height=pil_image.height, content=data)


def load_document_pages(input_path: Path, dpi: int, allow_docx: bool = False) -> List[PageImage]:
    """Load a PDF, image (PNG/JPG/JPEG), or DOCX (if allowed) as a list of page images.

    - PDF: rendered to page PNGs using pypdfium2 at the provided DPI
    - Image (.png/.jpg/.jpeg): treated as a single page; converted to PNG bytes
    - DOCX: optionally converted to PDF, then rendered like a PDF
    """
    suffix = input_path.suffix.lower()
    if suffix == ".docx":
        if not allow_docx:
            raise ValueError(
                "DOCX not allowed. Prefer exporting DOCX to PDF, or enable --allow-docx (requires Word/COM)."
            )
        pdf_path = _docx_to_pdf(input_path)
        pages = list(iter_pdf_pages_as_images(pdf_path, dpi=dpi))
        logger.info("Loaded %d pages", len(pages))
        return pages

    if suffix == ".pdf":
        pages = list(iter_pdf_pages_as_images(input_path, dpi=dpi))
        logger.info("Loaded %d pages", len(pages))
        return pages

    if suffix in {".png", ".jpg", ".jpeg"}:
        with Image.open(input_path) as img:
            rgb_img = img.convert("RGB")
            with BytesIO() as buf:
                rgb_img.save(buf, format="PNG")
                data = buf.getvalue()
            page = PageImage(index=0, width=rgb_img.width, height=rgb_img.height, content=data)
        logger.info("Loaded 1 image page from %s", input_path)
        return [page]

    raise ValueError(
        f"Unsupported input type: {suffix}. Expected one of .pdf, .docx, .png, .jpg, .jpeg"
    )
