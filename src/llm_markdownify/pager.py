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
    """Load a PDF (preferred) or DOCX (if allowed) as a list of rendered page images."""
    suffix = input_path.suffix.lower()
    if suffix == ".docx":
        if not allow_docx:
            raise ValueError(
                "DOCX not allowed. Prefer exporting DOCX to PDF, or enable --allow-docx (requires Word/COM)."
            )
        pdf_path = _docx_to_pdf(input_path)
    else:
        pdf_path = input_path

    pages = list(iter_pdf_pages_as_images(pdf_path, dpi=dpi))
    logger.info("Loaded %d pages", len(pages))
    return pages
