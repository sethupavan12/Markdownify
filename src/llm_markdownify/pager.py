# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import tempfile
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterator, List, Literal

import pypdfium2 as pdfium
from PIL import Image, ImageOps

from .logging import get_logger

try:
    from docx2pdf import convert as docx2pdf_convert  # type: ignore
except Exception:  # pragma: no cover - optional
    docx2pdf_convert = None  # type: ignore

logger = get_logger("llm_markdownify.pager")

ImageFormat = Literal["jpeg", "png"]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
SUPPORTED_SUFFIXES = {".pdf", ".docx"} | IMAGE_SUFFIXES

# PDFium is not thread-safe: rendering from several threads at once crashes the interpreter.
# Every pypdfium2 call in this module must hold this lock.
_PDFIUM_LOCK = threading.Lock()

_JPEG_QUALITY = 90


@dataclass
class PageImage:
    index: int
    width: int
    height: int
    content: bytes  # encoded image bytes (see mime)
    mime: str = "image/png"
    _data_url: str | None = field(default=None, init=False, repr=False, compare=False)
    _continuation_data_url: str | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def data_url(self) -> str:
        if self._data_url is None:
            b64 = base64.b64encode(self.content).decode("ascii")
            self._data_url = f"data:{self.mime};base64,{b64}"
        return self._data_url

    @property
    def continuation_data_url(self) -> str:
        """Smaller JPEG data URL for LLM continuation checks.

        Downscale to max width 1024px preserving aspect ratio to reduce upload size/latency.
        Cached after first computation.
        """
        if self._continuation_data_url is None:
            with BytesIO(self.content) as buf:
                img = Image.open(buf)
                img.load()
            img = _fit(img.convert("RGB"), 1024)
            data = _encode(img, "jpeg", quality=70)
            b64 = base64.b64encode(data).decode("ascii")
            self._continuation_data_url = f"data:image/jpeg;base64,{b64}"
        return self._continuation_data_url


def _fit(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest side is at most max_side pixels. Never upscales."""
    longest = max(img.width, img.height)
    if longest <= max_side:
        return img
    ratio = max_side / float(longest)
    size = (max(1, round(img.width * ratio)), max(1, round(img.height * ratio)))
    return img.resize(size, Image.LANCZOS)


def _to_rgb(img: Image.Image) -> Image.Image:
    """Flatten transparency onto white (a plain convert('RGB') turns transparent areas black)."""
    if img.mode in ("RGBA", "LA", "P", "PA"):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def _encode(img: Image.Image, fmt: ImageFormat, quality: int = _JPEG_QUALITY) -> bytes:
    with BytesIO() as out:
        if fmt == "jpeg":
            img.save(out, format="JPEG", quality=quality, optimize=True)
        else:
            img.save(out, format="PNG", optimize=False)
        return out.getvalue()


def _page_from_pil(index: int, img: Image.Image, max_side: int, fmt: ImageFormat) -> PageImage:
    img = _fit(_to_rgb(img), max_side)
    return PageImage(
        index=index,
        width=img.width,
        height=img.height,
        content=_encode(img, fmt),
        mime=f"image/{fmt}",
    )


def iter_pdf_pages(
    pdf_path: Path, dpi: int, max_side: int = 2048, fmt: ImageFormat = "jpeg"
) -> Iterator[PageImage]:
    """Yield PDF pages one at a time, rendered at `dpi` and capped at `max_side` pixels.

    Streaming keeps memory flat for very long documents. The cap matters: vision APIs reject or
    silently downscale very large images, and large scanned pages at high DPI exceed their limits.
    """
    logger.info("Rendering PDF pages at %s DPI (max %spx)", dpi, max_side)
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        with _PDFIUM_LOCK:
            num_pages = len(pdf)
        for i in range(num_pages):
            # Hold the lock only while touching PDFium; encoding below runs concurrently.
            with _PDFIUM_LOCK:
                page = pdf[i]
                try:
                    width_pt, height_pt = page.get_size()
                    scale = min(dpi / 72.0, max_side / max(width_pt, height_pt, 1.0))
                    bitmap = page.render(scale=scale)
                    try:
                        # Copy out of PDFium memory and free the bitmap while still holding the
                        # lock, so no PDFium call happens later from a garbage-collector finalizer.
                        pil_image = bitmap.to_pil().copy()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
            yield _page_from_pil(i, pil_image, max_side, fmt)
    finally:
        with _PDFIUM_LOCK:
            pdf.close()


def iter_pdf_pages_as_images(
    pdf_path: Path, dpi: int, max_side: int = 2048, fmt: ImageFormat = "jpeg"
) -> List[PageImage]:
    """All pages of a PDF as a list (see `iter_pdf_pages`)."""
    return list(iter_pdf_pages(pdf_path, dpi, max_side, fmt))


def iter_image_pages(path: Path, max_side: int, fmt: ImageFormat) -> Iterator[PageImage]:
    """Yield the pages of an image file. Multi-frame images (multi-page TIFF) give one per frame."""
    with Image.open(path) as img:
        n_frames = getattr(img, "n_frames", 1) if path.suffix.lower() in {".tif", ".tiff"} else 1
        for i in range(n_frames):
            img.seek(i)
            frame = ImageOps.exif_transpose(img)  # phone photos carry rotation in EXIF
            yield _page_from_pil(i, frame, max_side, fmt)


def _load_image_pages(path: Path, max_side: int, fmt: ImageFormat) -> List[PageImage]:
    return list(iter_image_pages(path, max_side, fmt))


def _docx_to_pdf(input_path: Path, out_dir: Path) -> Path:
    if docx2pdf_convert is None:
        raise RuntimeError(
            "DOCX support requires 'docx2pdf' and platform support for Word/COM. Prefer PDFs."
        )
    temp_pdf = out_dir / f"{input_path.stem}.pdf"
    logger.info("Converting DOCX to PDF: %s -> %s", input_path, temp_pdf)
    docx2pdf_convert(str(input_path), str(temp_pdf))
    return temp_pdf


def load_document_pages(
    input_path: Path,
    dpi: int,
    allow_docx: bool = False,
    max_side: int = 2048,
    image_format: ImageFormat = "jpeg",
) -> List[PageImage]:
    """Load a PDF, image, or DOCX (if allowed) as a list of page images.

    - PDF: rendered with pypdfium2 at `dpi`, longest side capped at `max_side`
    - Image (.png/.jpg/.jpeg/.webp/.tif/.tiff/.bmp/.gif): one page per frame, capped at `max_side`
    - DOCX: converted to PDF in a temp dir (via Word), then rendered like a PDF
    """
    suffix = input_path.suffix.lower()
    if suffix == ".docx":
        if not allow_docx:
            raise ValueError(
                "DOCX not allowed. Prefer exporting DOCX to PDF, or enable --allow-docx (requires Word/COM)."
            )
        with tempfile.TemporaryDirectory(prefix="llm-markdownify-") as tmp:
            pdf_path = _docx_to_pdf(input_path, Path(tmp))
            pages = iter_pdf_pages_as_images(pdf_path, dpi, max_side, image_format)
    elif suffix == ".pdf":
        pages = iter_pdf_pages_as_images(input_path, dpi, max_side, image_format)
    elif suffix in IMAGE_SUFFIXES:
        pages = _load_image_pages(input_path, max_side, image_format)
    else:
        raise ValueError(
            f"Unsupported input type: {suffix}. Expected one of {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    logger.info("Loaded %d page(s) from %s", len(pages), input_path)
    return pages
