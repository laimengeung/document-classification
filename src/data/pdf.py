"""Load any supported document file (image or PDF) as preprocessed page images.

RVL-CDIP has no PDFs, but the client's bulk imports do (Phase 2), so the input
pipeline accepts both now (PRD 5.1.1). Each PDF page is rendered with PyMuPDF
and then prepared exactly like an RVL-CDIP image.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data.preprocess import WORKING_COPY_SIZE, to_working_copy

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_PDF_DPI = 150


def render_pdf(path: str | Path, dpi: int = DEFAULT_PDF_DPI) -> list[Image.Image]:
    """Render every page of a PDF to an 8-bit grayscale PIL image at ``dpi``."""
    import pymupdf  # imported here so the rest of the package works without PyMuPDF

    pages = []
    with pymupdf.open(path) as document:
        for page in document:
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            pages.append(Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples))
    return pages


def load_document_pages(path: str | Path, size: int = WORKING_COPY_SIZE,
                        dpi: int = DEFAULT_PDF_DPI) -> list[Image.Image]:
    """Return one preprocessed (grayscale, ``size`` x ``size``) image per page.

    Images give one page (the first frame for multi-page TIFFs); PDFs give one per page.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return [to_working_copy(page, size) for page in render_pdf(path, dpi)]
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            return [to_working_copy(image, size)]
    raise ValueError(f"Unsupported file type: {path.suffix}")
