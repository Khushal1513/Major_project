"""
pdf_processor.py  –  Extract text from PDFs using PyMuPDF (fitz).
Falls back to OCR for scanned/image-only pages.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionResult:
    file_path: str
    total_pages: int
    full_text: str
    page_texts: list[str] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)
    extraction_method: str = "text"   # 'text' | 'ocr' | 'mixed'
    error: str | None = None


class PDFProcessor:
    """
    Extracts text from PDF files page-by-page.

    Parameters
    ----------
    ocr_fallback    : Pass an OCRProcessor instance to OCR scanned pages.
    min_text_length : Pages with fewer chars than this get sent to OCR.
    """

    def __init__(self, ocr_processor=None, min_text_length: int = 50) -> None:
        self._ocr = ocr_processor
        self.min_text_length = min_text_length

    def extract(self, file_path: str | Path) -> PDFExtractionResult:
        try:
            import fitz
        except ImportError:
            return PDFExtractionResult(
                file_path=str(file_path), total_pages=0, full_text="",
                error="PyMuPDF not installed. Run: pip install PyMuPDF"
            )

        file_path = Path(file_path)
        if not file_path.exists():
            return PDFExtractionResult(
                file_path=str(file_path), total_pages=0, full_text="",
                error=f"File not found: {file_path}"
            )

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            return PDFExtractionResult(
                file_path=str(file_path), total_pages=0, full_text="",
                error=f"Cannot open PDF: {e}"
            )

        page_texts, scanned_pages, methods = [], [], set()

        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text").strip()

            if len(text) >= self.min_text_length:
                page_texts.append(text)
                methods.add("text")
            elif self._ocr is not None:
                logger.info("Page %d sparse (%d chars) – falling back to OCR", i + 1, len(text))
                ocr_text = self._ocr_page(page, i + 1)
                page_texts.append(ocr_text)
                scanned_pages.append(i + 1)
                methods.add("ocr")
            else:
                page_texts.append(text)
                if len(text) < self.min_text_length:
                    scanned_pages.append(i + 1)

        doc.close()

        if "text" in methods and "ocr" in methods:
            method = "mixed"
        elif "ocr" in methods:
            method = "ocr"
        else:
            method = "text"

        return PDFExtractionResult(
            file_path=str(file_path),
            total_pages=len(page_texts),
            full_text="\n\n".join(page_texts),
            page_texts=page_texts,
            scanned_pages=scanned_pages,
            extraction_method=method,
        )

    def _ocr_page(self, page, page_number: int) -> str:
        try:
            import fitz, io
            from PIL import Image
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return self._ocr.extract_from_pil_image(img).text
        except Exception as e:
            logger.warning("OCR failed page %d: %s", page_number, e)
            return ""