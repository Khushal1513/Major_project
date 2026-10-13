"""
ocr_processor.py  –  EasyOCR wrapper for image and scanned-PDF text extraction.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OCRExtractionResult:
    file_path: str
    text: str
    confidence: float
    word_count: int
    error: str | None = None


class OCRProcessor:
    """
    Lazy-initialises an EasyOCR Reader on first use to keep startup fast.

    Parameters
    ----------
    languages : EasyOCR language codes, e.g. ["en", "hi"]
    use_gpu   : Use CUDA if available
    """

    def __init__(self, languages: list[str] | None = None, use_gpu: bool = False):
        self.languages = languages or ["en", "hi"]
        self.use_gpu = use_gpu
        self._reader = None

    # ── public ───────────────────────────────────────────────────────────

    def extract_from_file(self, file_path: str | Path) -> OCRExtractionResult:
        file_path = Path(file_path)
        if not file_path.exists():
            return OCRExtractionResult(file_path=str(file_path), text="",
                                       confidence=0.0, word_count=0,
                                       error=f"Not found: {file_path}")
        try:
            reader = self._get_reader()
            raw = reader.readtext(str(file_path), detail=1)
            return self._parse(str(file_path), raw)
        except Exception as e:
            logger.exception("OCR file error")
            return OCRExtractionResult(file_path=str(file_path), text="",
                                       confidence=0.0, word_count=0, error=str(e))

    def extract_from_pil_image(self, image) -> OCRExtractionResult:
        try:
            import numpy as np
            reader = self._get_reader()
            raw = reader.readtext(np.array(image), detail=1)
            return self._parse("<pil_image>", raw)
        except Exception as e:
            logger.exception("OCR PIL error")
            return OCRExtractionResult(file_path="<pil_image>", text="",
                                       confidence=0.0, word_count=0, error=str(e))

    # ── private ──────────────────────────────────────────────────────────

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
                logger.info("Initialising EasyOCR (langs=%s gpu=%s)…", self.languages, self.use_gpu)
                self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
            except ImportError:
                raise RuntimeError("easyocr not installed. Run: pip install easyocr")
        return self._reader

    @staticmethod
    def _parse(file_path: str, raw: list) -> OCRExtractionResult:
        if not raw:
            return OCRExtractionResult(file_path=file_path, text="", confidence=0.0, word_count=0)
        texts, confs = [], []
        for item in raw:
            text, conf = (item[1], float(item[2])) if len(item) == 3 else (item[1], 1.0)
            if text.strip():
                texts.append(text.strip())
                confs.append(conf)
        full = " ".join(texts)
        return OCRExtractionResult(
            file_path=file_path,
            text=full,
            confidence=round(sum(confs) / len(confs), 4) if confs else 0.0,
            word_count=len(full.split()),
        )