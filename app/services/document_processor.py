"""
document_processor.py
Orchestrates: detect type → extract text → classify → chunk → embed → upsert Pinecone.

Designed to work with the existing IndianLawRAG class – it reuses the same
Pinecone index and SentenceTransformer model already initialised by IndianLawRAG.
"""
from __future__ import annotations
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.services.ocr_processor import OCRProcessor
from app.services.pdf_processor import PDFProcessor
from app.utils.legal_classifier import LegalClassification, LegalClassifier
from app.utils.text_chunker import TextChunk, TextChunker

logger = logging.getLogger(__name__)

PDF_EXTS   = {".pdf"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DOC_NAMESPACE = os.getenv("DOCUMENT_NAMESPACE", "user_documents")
OCR_USE_GPU   = os.getenv("OCR_USE_GPU", "false").lower() == "true"


@dataclass
class ProcessedDocument:
    document_id: str
    filename: str
    file_type: str
    total_pages: int
    full_text: str
    chunks: list[TextChunk] = field(default_factory=list)
    legal_classification: LegalClassification | None = None
    pinecone_upserted: bool = False
    error: str | None = None


class DocumentProcessor:
    """
    Parameters
    ----------
    pinecone_index   : rag_instance.index   (already open)
    embedding_model  : rag_instance.embedding_model
    """

    def __init__(self, pinecone_index, embedding_model):
        self._index    = pinecone_index
        self._embedder = embedding_model
        self._ocr      = OCRProcessor(languages=["en", "hi"], use_gpu=OCR_USE_GPU)
        self._pdf      = PDFProcessor(ocr_processor=self._ocr)
        self._chunker  = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self._clf      = LegalClassifier()

    # ── public ───────────────────────────────────────────────────────────

    def process(self, file_path: str | Path, original_filename: str) -> ProcessedDocument:
        file_path   = Path(file_path)
        doc_id      = self._make_id(original_filename)
        file_type   = self._detect(file_path)

        if file_type is None:
            return ProcessedDocument(
                document_id=doc_id, filename=original_filename, file_type="unknown",
                total_pages=0, full_text="",
                error=f"Unsupported file type: {file_path.suffix}",
            )

        full_text, pages, err = self._extract(file_path, file_type)
        if err:
            return ProcessedDocument(document_id=doc_id, filename=original_filename,
                                     file_type=file_type, total_pages=pages,
                                     full_text="", error=err)
        if not full_text.strip():
            return ProcessedDocument(document_id=doc_id, filename=original_filename,
                                     file_type=file_type, total_pages=pages,
                                     full_text="", error="No text could be extracted.")

        clf    = self._clf.classify(full_text)
        chunks = self._chunker.split(full_text, doc_id,
                                     extra_metadata={"filename": original_filename,
                                                     "file_type": file_type})
        upserted = self._embed_upsert(chunks, doc_id)

        return ProcessedDocument(
            document_id=doc_id, filename=original_filename, file_type=file_type,
            total_pages=pages, full_text=full_text, chunks=chunks,
            legal_classification=clf, pinecone_upserted=upserted,
        )

    # ── private ──────────────────────────────────────────────────────────

    def _extract(self, path: Path, ftype: str):
        if ftype == "pdf":
            r = self._pdf.extract(path)
            return (r.full_text, r.total_pages, r.error)
        r = self._ocr.extract_from_file(path)
        return (r.text, 1, r.error)

    def _embed_upsert(self, chunks: list[TextChunk], doc_id: str) -> bool:
        if not chunks:
            return False
        try:
            embeddings = self._embedder.encode([c.text for c in chunks],
                                               show_progress_bar=False)
            vectors = [
                {"id": c.chunk_id,
                 "values": emb.tolist(),
                 "metadata": {**c.metadata, "text": c.text[:1000]}}
                for c, emb in zip(chunks, embeddings)
            ]
            for i in range(0, len(vectors), 100):
                self._index.upsert(vectors=vectors[i:i+100], namespace=DOC_NAMESPACE)
            logger.info("Upserted %d vectors for %s", len(vectors), doc_id)
            return True
        except Exception as e:
            logger.exception("Upsert failed: %s", e)
            return False

    @staticmethod
    def _detect(path: Path) -> str | None:
        ext = path.suffix.lower()
        if ext in PDF_EXTS:   return "pdf"
        if ext in IMAGE_EXTS: return "image"
        return None

    @staticmethod
    def _make_id(filename: str) -> str:
        safe = Path(filename).stem[:20].replace(" ", "_")
        return f"doc_{safe}_{uuid.uuid4().hex[:8]}"