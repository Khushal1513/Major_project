"""
app/api/routes.py

FastAPI router for the Document Analysis feature.

Endpoints
---------
POST /upload-document   – accept a file, process it, return analysis
POST /ask-document      – answer a question about an uploaded document
GET  /document/{doc_id} – retrieve metadata for a processed document
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_document_processor, get_document_rag_service
from app.config import get_settings
from app.models.request_models import DocumentQuestionRequest
from app.models.response_models import DocumentQAResponse, DocumentUploadResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Document Analysis"])

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/webp",
}
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}


# ──────────────────────────────────────────────────────────────
#  POST /upload-document
# ──────────────────────────────────────────────────────────────

@router.post(
    "/upload-document",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and analyse a legal document (PDF or image)",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or empty file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or image file to analyse"),
    doc_processor=Depends(get_document_processor),
    rag_service=Depends(get_document_rag_service),
):
    """
    Upload a legal document (PDF or image).

    The endpoint will:
    1. Validate the file type and size.
    2. Extract text via PyMuPDF or EasyOCR.
    3. Chunk the text and store embeddings in Pinecone.
    4. Generate a legal analysis using Groq.
    5. Return a DocumentUploadResponse with the analysis.
    """
    settings = get_settings()

    # ── Validate file extension ──────────────────────────────────────────
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # ── Check file size ──────────────────────────────────────────────────
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB} MB.",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Save to upload directory ─────────────────────────────────────────
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / filename

    try:
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info("Saved upload: %s (%d bytes)", save_path, len(contents))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # ── Process document ─────────────────────────────────────────────────
    try:
        processed = doc_processor.process(
            file_path=save_path,
            original_filename=filename,
        )
    except Exception as exc:
        logger.exception("Document processing failed")
        raise HTTPException(status_code=500, detail=f"Document processing error: {exc}")
    finally:
        # Clean up the temporary file
        try:
            os.remove(save_path)
        except OSError:
            pass

    if processed.error:
        raise HTTPException(status_code=422, detail=processed.error)

    # ── Generate legal analysis ──────────────────────────────────────────
    try:
        response = rag_service.analyse_document(processed)
    except Exception as exc:
        logger.exception("Document analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")

    return response


# ──────────────────────────────────────────────────────────────
#  POST /ask-document
# ──────────────────────────────────────────────────────────────

@router.post(
    "/ask-document",
    response_model=DocumentQAResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about an uploaded document",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "RAG pipeline error"},
    },
)
async def ask_document(
    request: DocumentQuestionRequest,
    rag_service=Depends(get_document_rag_service),
):
    """
    Answer a question about a previously uploaded document.

    Uses Hybrid RAG:
    - Retrieves relevant chunks from the uploaded document's namespace.
    - Retrieves relevant chunks from the Indian law knowledge base.
    - Combines both contexts and generates a grounded answer using Groq.

    Example questions:
    - "Summarize this document."
    - "Which constitutional articles are relevant here?"
    - "Does this notice affect my fundamental rights?"
    - "Explain clause 4 in simple language."
    """
    try:
        response = rag_service.answer_question(
            document_id=request.document_id,
            question=request.question,
        )
    except Exception as exc:
        logger.exception("Document Q&A failed")
        raise HTTPException(status_code=500, detail=f"Q&A error: {exc}")

    return response


# ──────────────────────────────────────────────────────────────
#  Health check (convenience)
# ──────────────────────────────────────────────────────────────

@router.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "service": "Indian Law RAG – Document Analysis"}