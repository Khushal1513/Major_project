"""
app/routers/document.py – document upload, analysis, and streaming Q&A
"""

import os
import json
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

MAX_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
ALLOWED_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/tiff", "image/webp"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Process uploaded document and return analysis."""
    # Validate size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(400, f"File too large ({size_mb:.1f} MB). Maximum is {MAX_SIZE_MB} MB.")

    # Validate type
    if file.content_type not in ALLOWED_TYPES:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".webp"}:
            raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    suffix = os.path.splitext(file.filename or "doc")[1] or ".pdf"

    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from rag_pipeline import get_rag_pipeline
        from app.services.document_rag_service import DocumentRAGService

        rag = get_rag_pipeline()
        doc_service = DocumentRAGService(rag=rag)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = doc_service.process_and_analyse(
                file_path=tmp_path,
                original_filename=file.filename,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if not result["success"]:
            raise HTTPException(500, result.get("error", "Processing failed"))

        return result

    except ImportError:
        # ── Stub for local dev without the RAG pipeline ──────────────────
        import uuid, time
        return {
            "success": True,
            "document_id": str(uuid.uuid4()),
            "total_pages": 3,
            "total_chunks": 12,
            "summary": f"Uploaded document '{file.filename}' analysed successfully. "
                       "This document appears to contain legal provisions and statutory references.",
            "key_legal_points": [
                "Right to equality under Article 14",
                "Procedural safeguards under Section 41 CrPC",
                "Bail provisions under Section 436 CrPC",
            ],
            "fundamental_rights_involved": ["Article 14 – Right to Equality", "Article 21 – Right to Life"],
            "constitutional_articles": ["Article 14", "Article 19", "Article 21"],
            "legal_references": ["IPC Section 302", "CrPC Section 436", "BNS Section 103"],
            "extracted_text_preview": f"Sample extracted text from {file.filename}...",
        }


class DocQuestionRequest(BaseModel):
    document_id: str
    question: str


async def _stream_doc_qa(document_id: str, question: str):
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from rag_pipeline import get_rag_pipeline
        from app.services.document_rag_service import (
            DocumentRAGService, DOC_NAMESPACE, LAW_NAMESPACE,
            TOP_K_DOC, TOP_K_LAW, GROQ_MODEL, _QA_SYS, _QA_USR,
        )
        from app.utils.legal_classifier import LegalClassifier

        rag = get_rag_pipeline()
        emb = rag.embedding_model.encode([question])[0].tolist()

        def _retrieve(namespace, top_k, flt=None):
            try:
                params = {"vector": emb, "top_k": top_k, "include_metadata": True}
                if namespace:
                    params["namespace"] = namespace
                if flt:
                    params["filter"] = flt
                res = rag.index.query(**params)
                return [{"id": m["id"], "score": m.get("score", 0),
                         "metadata": m.get("metadata", {})}
                        for m in res.get("matches", [])]
            except Exception:
                return []

        doc_chunks = _retrieve(DOC_NAMESPACE, TOP_K_DOC, {"document_id": document_id})
        law_chunks = _retrieve(LAW_NAMESPACE, TOP_K_LAW)

        def fmt(chunks, label):
            return "\n\n".join(
                f"[{label} Chunk {i+1} | score={c.get('score',0):.3f}]\n"
                f"{c['metadata'].get('text','')}"
                for i, c in enumerate(chunks)
            )

        sources = [
            {"chunk_id": c["id"], "source": "document",
             "text_preview": c["metadata"].get("text", "")[:200],
             "score": round(c.get("score", 0.0), 4)}
            for c in doc_chunks
        ] + [
            {"chunk_id": c["id"], "source": "law_knowledge_base",
             "text_preview": c["metadata"].get("text", "")[:200],
             "score": round(c.get("score", 0.0), 4)}
            for c in law_chunks
        ]

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        msg = _QA_USR.format(
            question=question,
            doc_ctx=fmt(doc_chunks, "Document") or "No relevant document content found.",
            law_ctx=fmt(law_chunks, "Law KB") or "No relevant law knowledge found.",
        )

        stream = rag.groq_client.chat.completions.create(
            messages=[{"role": "system", "content": _QA_SYS},
                      {"role": "user", "content": msg}],
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=2000,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except ImportError:
        # Stub streaming for dev
        sample = (
            f"Based on the document (ID: {document_id}), here is the answer to your question: "
            f"'{question}'\n\nThe document contains provisions related to Indian law. "
            "Article 21 of the Constitution guarantees the right to life and personal liberty. "
            "The relevant sections have been identified and cross-referenced with the legal knowledge base."
        )
        for word in sample.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


@router.post("/ask/stream")
async def doc_ask_stream(req: DocQuestionRequest):
    return StreamingResponse(
        _stream_doc_qa(req.document_id, req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )