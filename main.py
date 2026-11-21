from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import logging
import tempfile
import os

from rag_pipeline import get_rag_pipeline
from app.services.document_rag_service import DocumentRAGService

logger = logging.getLogger(__name__)

app = FastAPI(title="Indian Laws AI API")

# Enable React Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with React URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once (bi-encoder, Pinecone, Groq, and now the cross-encoder)
rag_pipeline = get_rag_pipeline()
rag_pipeline.warmup_reranker()   # logs and continues if the cross-encoder can't load
doc_rag_service = DocumentRAGService(rag=rag_pipeline)


@app.get("/")
def home():
    return {"message": "Indian Laws AI Backend Running"}


# ---------------------------
# LAW QUESTION API
# ---------------------------
@app.post("/ask")
async def ask_question(data: dict):

    query = data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        top_k = int(data.get("top_k", 5))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="top_k must be an integer")
    top_k = max(1, min(top_k, 20))

    # Optional: ask for the Retrieval Analysis payload (off by default)
    include_analysis = bool(data.get("include_analysis", False))

    if not rag_pipeline.is_indian_law_query(query):
        return {
            "success": False,
            "answer": "I can only answer Indian law related questions."
        }

    # Stage 1 (bi-encoder -> Pinecone Top-20) + Stage 2 (cross-encoder -> Top-k)
    try:
        contexts, analysis = await run_in_threadpool(
            rag_pipeline.retrieve_with_analysis, query, top_k
        )
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {e}")

    if not contexts or all(not c["text"] for c in contexts):
        resp = {
            "success": True,
            "answer": "No relevant legal information found. "
                      "Please rephrase or mention specific sections/articles.",
            "sources": []
        }
        if include_analysis:
            resp["retrieval_analysis"] = analysis
        return resp

    context_text = "\n\n".join(
        f"[Source: {c['source']} - {c['section']}]\n{c['text']}"
        for c in contexts
    )

    response = rag_pipeline.groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content":
                "You are an expert Indian Laws AI Assistant."
            },
            {
                "role": "user",
                "content":
                f"Context:\n{context_text}\n\nQuestion:{query}"
            }
        ],
        temperature=0.3,
        max_tokens=2048
    )

    answer = response.choices[0].message.content

    result = {
        "success": True,
        "answer": answer,
        "sources": contexts
    }
    if include_analysis:
        result["retrieval_analysis"] = analysis
    return result


# ---------------------------
# DOCUMENT UPLOAD API
# ---------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = doc_rag_service.process_and_analyse(
            file_path=tmp_path,
            original_filename=file.filename
        )
    finally:
        os.remove(tmp_path)

    return result


# ---------------------------
# DOCUMENT QUESTION API
# ---------------------------
# NOTE: unchanged. This goes through DocumentRAGService.answer_question
# (app/services/document_rag_service.py), which was not provided, so this
# endpoint is still single-stage until that file is updated.
@app.post("/document-chat")
async def document_chat(data: dict):

    document_id = data.get("document_id")
    question = data.get("question")

    if not document_id or not question:
        raise HTTPException(
            status_code=400,
            detail="document_id and question required"
        )

    result = doc_rag_service.answer_question(
        document_id=document_id,
        question=question
    )

    return result