from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from rag_pipeline import get_rag_pipeline
from app.services.document_rag_service import DocumentRAGService

app = FastAPI(title="Indian Laws AI API")

# Enable React Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with React URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once
rag_pipeline = get_rag_pipeline()
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
    top_k = data.get("top_k", 5)

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    if not rag_pipeline.is_indian_law_query(query):
        return {
            "success": False,
            "answer": "I can only answer Indian law related questions."
        }

    contexts = rag_pipeline.retrieve_context(query, top_k)

    context_text = "\n\n".join(
        f"[Source: {c['source']} - {c['section']}]\n{c['text']}"
        for c in contexts
    )

    response = rag_pipeline.groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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

    return {
        "success": True,
        "answer": answer,
        "sources": contexts
    }


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