"""
app/routers/chat.py – streaming chat endpoint
"""

import json
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    session_id: Optional[str] = None


async def _stream_rag(query: str, top_k: int):
    """
    Async generator that yields SSE events.
    Replace the body with your real RAG pipeline calls.
    """
    try:
        # ── 1. Import your existing pipeline ────────────────────────────
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from rag_pipeline import get_rag_pipeline

        rag = get_rag_pipeline()

        # ── 2. Validate query ────────────────────────────────────────────
        if not rag.is_indian_law_query(query):
            payload = json.dumps({
                "type": "error",
                "content": "I can only answer questions related to Indian Constitution and Laws (IPC, CrPC, BNS, BNSS, etc.).",
            })
            yield f"data: {payload}\n\n"
            return

        # ── 3. Retrieve context ──────────────────────────────────────────
        contexts = rag.retrieve_context(query, top_k)
        if not contexts or all(c["text"] == "" for c in contexts):
            payload = json.dumps({
                "type": "error",
                "content": "No relevant legal information found. Please rephrase or mention specific sections/articles.",
            })
            yield f"data: {payload}\n\n"
            return

        # ── 4. Send context metadata ─────────────────────────────────────
        ctx_payload = json.dumps({"type": "contexts", "contexts": contexts})
        yield f"data: {ctx_payload}\n\n"

        # ── 5. Build prompt ──────────────────────────────────────────────
        context_text = "\n\n".join(
            f"[Source: {c['source']} - {c['section']}]\n{c['text']}" for c in contexts
        )
        system_prompt = (
            "You are an expert AI assistant specialised in Indian Constitution and Laws, "
            "including IPC, CrPC, BNS, BNSS, and other Indian legal statutes.\n"
            "Provide accurate legal explanations, cite sections/articles when relevant, "
            "and explain concepts clearly. Use only the provided legal context."
        )
        user_prompt = (
            f"Based on the following legal context, answer the question.\n\n"
            f"CONTEXT:\n{context_text}\n\nUSER QUESTION:\n{query}\n\nAnswer:"
        )

        # ── 6. Stream Groq tokens ────────────────────────────────────────
        stream = rag.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048,
            top_p=0.9,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                payload = json.dumps({"type": "token", "content": delta})
                yield f"data: {payload}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        payload = json.dumps({"type": "error", "content": str(e)})
        yield f"data: {payload}\n\n"


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        _stream_rag(req.query, req.top_k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )