"""
main.py – Indian Laws AI Assistant (FastAPI)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, document

app = FastAPI(
    title="Indian Laws AI Assistant",
    description="RAG-powered Indian Law assistant with streaming responses",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(document.router, prefix="/api/document", tags=["Document"])


@app.get("/")
async def root():
    return {"status": "ok", "message": "Indian Laws AI API is running"}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}