"""
app/config.py
Centralised configuration for the Indian Law RAG application.
All values are read from environment variables with sensible defaults.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    #  Existing RAG settings (already in your .env)                       #
    # ------------------------------------------------------------------ #
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "indian-law-rag")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ------------------------------------------------------------------ #
    #  Document Analysis settings (NEW)                                   #
    # ------------------------------------------------------------------ #
    # Where uploaded files are stored temporarily before processing
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")

    # Pinecone namespace for user-uploaded document chunks
    DOCUMENT_NAMESPACE: str = os.getenv("DOCUMENT_NAMESPACE", "user_documents")

    # Pinecone namespace for the existing Indian-law knowledge base
    LAW_NAMESPACE: str = os.getenv("LAW_NAMESPACE", "indian_laws")

    # Text chunking parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # How many chunks to retrieve from each namespace during hybrid search
    TOP_K_DOCUMENT: int = int(os.getenv("TOP_K_DOCUMENT", "5"))
    TOP_K_LAW: int = int(os.getenv("TOP_K_LAW", "5"))

    # Maximum file size accepted (bytes) – default 20 MB
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

    # OCR language list (EasyOCR language codes)
    OCR_LANGUAGES: list[str] = ["en", "hi"]  # English + Hindi

    # EasyOCR: use GPU if available
    OCR_USE_GPU: bool = os.getenv("OCR_USE_GPU", "false").lower() == "true"

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()