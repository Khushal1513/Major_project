"""
app/api/dependencies.py

FastAPI dependency injection for shared service instances.
All heavy objects (Pinecone, SentenceTransformer, Groq, etc.) are
initialised once at startup and reused across requests.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Singleton factories
#  (decorated with @lru_cache so they are created once per process)
# ──────────────────────────────────────────────────────────────

@lru_cache()
def get_pinecone_index():
    """Initialise and return the Pinecone Index object."""
    from pinecone import Pinecone

    settings = get_settings()
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    logger.info("Pinecone index '%s' connected.", settings.PINECONE_INDEX_NAME)
    return index


@lru_cache()
def get_embedding_model():
    """Load and return the SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    logger.info("Loading embedding model: %s …", settings.EMBEDDING_MODEL)
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")
    return model


@lru_cache()
def get_groq_client():
    """Initialise and return the Groq client."""
    from groq import Groq

    settings = get_settings()
    client = Groq(api_key=settings.GROQ_API_KEY)
    logger.info("Groq client initialised.")
    return client


@lru_cache()
def get_indian_law_rag():
    """
    Instantiate your existing IndianLawRAG class.
    Adjust the import path to match your actual module location.
    """
    # ── IMPORTANT: update this import to match your project layout ──────
    # e.g. from rag.indian_law_rag import IndianLawRAG
    from rag.indian_law_rag import IndianLawRAG   # <── adjust this

    settings = get_settings()
    rag = IndianLawRAG(
        pinecone_api_key=settings.PINECONE_API_KEY,
        pinecone_index_name=settings.PINECONE_INDEX_NAME,
        groq_api_key=settings.GROQ_API_KEY,
        embedding_model_name=settings.EMBEDDING_MODEL,
        groq_model=settings.GROQ_MODEL,
    )
    logger.info("IndianLawRAG pipeline initialised.")
    return rag


def get_document_processor():
    """Create (and return) a DocumentProcessor instance."""
    from app.services.document_processor import DocumentProcessor

    return DocumentProcessor(
        pinecone_index=get_pinecone_index(),
        embedding_model=get_embedding_model(),
    )


def get_document_rag_service():
    """Create (and return) a DocumentRAGService instance."""
    from app.services.document_rag_service import DocumentRAGService

    return DocumentRAGService(
        indian_law_rag=get_indian_law_rag(),
        pinecone_index=get_pinecone_index(),
        embedding_model=get_embedding_model(),
        groq_client=get_groq_client(),
    )