"""
app/main.py

FastAPI application entry point.
Mount the existing RAG routes alongside the new Document Analysis routes.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as document_router
from app.config import get_settings

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up shared resources on startup."""
    settings = get_settings()

    # Ensure upload directory exists
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Upload directory: %s", settings.UPLOAD_DIR)

    # Pre-warm heavy models (runs in the main thread; blocks until ready)
    # Comment these out if you prefer lazy initialisation.
    try:
        from app.api.dependencies import get_embedding_model, get_pinecone_index
        get_pinecone_index()
        get_embedding_model()
        logger.info("Core services warmed up.")
    except Exception as exc:
        logger.warning("Warm-up warning (non-fatal): %s", exc)

    yield  # ← application runs here

    logger.info("Application shutting down.")


# ── App factory ──────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Indian Constitution & Law RAG API",
        description=(
            "AI-powered legal assistant for the Indian Constitution, IPC/BNS, "
            "CrPC/BNSS, and other Indian statutes.  "
            "Now with **Document Analysis** – upload any PDF or image and ask questions."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ─────────────────────────────────────────────────────────
    # New document analysis endpoints
    app.include_router(document_router)

    # ── If you have existing routers, include them here ─────────────────
    # from your_existing_module.routes import existing_router
    # app.include_router(existing_router)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "message": "Indian Law RAG API is running.",
            "docs": "/docs",
            "features": [
                "Indian Constitution & Law Q&A",
                "Document Analysis (PDF / Image upload)",
                "Hybrid RAG with Pinecone + Groq",
            ],
        }

    return app


app = create_app()


# ── Run directly with: python -m app.main ───────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )