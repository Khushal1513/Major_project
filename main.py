"""
FastAPI Server for IPC RAG AI System
REST API endpoints for legal question answering
"""

import os
import logging
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from rag_pipeline_extra import RAGPipeline, IPCAnswer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup RAG pipeline"""
    global rag_pipeline
    
    logger.info("Starting IPC RAG AI Server...")
    
    # Load environment variables
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/llama-2-7b-chat.Q4_K_M.gguf")
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ipc-rag")
    
    # Validate configuration
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY not set in environment")
        raise ValueError("PINECONE_API_KEY is required")
    
    # If the configured model path doesn't exist, try to auto-discover any .gguf in the models directory
    if not os.path.exists(LLAMA_MODEL_PATH):
        models_dir = os.path.dirname(LLAMA_MODEL_PATH) or "models"
        discovered_model = None
        try:
            if os.path.isdir(models_dir):
                gguf_files = [f for f in os.listdir(models_dir) if f.lower().endswith('.gguf')]
                if gguf_files:
                    discovered_model = os.path.join(models_dir, gguf_files[0])
        except Exception:
            discovered_model = None

        if discovered_model and os.path.exists(discovered_model):
            logger.warning(f"Configured LLaMA model not found; using discovered model: {discovered_model}")
            LLAMA_MODEL_PATH = discovered_model
        else:
            logger.error(f"LLaMA model not found at: {LLAMA_MODEL_PATH}")
            raise FileNotFoundError(f"LLaMA model not found: {LLAMA_MODEL_PATH}")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            pinecone_api_key=PINECONE_API_KEY,
            llama_model_path=LLAMA_MODEL_PATH,
            index_name=INDEX_NAME
        )
        logger.info("RAG Pipeline initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise
    finally:
        logger.info("Shutting down IPC RAG AI Server...")


# Initialize FastAPI app
app = FastAPI(
    title="IPC RAG AI API",
    description="AI-powered Retrieval-Augmented Generation system for Indian Penal Code queries",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Request/Response Models ===

class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=5, max_length=500, description="Legal question about IPC")
    top_k: int = Field(5, ge=1, le=20, description="Number of contexts to retrieve")
    include_contexts: bool = Field(True, description="Include retrieved contexts in response")


class ContextResponse(BaseModel):
    """Response model for retrieved context"""
    section_number: str
    section_title: str
    text: str
    score: float
    chunk_index: int


class AnswerResponse(BaseModel):
    """Response model for IPC answer"""
    question: str
    answer: str
    section_references: List[str]
    confidence: str
    metadata: Dict
    retrieved_contexts: Optional[List[ContextResponse]] = None


class SearchRequest(BaseModel):
    """Request model for pure semantic search"""
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=20)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    pipeline_ready: bool


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str


# === API Endpoints ===

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "IPC RAG AI API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/ask": "Ask IPC question (POST)",
            "/search": "Semantic search (POST)",
            "/sections/{section_number}": "Get specific section (GET)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    pipeline_ready = rag_pipeline is not None
    
    return HealthResponse(
        status="healthy" if pipeline_ready else "initializing",
        version="1.0.0",
        pipeline_ready=pipeline_ready
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a legal question about IPC using RAG pipeline
    
    - **question**: Your question about Indian Penal Code
    - **top_k**: Number of relevant sections to retrieve (1-20)
    - **include_contexts**: Include retrieved context details in response
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    
    try:
        # Process question
        logger.info(f"Processing question: {request.question}")
        
        answer = rag_pipeline.answer_question(
            question=request.question,
            top_k=request.top_k,
            return_contexts=request.include_contexts
        )
        
        # Build response
        response = AnswerResponse(
            question=answer.question,
            answer=answer.answer,
            section_references=answer.section_references,
            confidence=answer.confidence,
            metadata=answer.metadata,
            retrieved_contexts=[
                ContextResponse(
                    section_number=ctx.section_number,
                    section_title=ctx.section_title,
                    text=ctx.text,
                    score=ctx.score,
                    chunk_index=ctx.chunk_index
                )
                for ctx in answer.retrieved_contexts
            ] if request.include_contexts else None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/search", response_model=List[ContextResponse])
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search on IPC database
    
    - **query**: Search query
    - **top_k**: Number of results to return
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    
    try:
        logger.info(f"Searching: {request.query}")
        
        # Retrieve contexts
        contexts = rag_pipeline.retriever.retrieve(
            query=request.query,
            top_k=request.top_k
        )
        
        # Build response
        response = [
            ContextResponse(
                section_number=ctx.section_number,
                section_title=ctx.section_title,
                text=ctx.text,
                score=ctx.score,
                chunk_index=ctx.chunk_index
            )
            for ctx in contexts
        ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during search: {str(e)}"
        )


@app.get("/sections/{section_number}", response_model=List[ContextResponse])
async def get_section(section_number: str, top_k: int = 3):
    """
    Get information about a specific IPC section
    
    - **section_number**: IPC section number (e.g., "302", "420")
    - **top_k**: Number of results to return
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    
    try:
        query = f"Section {section_number} IPC"
        logger.info(f"Fetching section: {section_number}")
        
        contexts = rag_pipeline.retriever.retrieve(query=query, top_k=top_k)
        
        # Filter to only include the requested section
        filtered_contexts = [
            ctx for ctx in contexts 
            if ctx.section_number == section_number
        ]
        
        if not filtered_contexts:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section {section_number} not found"
            )
        
        response = [
            ContextResponse(
                section_number=ctx.section_number,
                section_title=ctx.section_title,
                text=ctx.text,
                score=ctx.score,
                chunk_index=ctx.chunk_index
            )
            for ctx in filtered_contexts
        ]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching section: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching section: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    
    try:
        # Get Pinecone index stats
        stats = rag_pipeline.retriever.index.describe_index_stats()
        
        return {
            "total_vectors": stats.get('total_vector_count', 0),
            "dimension": stats.get('dimension', 0),
            "index_fullness": stats.get('index_fullness', 0),
            "namespaces": stats.get('namespaces', {})
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching stats: {str(e)}"
        )


# === Exception Handlers ===

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# === Main Entry Point ===

def main():
    """Run the FastAPI server"""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )


if __name__ == "__main__":
    main()