import os
import time
import logging
import threading
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # very old sentence-transformers
    CrossEncoder = None

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Two-stage retrieval configuration (all optional env vars)
# ---------------------------------------------------------------------------
# NOTE: the Hugging Face id has a hyphen between "L" and "6".
CROSS_ENCODER_NAME = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Set RERANKER_ENABLED=false to restore the original single-stage behaviour.
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").strip().lower() in ("1", "true", "yes")
CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "20"))

# Cross-encoder is a process-wide singleton: loaded once, even if
# get_rag_pipeline() is called many times.
_cross_encoder = None
_cross_encoder_failed = False
_ce_lock = threading.Lock()


def get_cross_encoder():
    """Return the shared CrossEncoder, or None if it could not be loaded."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder is not None:
        return _cross_encoder
    if _cross_encoder_failed or CrossEncoder is None:
        return None
    with _ce_lock:
        if _cross_encoder is None and not _cross_encoder_failed:
            try:
                logger.info("Loading cross-encoder: %s", CROSS_ENCODER_NAME)
                _cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, max_length=512)
            except Exception:
                logger.exception("Cross-encoder failed to load; re-ranking disabled until restart")
                _cross_encoder_failed = True
    return _cross_encoder


def _normalize_match(match) -> Dict:
    meta = match.get("metadata") or {}
    return {
        "id": match.get("id"),
        "score": float(match.get("score", 0.0) or 0.0),
        "metadata": dict(meta),
    }


def _analysis_row(m: Dict, rank: int) -> Dict:
    meta = m["metadata"]
    rs = m.get("rerank_score")
    return {
        "rank": rank,
        "id": m["id"],
        "bi_encoder_score": round(m["score"], 4),
        "cross_encoder_score": None if rs is None else round(rs, 4),
        "source": meta.get("source", "Unknown"),
        "section": meta.get("section", ""),
        "text_preview": (meta.get("text") or "")[:200],
    }


class IndianLawRAG:
    def __init__(self):
        """Initialize the RAG pipeline with Pinecone and Groq"""
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = os.getenv('PINECONE_INDEX_NAME')
        self.index = self.pc.Index(self.index_name)
        
        # Initialize embedding model (Stage-1 bi-encoder)
        embedding_model_name = os.getenv(
            'EMBEDDING_MODEL',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Indian law keywords for validation
        self.indian_law_keywords = [
            'ipc', 'indian penal code', 'constitution', 'article', 'section',
            'crpc', 'criminal procedure code', 'supreme court', 'high court',
            'fundamental rights', 'directive principles', 'amendment',
            'bharatiya nyaya sanhita', 'bns','right', 'rights',
            'bharatiya nagarik suraksha sanhita', 'bnss',
            'law', 'legal', 'india', 'indian', 'act', 'rights',
            'criminal', 'civil', 'procedure', 'evidence act', 'contract act'
        ]
    
    def is_indian_law_query(self, query: str) -> bool:
        """Check if the query is related to Indian laws"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.indian_law_keywords)
    
    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text"""
        embeddings = self.embedding_model.encode(
            text,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # Two-stage retrieval
    # ------------------------------------------------------------------
    def warmup_reranker(self) -> bool:
        """Load the cross-encoder now (call once at startup)."""
        return get_cross_encoder() is not None

    def _pinecone_query(self, vector, top_k: int, namespace: Optional[str] = None,
                        metadata_filter: Optional[dict] = None):
        params = {"vector": vector, "top_k": top_k, "include_metadata": True}
        if namespace:                      # empty/None namespace = default (as before)
            params["namespace"] = namespace
        if metadata_filter:
            params["filter"] = metadata_filter
        results = self.index.query(**params)
        return results["matches"]

    def rerank_matches(self, query: str, matches: list, final_k: int) -> Tuple[List[Dict], Dict]:
        """
        Stage 2. Score every (query, chunk) pair with the cross-encoder and
        keep the best `final_k`. Never raises: on any cross-encoder problem it
        falls back to the bi-encoder (Pinecone) order.
        """
        t0 = time.perf_counter()
        cands = [_normalize_match(m) for m in matches]
        for i, c in enumerate(cands, 1):
            c["original_rank"] = i
            c["rerank_score"] = None

        analysis = {
            "bi_encoder_model": self.embedding_model_name,
            "cross_encoder_model": CROSS_ENCODER_NAME,
            "reranker_enabled": RERANKER_ENABLED,
            "candidates_retrieved": len(cands),
            "final_k": final_k,
            "initial_ranking": [_analysis_row(c, c["original_rank"]) for c in cands],
            "reranked": [],
            "final": [],
            "fallback_used": False,
            "error": None,
        }

        ordered = cands
        if cands and RERANKER_ENABLED:
            try:
                model = get_cross_encoder()
                if model is None:
                    raise RuntimeError("cross-encoder model unavailable")
                scorable = [c for c in cands if (c["metadata"].get("text") or "").strip()]
                if not scorable:
                    raise ValueError("no candidate has text to re-rank")
                scores = model.predict(
                    [(query, c["metadata"]["text"]) for c in scorable],
                    batch_size=16,
                    show_progress_bar=False,
                )
                for c, s in zip(scorable, scores):
                    c["rerank_score"] = float(s)   # raw logit, not a probability
                scorable.sort(key=lambda c: c["rerank_score"], reverse=True)
                unscorable = [c for c in cands if c["rerank_score"] is None]
                ordered = scorable + unscorable
                analysis["reranked"] = [_analysis_row(c, i) for i, c in enumerate(ordered, 1)]
            except Exception as e:
                logger.exception("Cross-encoder re-ranking failed; using bi-encoder order")
                analysis["fallback_used"] = True
                analysis["error"] = str(e)
                for c in cands:
                    c["rerank_score"] = None
                ordered = cands

        final = ordered[:final_k]
        for i, c in enumerate(final, 1):
            c["rank"] = i
        analysis["final"] = [
            {**_analysis_row(c, c["rank"]),
             "original_rank": c["original_rank"],
             "rank_change": c["original_rank"] - c["rank"]}   # positive = moved up
            for c in final
        ]
        analysis["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return final, analysis

    def retrieve_ranked(self, query: str, query_vector: List[float], top_k: int = 3,
                        namespace: Optional[str] = None,
                        metadata_filter: Optional[dict] = None,
                        candidate_k: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        """
        Stage 1 + Stage 2 for a pre-computed query vector.
        Returns (final matches, analysis). Pinecone errors propagate to the caller.
        Final matches are dicts: id, score (original Pinecone similarity),
        metadata, original_rank, rank, rerank_score.
        """
        t0 = time.perf_counter()
        fetch_k = max(candidate_k or CANDIDATE_K, top_k) if RERANKER_ENABLED else top_k
        matches = self._pinecone_query(query_vector, fetch_k, namespace, metadata_filter)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
        final, analysis = self.rerank_matches(query, matches, top_k)
        analysis["retrieval_ms"] = retrieval_ms
        return final, analysis

    @staticmethod
    def _to_context(m: Dict) -> Dict:
        meta = m["metadata"]
        return {
            'text': meta.get('text', ''),
            'score': m['score'],                      # original Pinecone similarity
            'source': meta.get('source', 'Unknown'),
            'section': meta.get('section', ''),
            # additive fields (safe for existing consumers)
            'id': m['id'],
            'rank': m['rank'],
            'original_rank': m['original_rank'],
            'rerank_score': m['rerank_score'],
        }

    def retrieve_with_analysis(self, query: str, top_k: int = 3,
                               namespace: Optional[str] = None,
                               metadata_filter: Optional[dict] = None) -> Tuple[List[Dict], Dict]:
        """Embed query -> Pinecone Top-N -> cross-encoder -> Top-k contexts + analysis."""
        query_embedding = self.get_embeddings(query)
        final, analysis = self.retrieve_ranked(
            query, query_embedding, top_k, namespace=namespace, metadata_filter=metadata_filter
        )
        return [self._to_context(m) for m in final], analysis

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant context (bi-encoder Top-20 re-ranked to Top-k)."""
        contexts, _ = self.retrieve_with_analysis(query, top_k)
        return contexts
    
    def generate_response(self, query: str, contexts: List[Dict]) -> str:
        """Generate response using Groq API with retrieved context"""
        context_text = "\n\n".join([
            f"[Source: {ctx['source']} - {ctx['section']}]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        system_prompt = """You are an expert AI assistant specialized in Indian Constitution and Laws, including IPC, CrPC, BNS, BNSS, and other Indian legal statutes.

Your responsibilities:
1. Provide accurate legal explanations
2. Cite sections/articles when relevant
3. Explain concepts clearly
4. Use only the provided legal context
"""

        user_prompt = f"""Based on the following legal context, answer the question.

CONTEXT:
{context_text}

USER QUESTION:
{query}

Answer:"""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="openai/gpt-oss-120b",
                temperature=0.3,
                max_tokens=2048,
                top_p=0.9
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, user_query: str, top_k: int = 5, include_analysis: bool = False) -> Dict:
        """Main query method for the RAG pipeline"""
        if not self.is_indian_law_query(user_query):
            return {
                'answer': (
                    "I can only answer questions related to Indian Constitution "
                    "and Laws (IPC, CrPC, BNS, BNSS, etc.)."
                ),
                'contexts': [],
                'is_valid_query': False
            }
        
        contexts, analysis = self.retrieve_with_analysis(user_query, top_k)
        
        if not contexts or all(ctx['text'] == '' for ctx in contexts):
            result = {
                'answer': (
                    "No relevant legal information found. "
                    "Please rephrase or mention specific sections/articles."
                ),
                'contexts': [],
                'is_valid_query': True
            }
            if include_analysis:
                result['retrieval_analysis'] = analysis
            return result
        
        answer = self.generate_response(user_query, contexts)
        
        result = {
            'answer': answer,
            'contexts': contexts,
            'is_valid_query': True
        }
        if include_analysis:
            result['retrieval_analysis'] = analysis
        return result


# Single shared instance: the embedding model, Pinecone and Groq clients are
# created once, even though callers invoke get_rag_pipeline() per request.
_rag_instance: Optional[IndianLawRAG] = None
_rag_lock = threading.Lock()


def get_rag_pipeline():
    """Factory function (returns a process-wide singleton)"""
    global _rag_instance
    if _rag_instance is None:
        with _rag_lock:
            if _rag_instance is None:
                _rag_instance = IndianLawRAG()
    return _rag_instance