"""
document_rag_service.py
Hybrid RAG for uploaded documents.

Reuses IndianLawRAG's Groq client, embedding model, and Pinecone index.
No extra clients are created – everything is borrowed from the existing pipeline.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Any

from app.services.document_processor import DocumentProcessor, ProcessedDocument
from app.utils.legal_classifier import LegalClassifier

logger = logging.getLogger(__name__)

DOC_NAMESPACE  = os.getenv("DOCUMENT_NAMESPACE", "user_documents")
LAW_NAMESPACE  = os.getenv("LAW_NAMESPACE", "")          # leave blank = no namespace (default)
TOP_K_DOC      = int(os.getenv("TOP_K_DOCUMENT", "5"))
TOP_K_LAW      = int(os.getenv("TOP_K_LAW",      "5"))
GROQ_MODEL     = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ── Prompts ──────────────────────────────────────────────────────────────────

_ANALYSIS_SYS = """You are an expert Indian constitutional and legal analyst.
Analyse the document and return ONLY a valid JSON object with these exact keys:
{
  "summary": "<2-4 sentence plain-English summary>",
  "key_legal_points": ["<point>", ...],
  "constitutional_articles": ["Article X – <name>", ...],
  "legal_references": ["<statute/section>", ...],
  "fundamental_rights_involved": ["Article X – <right>", ...]
}
No markdown, no code fences, no extra text."""

_ANALYSIS_USR = """Analyse this legal document:

--- TEXT (first 3000 chars) ---
{text}
---
Pre-detected articles : {articles}
Pre-detected statutes : {statutes}

Return the JSON analysis."""

_QA_SYS = """You are a knowledgeable Indian legal assistant specialising in:
- Constitution of India (all Articles, Schedules, Amendments)
- IPC / BNS, CrPC / BNSS, Evidence Act / BSA
- All major Indian statutes

Answer questions by combining content from the uploaded document AND Indian law knowledge.
Be precise, cite Articles/sections, and use plain language where possible."""

_QA_USR = """Question: {question}

=== FROM UPLOADED DOCUMENT ===
{doc_ctx}

=== INDIAN LAW KNOWLEDGE BASE ===
{law_ctx}

Instructions:
- Use both contexts to answer.
- Cite Constitution Articles, IPC/BNS sections, etc. where relevant.
- If Fundamental Rights or DPSPs are involved, explain how.
- If information is not in the contexts, say so clearly.

Answer:"""


class DocumentRAGService:
    """
    Parameters
    ----------
    rag : an IndianLawRAG instance (provides .index, .embedding_model, .groq_client)
    """

    def __init__(self, rag):
        self._rag      = rag
        self._index    = rag.index
        self._embedder = rag.embedding_model
        self._groq     = rag.groq_client
        self._clf      = LegalClassifier()
        self._doc_proc = DocumentProcessor(
            pinecone_index=self._index,
            embedding_model=self._embedder,
        )

    # ── upload pipeline ───────────────────────────────────────────────────

    def process_and_analyse(self, file_path: str, original_filename: str) -> dict:
        """Full upload pipeline: extract → store → analyse. Returns a rich dict."""
        processed = self._doc_proc.process(file_path, original_filename)

        if processed.error:
            return {"success": False, "error": processed.error}

        clf = processed.legal_classification
        analysis = self._analyse_llm(
            text=processed.full_text[:3000],
            articles=clf.constitutional_articles if clf else [],
            statutes=clf.statutes if clf else [],
        )

        # Merge rule-based + LLM results (LLM wins, rule-based fills gaps)
        return {
            "success":                    True,
            "document_id":                processed.document_id,
            "filename":                   processed.filename,
            "file_type":                  processed.file_type,
            "total_pages":                processed.total_pages,
            "total_chunks":               len(processed.chunks),
            "extracted_text_preview":     processed.full_text[:500],
            "summary":                    analysis.get("summary", ""),
            "key_legal_points":           analysis.get("key_legal_points", []),
            "constitutional_articles":    analysis.get("constitutional_articles") or (clf.constitutional_articles if clf else []),
            "legal_references":           analysis.get("legal_references") or (clf.statutes if clf else []),
            "fundamental_rights_involved":analysis.get("fundamental_rights_involved") or (clf.fundamental_rights if clf else []),
        }

    # ── Q&A ───────────────────────────────────────────────────────────────

    def answer_question(self, document_id: str, question: str) -> dict:
        """Hybrid RAG: doc chunks + law KB → Groq answer."""
        q_emb = self._embedder.encode([question])[0].tolist()

        doc_chunks = self._retrieve(q_emb, DOC_NAMESPACE, TOP_K_DOC,
                                    filter_meta={"document_id": document_id})
        law_chunks = self._retrieve(q_emb, LAW_NAMESPACE, TOP_K_LAW)

        doc_ctx = self._fmt_ctx(doc_chunks, "Document")
        law_ctx = self._fmt_ctx(law_chunks, "Law KB")

        answer = self._qa_llm(question, doc_ctx, law_ctx)

        ans_clf = self._clf.classify(answer)

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

        fr_analysis = None
        if ans_clf.fundamental_rights:
            fr_analysis = self._fr_analysis(question, answer, ans_clf.fundamental_rights)

        return {
            "document_id":              document_id,
            "question":                 question,
            "answer":                   answer,
            "constitutional_articles":  ans_clf.constitutional_articles,
            "legal_references":         ans_clf.statutes,
            "fundamental_rights_analysis": fr_analysis,
            "sources":                  sources,
        }

    # ── private: Pinecone ─────────────────────────────────────────────────

    def _retrieve(self, emb, namespace, top_k, filter_meta=None) -> list[dict]:
        try:
            params: dict[str, Any] = {
                "vector": emb, "top_k": top_k,
                "include_metadata": True,
            }
            if namespace:
                params["namespace"] = namespace
            if filter_meta:
                params["filter"] = filter_meta
            res = self._index.query(**params)
            return [{"id": m["id"], "score": m.get("score", 0.0),
                     "metadata": m.get("metadata", {})}
                    for m in res.get("matches", [])]
        except Exception as e:
            logger.warning("Pinecone retrieval failed (ns=%s): %s", namespace, e)
            return []

    # ── private: Groq ─────────────────────────────────────────────────────

    def _analyse_llm(self, text: str, articles: list, statutes: list) -> dict:
        msg = _ANALYSIS_USR.format(
            text=text,
            articles=", ".join(articles) or "none",
            statutes=", ".join(statutes) or "none",
        )
        try:
            resp = self._groq.chat.completions.create(
                messages=[{"role": "system", "content": _ANALYSIS_SYS},
                          {"role": "user",   "content": msg}],
                model=GROQ_MODEL, temperature=0.1, max_tokens=1500,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"): raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return {"summary": "Analysis could not be parsed.", "key_legal_points": []}
        except Exception as e:
            logger.exception("Analysis LLM error")
            return {"summary": f"Analysis error: {e}", "key_legal_points": []}

    def _qa_llm(self, question: str, doc_ctx: str, law_ctx: str) -> str:
        msg = _QA_USR.format(
            question=question,
            doc_ctx=doc_ctx or "No relevant document content found.",
            law_ctx=law_ctx or "No relevant law knowledge found.",
        )
        try:
            resp = self._groq.chat.completions.create(
                messages=[{"role": "system", "content": _QA_SYS},
                          {"role": "user",   "content": msg}],
                model=GROQ_MODEL, temperature=0.2, max_tokens=2000,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("Q&A LLM error")
            return f"Error generating answer: {e}"

    def _fr_analysis(self, question: str, answer: str, frs: list) -> str:
        try:
            p = (f"Based on this Q&A about an Indian legal document, give a 2-3 sentence "
                 f"analysis of how these Fundamental Rights ({', '.join(frs)}) are relevant.\n\n"
                 f"Question: {question}\nAnswer: {answer}")
            resp = self._groq.chat.completions.create(
                messages=[{"role": "user", "content": p}],
                model=GROQ_MODEL, temperature=0.1, max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    @staticmethod
    def _fmt_ctx(chunks: list[dict], label: str) -> str:
        return "\n\n".join(
            f"[{label} Chunk {i+1} | score={c.get('score',0):.3f}]\n"
            f"{c['metadata'].get('text','')}"
            for i, c in enumerate(chunks)
        )