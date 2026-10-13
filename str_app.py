"""
str_app.py  –  Indian Laws AI Assistant (Streamlit)
Fixes:
  1. Removed broken Wikipedia image → replaced with inline SVG Ashoka Chakra emblem
  2. Added Groq streaming → answers appear token-by-token, no more waiting
"""

import os
import sys
import tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from rag_pipeline import get_rag_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Laws AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { padding: 2rem; }
.stTextInput > div > div > input { font-size: 16px; }

.source-box {
    background-color:#f8f9fa; padding:20px; border-radius:12px; margin:15px 0;
    border-left:5px solid #1f77b4; box-shadow:0 2px 4px rgba(0,0,0,.1);
    color:#1a1a1a; font-size:15px; line-height:1.6;
}
.source-box strong { color:#0d47a1; font-size:16px; }
.source-box em {
    color:#333; font-style:normal; display:block; margin-top:10px;
    padding:10px; background:#fff; border-radius:6px; border-left:3px solid #64b5f6;
}
.answer-box {
    background-color:#e8f5e9; padding:25px; border-radius:12px; margin:20px 0;
    border-left:5px solid #2e7d32; box-shadow:0 2px 6px rgba(0,0,0,.1);
    color:#1a1a1a; font-size:16px; line-height:1.8;
}
.answer-box strong { color:#1b5e20; font-size:18px; display:block; margin-bottom:15px; }
.warning-box {
    background-color:#fff8e1; padding:25px; border-radius:12px; margin:20px 0;
    border-left:5px solid #f57c00; box-shadow:0 2px 6px rgba(0,0,0,.1);
    color:#1a1a1a; font-size:16px; line-height:1.8;
}
.warning-box strong { color:#e65100; font-size:18px; display:block; margin-bottom:15px; }
.user-message {
    background-color:#e3f2fd; padding:15px 20px; border-radius:10px; margin:10px 0;
    color:#0d47a1; font-size:16px; font-weight:500; border-left:4px solid #1976d2;
}
.doc-analysis-box {
    background:#f3e5f5; padding:20px; border-radius:12px; margin:15px 0;
    border-left:5px solid #7b1fa2; color:#1a1a1a; font-size:15px; line-height:1.7;
}
.doc-analysis-box strong { color:#4a148c; }
.tag-pill {
    display:inline-block; background:#e8eaf6; color:#283593; padding:3px 10px;
    border-radius:20px; margin:3px; font-size:13px; font-weight:500;
}
.emblem-container {
    display:flex; justify-content:center; margin-bottom:12px;
}
</style>
""", unsafe_allow_html=True)

# ── Ashoka Chakra SVG (no external URL needed) ────────────────────────────────
EMBLEM_SVG = """
<div class="emblem-container">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="110" height="110">
  <!-- Outer ring -->
  <circle cx="100" cy="100" r="90" fill="none" stroke="#1a237e" stroke-width="6"/>
  <!-- Ashoka Chakra - 24 spokes -->
  <circle cx="100" cy="100" r="60" fill="none" stroke="#1565c0" stroke-width="3"/>
  <circle cx="100" cy="100" r="10" fill="#1565c0"/>
  <!-- 24 spokes -->
  <g stroke="#1565c0" stroke-width="2.2" transform="translate(100,100)">
    <line x1="0" y1="-10" x2="0" y2="-58"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(15)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(30)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(45)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(60)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(75)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(90)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(105)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(120)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(135)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(150)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(165)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(180)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(195)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(210)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(225)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(240)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(255)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(270)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(285)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(300)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(315)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(330)"/>
    <line x1="0" y1="-10" x2="0" y2="-58" transform="rotate(345)"/>
  </g>
  <!-- Lions silhouette (simplified) -->
  <text x="100" y="155" text-anchor="middle" font-size="44" fill="#1a237e">🦁</text>
  <!-- Satyamev Jayate text -->
  <text x="100" y="185" text-anchor="middle" font-size="11"
        fill="#1a237e" font-family="serif" font-weight="bold"
        letter-spacing="1">सत्यमेव जयते</text>
</svg>
</div>
"""

# ── Session-state init ────────────────────────────────────────────────────────
def _init():
    defaults = {
        "chat_history":     [],
        "doc_chat_history": [],
        "uploaded_doc_id":  None,
        "upload_result":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Load pipeline (cached once) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading AI models…")
def load_pipeline():
    rag = get_rag_pipeline()
    from app.services.document_rag_service import DocumentRAGService
    return rag, DocumentRAGService(rag=rag)

rag_pipeline, doc_rag_service = load_pipeline()


# ── Streaming helpers ─────────────────────────────────────────────────────────
def _stream_rag_answer(query: str, top_k: int):
    """
    Run Pinecone retrieval (same as IndianLawRAG.query) then stream the
    Groq completion token-by-token.  Returns (streamed_text, contexts, is_valid).
    """
    if not rag_pipeline.is_indian_law_query(query):
        return (
            "I can only answer questions related to Indian Constitution and Laws "
            "(IPC, CrPC, BNS, BNSS, etc.).",
            [],
            False,
        )

    contexts = rag_pipeline.retrieve_context(query, top_k)
    if not contexts or all(c["text"] == "" for c in contexts):
        return (
            "No relevant legal information found. "
            "Please rephrase or mention specific sections/articles.",
            [],
            True,
        )

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

    # Stream from Groq
    stream = rag_pipeline.groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=2048,
        top_p=0.9,
        stream=True,          # ← KEY CHANGE
    )

    placeholder = st.empty()
    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_text += delta
        placeholder.markdown(
            f'<div class="answer-box"><strong>🤖 AI Assistant:</strong>'
            f'<div style="margin-top:10px">{full_text}▌</div></div>',
            unsafe_allow_html=True,
        )
    # Final render without cursor
    placeholder.markdown(
        f'<div class="answer-box"><strong>🤖 AI Assistant:</strong>'
        f'<div style="margin-top:10px">{full_text}</div></div>',
        unsafe_allow_html=True,
    )
    return full_text, contexts, True


def _stream_doc_answer(doc_id: str, question: str):
    """
    Hybrid RAG for document Q&A with streaming.
    Retrieval is done first, then Groq streams the answer.
    """
    from app.services.document_rag_service import (
        DOC_NAMESPACE, LAW_NAMESPACE, TOP_K_DOC, TOP_K_LAW,
        GROQ_MODEL, _QA_SYS, _QA_USR,
    )
    from app.utils.legal_classifier import LegalClassifier

    emb = rag_pipeline.embedding_model.encode([question])[0].tolist()

    def _retrieve(namespace, top_k, flt=None):
        try:
            params = {"vector": emb, "top_k": top_k, "include_metadata": True}
            if namespace:
                params["namespace"] = namespace
            if flt:
                params["filter"] = flt
            res = rag_pipeline.index.query(**params)
            return [{"id": m["id"], "score": m.get("score", 0),
                     "metadata": m.get("metadata", {})}
                    for m in res.get("matches", [])]
        except Exception:
            return []

    doc_chunks = _retrieve(DOC_NAMESPACE, TOP_K_DOC, {"document_id": doc_id})
    law_chunks = _retrieve(LAW_NAMESPACE, TOP_K_LAW)

    def fmt(chunks, label):
        return "\n\n".join(
            f"[{label} Chunk {i+1} | score={c.get('score',0):.3f}]\n"
            f"{c['metadata'].get('text','')}"
            for i, c in enumerate(chunks)
        )

    msg = _QA_USR.format(
        question=question,
        doc_ctx=fmt(doc_chunks, "Document") or "No relevant document content found.",
        law_ctx=fmt(law_chunks, "Law KB")   or "No relevant law knowledge found.",
    )

    stream = rag_pipeline.groq_client.chat.completions.create(
        messages=[{"role": "system", "content": _QA_SYS},
                  {"role": "user",   "content": msg}],
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=2000,
        stream=True,
    )

    placeholder = st.empty()
    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_text += delta
        placeholder.markdown(
            f'<div class="answer-box"><strong>🤖 AI Assistant:</strong>'
            f'<div style="margin-top:10px">{full_text}▌</div></div>',
            unsafe_allow_html=True,
        )
    placeholder.markdown(
        f'<div class="answer-box"><strong>🤖 AI Assistant:</strong>'
        f'<div style="margin-top:10px">{full_text}</div></div>',
        unsafe_allow_html=True,
    )

    clf = LegalClassifier()
    ans_clf = clf.classify(full_text)

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

    return {
        "question":                   question,
        "answer":                     full_text,
        "constitutional_articles":    ans_clf.constitutional_articles,
        "legal_references":           ans_clf.statutes,
        "fundamental_rights_analysis": None,
        "sources":                    sources,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(EMBLEM_SVG, unsafe_allow_html=True)   # ← inline SVG, no network call
    st.title("⚖️ Indian Laws AI")
    st.markdown("---")
    st.markdown("""
### About
- 📜 **IPC / BNS**
- 📖 **Constitution of India**
- ⚖️ **CrPC / BNSS / Evidence**
- 📋 **Other Indian Statutes**
- 📄 **Document Analysis (NEW)**

### Features
- ✅ Accurate legal answers
- 📚 Source citations
- ⚡ Streamed responses
- 📤 PDF & Image upload
""")
    st.markdown("---")
    top_k = st.slider("Sources to retrieve", 3, 10, 5)
    st.markdown("---")
    if st.button("🗑️ Clear All History", use_container_width=True):
        st.session_state.chat_history     = []
        st.session_state.doc_chat_history = []
        st.session_state.uploaded_doc_id  = None
        st.session_state.upload_result    = None
        st.rerun()
    st.markdown("---")
    st.caption("⚠️ AI assistant only. Consult a qualified lawyer for legal advice.")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_doc = st.tabs(["💬 Law Q&A", "📄 Document Analysis"])


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 1 – Law Q&A with streaming
# ════════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.title("🏛️ Indian Constitution & Laws AI Assistant")
    st.markdown("Ask anything about Indian laws, constitutional provisions, IPC sections, and legal matters!")

    with st.expander("💡 Example Questions"):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("What is Section 302 IPC?",        key="ex1"): st.session_state.eq = "What is Section 302 IPC?"
            if st.button("Explain Article 21",               key="ex2"): st.session_state.eq = "Explain Article 21 of Constitution"
            if st.button("What are Fundamental Rights?",     key="ex3"): st.session_state.eq = "What are Fundamental Rights?"
        with c2:
            if st.button("Difference between IPC and BNS",  key="ex4"): st.session_state.eq = "What is the difference between IPC and BNS?"
            if st.button("What is CrPC?",                   key="ex5"): st.session_state.eq = "What is CrPC?"
            if st.button("Explain Section 377 IPC",         key="ex6"): st.session_state.eq = "Explain Section 377 IPC"

    preset = st.session_state.pop("eq", None)
    st.markdown("---")

    # Render existing chat history (already streamed, stored as full text)
    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="user-message">🙋 <strong>You:</strong> {chat["query"]}</div>',
                    unsafe_allow_html=True)
        if chat["is_valid_query"]:
            ans = chat["answer"].replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
            st.markdown(f'<div class="answer-box"><strong>🤖 AI Assistant:</strong>'
                        f'<div style="margin-top:10px">{ans}</div></div>',
                        unsafe_allow_html=True)
            if chat["contexts"]:
                with st.expander(f"📚 View {len(chat['contexts'])} Sources"):
                    for j, ctx in enumerate(chat["contexts"]):
                        txt = ctx["text"][:400].replace("<","&lt;").replace(">","&gt;")
                        st.markdown(f"""<div class="source-box">
                            <strong>📄 Source {j+1}</strong>
                            <span style="color:#666"> (Relevance: {ctx['score']:.3f})</span><br>
                            <strong style="color:#1565c0">From:</strong>
                            <span style="color:#424242"> {ctx['source']} – {ctx['section']}</span>
                            <em>{txt}…</em></div>""", unsafe_allow_html=True)
        else:
            ans = chat["answer"].replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
            st.markdown(f'<div class="warning-box"><strong>⚠️ AI Assistant:</strong>'
                        f'<div style="margin-top:10px">{ans}</div></div>',
                        unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

    query_input = st.text_input(
        "Ask your question about Indian laws:",
        value=preset or "",
        placeholder="E.g., What is Section 420 IPC?",
        key="chat_input",
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🔍 Ask", type="primary", use_container_width=True)

    if submit and query_input:
        st.markdown(f'<div class="user-message">🙋 <strong>You:</strong> {query_input}</div>',
                    unsafe_allow_html=True)
        # Stream the answer live
        answer, contexts, is_valid = _stream_rag_answer(query_input, top_k)

        # Show sources after streaming
        if is_valid and contexts:
            with st.expander(f"📚 View {len(contexts)} Sources"):
                for j, ctx in enumerate(contexts):
                    txt = ctx["text"][:400].replace("<","&lt;").replace(">","&gt;")
                    st.markdown(f"""<div class="source-box">
                        <strong>📄 Source {j+1}</strong>
                        <span style="color:#666"> (Relevance: {ctx['score']:.3f})</span><br>
                        <strong style="color:#1565c0">From:</strong>
                        <span style="color:#424242"> {ctx['source']} – {ctx['section']}</span>
                        <em>{txt}…</em></div>""", unsafe_allow_html=True)

        # Save to history
        st.session_state.chat_history.append({
            "query":          query_input,
            "answer":         answer,
            "contexts":       contexts,
            "is_valid_query": is_valid,
        })


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 2 – Document Analysis with streaming Q&A
# ════════════════════════════════════════════════════════════════════════════════
with tab_doc:
    st.title("📄 Document Analysis & Legal Assistance")
    st.markdown("Upload a **PDF or image** (court judgment, government notice, contract, etc.) "
                "and get instant AI-powered legal analysis.")

    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or image file",
        type=["pdf", "jpg", "jpeg", "png", "tiff", "webp"],
        help="Max 20 MB. Supports scanned documents via OCR.",
        key="doc_uploader",
    )

    MAX_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > MAX_SIZE_MB:
            st.error(f"❌ File too large ({file_size_mb:.1f} MB). Maximum is {MAX_SIZE_MB} MB.")
        else:
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.info(f"📎 **{uploaded_file.name}**  |  {file_size_mb:.2f} MB  |  {uploaded_file.type}")
            with col_btn:
                process_btn = st.button("🚀 Process Document", type="primary", use_container_width=True)

            if process_btn:
                with st.spinner("⏳ Extracting text, generating embeddings, and analysing…"):
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        result = doc_rag_service.process_and_analyse(
                            file_path=tmp_path,
                            original_filename=uploaded_file.name,
                        )
                    finally:
                        try: os.remove(tmp_path)
                        except OSError: pass

                if not result["success"]:
                    st.error(f"❌ Processing failed: {result['error']}")
                else:
                    st.session_state.uploaded_doc_id  = result["document_id"]
                    st.session_state.upload_result    = result
                    st.session_state.doc_chat_history = []
                    st.success(
                        f"✅ Document processed!  ID: `{result['document_id']}`  |  "
                        f"{result['total_pages']} page(s)  |  {result['total_chunks']} chunks stored"
                    )

    # ── Analysis panel ────────────────────────────────────────────────────
    if st.session_state.upload_result:
        res = st.session_state.upload_result
        st.markdown("---")
        st.markdown("### 📊 Document Analysis")

        st.markdown("#### 📝 Summary")
        st.markdown(f'<div class="doc-analysis-box">{res["summary"]}</div>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### ⚖️ Key Legal Points")
            for pt in res["key_legal_points"] or ["None identified"]:
                st.markdown(f"• {pt}")
        with col_b:
            st.markdown("#### 🛡️ Fundamental Rights Involved")
            if res["fundamental_rights_involved"]:
                for fr in res["fundamental_rights_involved"]:
                    st.markdown(f'<span class="tag-pill">⚖️ {fr}</span>', unsafe_allow_html=True)
            else:
                st.caption("None identified")

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("#### 📜 Constitutional Articles")
            if res["constitutional_articles"]:
                for art in res["constitutional_articles"]:
                    st.markdown(f'<span class="tag-pill">📌 {art}</span>', unsafe_allow_html=True)
            else:
                st.caption("None identified")
        with col_d:
            st.markdown("#### 📋 Legal References")
            if res["legal_references"]:
                for ref in res["legal_references"]:
                    st.markdown(f'<span class="tag-pill">📖 {ref}</span>', unsafe_allow_html=True)
            else:
                st.caption("None identified")

        with st.expander("🔍 Extracted Text Preview"):
            st.text_area("", value=res["extracted_text_preview"], height=200, disabled=True)

        # ── Doc Q&A with streaming ────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💬 Ask Questions About This Document")
        st.caption(f"Document ID: `{st.session_state.uploaded_doc_id}`")

        with st.expander("💡 Example Document Questions"):
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("Summarize this document",              key="dq1"): st.session_state.dq = "Summarize this document in detail."
                if st.button("What are the important points?",       key="dq2"): st.session_state.dq = "What are the most important legal points in this document?"
                if st.button("Which constitutional articles apply?", key="dq3"): st.session_state.dq = "Which constitutional articles are relevant to this document?"
            with dc2:
                if st.button("Does this affect fundamental rights?", key="dq4"): st.session_state.dq = "Does this document affect any fundamental rights?"
                if st.button("What legal actions are mentioned?",    key="dq5"): st.session_state.dq = "What legal actions or remedies are mentioned in this document?"
                if st.button("Explain in simple language",           key="dq6"): st.session_state.dq = "Explain the key clauses of this document in simple language."

        preset_dq = st.session_state.pop("dq", None)

        # Render doc chat history
        for dc in st.session_state.doc_chat_history:
            st.markdown(f'<div class="user-message">🙋 <strong>You:</strong> {dc["question"]}</div>',
                        unsafe_allow_html=True)
            ans = dc["answer"].replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
            st.markdown(f'<div class="answer-box"><strong>🤖 AI Assistant:</strong>'
                        f'<div style="margin-top:10px">{ans}</div></div>',
                        unsafe_allow_html=True)
            if dc.get("constitutional_articles") or dc.get("legal_references"):
                with st.expander("📌 Legal References Found in Answer"):
                    for art in dc.get("constitutional_articles", []):
                        st.markdown(f'<span class="tag-pill">📌 {art}</span>', unsafe_allow_html=True)
                    for ref in dc.get("legal_references", []):
                        st.markdown(f'<span class="tag-pill">📖 {ref}</span>', unsafe_allow_html=True)
            if dc.get("sources"):
                with st.expander(f"📚 View {len(dc['sources'])} Sources"):
                    for src in dc["sources"]:
                        icon  = "📄" if src["source"] == "document" else "📚"
                        label = "Uploaded Document" if src["source"] == "document" else "Law Knowledge Base"
                        st.markdown(f"""<div class="source-box">
                            <strong>{icon} {label}</strong>
                            <span style="color:#666"> (score: {src['score']:.3f})</span>
                            <em>{src['text_preview']}…</em></div>""",
                            unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")

        doc_question = st.text_input(
            "Ask a question about the uploaded document:",
            value=preset_dq or "",
            placeholder="E.g., Does this notice violate Article 21?",
            key="doc_question_input",
        )
        ask_doc_btn = st.button("🔍 Ask About Document", type="primary", key="ask_doc_btn")

        if ask_doc_btn and doc_question and st.session_state.uploaded_doc_id:
            st.markdown(f'<div class="user-message">🙋 <strong>You:</strong> {doc_question}</div>',
                        unsafe_allow_html=True)
            # Stream the answer
            qa_result = _stream_doc_answer(st.session_state.uploaded_doc_id, doc_question)

            # Sources after streaming
            if qa_result.get("sources"):
                with st.expander(f"📚 View {len(qa_result['sources'])} Sources"):
                    for src in qa_result["sources"]:
                        icon  = "📄" if src["source"] == "document" else "📚"
                        label = "Uploaded Document" if src["source"] == "document" else "Law Knowledge Base"
                        st.markdown(f"""<div class="source-box">
                            <strong>{icon} {label}</strong>
                            <span style="color:#666"> (score: {src['score']:.3f})</span>
                            <em>{src['text_preview']}…</em></div>""",
                            unsafe_allow_html=True)

            st.session_state.doc_chat_history.append(qa_result)

        elif ask_doc_btn and not st.session_state.uploaded_doc_id:
            st.warning("⚠️ Please upload and process a document first.")

    else:
        st.markdown("---")
        st.info("""
**How to use Document Analysis:**

1. 📤 **Upload** a PDF or image (court judgment, government notice, contract, etc.)
2. 🚀 **Click "Process Document"** — text is extracted and stored in Pinecone
3. 📊 **View the auto-generated analysis** — summary, key legal points, constitutional articles
4. 💬 **Ask questions** about your document using Hybrid RAG (streamed answers)
        """)
        with st.expander("📋 Supported Document Types"):
            st.markdown("""
| Type | Examples |
|------|----------|
| 📄 PDF | Court judgments, government notifications, legal contracts, FIR copies |
| 🖼️ Image | Scanned documents, photos of notices, handwritten legal papers |
| 📑 Scanned PDFs | Automatically handled via EasyOCR fallback |
            """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#424242; padding:20px;'>
    <p style='font-size:16px; font-weight:500;'>
        ⚖️ Indian Laws AI Assistant | Pinecone + Groq + EasyOCR | RAG Technology
    </p>
    <p style='font-size:14px; color:#666; margin-top:10px;'>
        <strong>Disclaimer:</strong> For informational purposes only.
        Always consult a qualified legal professional for legal advice.
    </p>
</div>
""", unsafe_allow_html=True)