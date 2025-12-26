import streamlit as st
from rag_pipeline import get_rag_pipeline
import time

# Page configuration
st.set_page_config(
    page_title="Indian Laws AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved readability
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1a1a1a;
        font-size: 15px;
        line-height: 1.6;
    }
    .source-box strong {
        color: #0d47a1;
        font-size: 16px;
    }
    .source-box em {
        color: #333;
        font-style: normal;
        display: block;
        margin-top: 10px;
        padding: 10px;
        background-color: #fff;
        border-radius: 6px;
        border-left: 3px solid #64b5f6;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        color: #1a1a1a;
        font-size: 16px;
        line-height: 1.8;
    }
    .answer-box strong {
        color: #1b5e20;
        font-size: 18px;
        display: block;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #fff8e1;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        border-left: 5px solid #f57c00;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        color: #1a1a1a;
        font-size: 16px;
        line-height: 1.8;
    }
    .warning-box strong {
        color: #e65100;
        font-size: 18px;
        display: block;
        margin-bottom: 15px;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #0d47a1;
        font-size: 16px;
        font-weight: 500;
        border-left: 4px solid #1976d2;
    }
    /* Improve text contrast in expanders */
    .streamlit-expanderHeader {
        font-size: 16px;
        font-weight: 600;
        color: #1a1a1a !important;
    }
    /* Better button styling */
    .stButton > button {
        font-weight: 500;
        font-size: 15px;
    }
    /* Sidebar text improvements */
    .css-1d391kg, .css-1lcbmhc {
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'rag_pipeline' not in st.session_state:
    with st.spinner("🔄 Initializing AI Assistant..."):
        st.session_state.rag_pipeline = get_rag_pipeline()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/1200px-Emblem_of_India.svg.png", width=150)
    st.title("⚖️ Indian Laws AI")
    st.markdown("---")
    
    st.markdown("""
    ### About
    This AI assistant specializes in:
    - 📜 **Indian Penal Code (IPC)**
    - 📖 **Constitution of India**
    - ⚖️ **CrPC & Evidence Act**
    - 🆕 **BNS & BNSS**
    - 📋 **Other Indian Legal Acts**
    
    ### Features
    - ✅ Accurate legal information
    - 📚 Context-based responses
    - 🔍 Source citations
    - 💡 Clear explanations
    """)
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    top_k = st.slider("Number of sources to retrieve", 3, 10, 5, 
                      help="More sources = more comprehensive but slower")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.caption("⚠️ This is an AI assistant. For legal advice, consult a qualified lawyer.")

# Main content
st.title("🏛️ Indian Constitution & Laws AI Assistant")
st.markdown("Ask me anything about Indian laws, constitutional provisions, IPC sections, and legal matters!")

# Example queries
with st.expander("💡 Example Questions"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What is Section 302 IPC?"):
            st.session_state.example_query = "What is Section 302 IPC?"
        if st.button("Explain Article 21 of Constitution"):
            st.session_state.example_query = "Explain Article 21 of Constitution"
        if st.button("What are Fundamental Rights?"):
            st.session_state.example_query = "What are Fundamental Rights?"
    
    with col2:
        if st.button("Difference between IPC and BNS"):
            st.session_state.example_query = "What is the difference between IPC and BNS?"
        if st.button("What is CrPC?"):
            st.session_state.example_query = "What is CrPC?"
        if st.button("Explain Section 377 IPC"):
            st.session_state.example_query = "Explain Section 377 IPC"

# Check if example query was clicked
if 'example_query' in st.session_state:
    query = st.session_state.example_query
    del st.session_state.example_query
else:
    query = None

# Chat interface
st.markdown("---")

# Display chat history
for i, chat in enumerate(st.session_state.chat_history):
    with st.container():
        # User message with better styling
        st.markdown(f"""
        <div class="user-message">
            🙋 <strong>You:</strong> {chat['query']}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant response
        if chat['is_valid_query']:
            # Escape HTML in answer to prevent rendering issues
            answer_text = chat['answer'].replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            
            st.markdown(f"""
            <div class="answer-box">
                <strong>🤖 AI Assistant:</strong>
                <div style="margin-top: 10px;">
                    {answer_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources
            if chat['contexts']:
                with st.expander(f"📚 View {len(chat['contexts'])} Sources", expanded=False):
                    for j, ctx in enumerate(chat['contexts']):
                        # Truncate text for display
                        display_text = ctx['text'][:400] if len(ctx['text']) > 400 else ctx['text']
                        display_text = display_text.replace('<', '&lt;').replace('>', '&gt;')
                        
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>📄 Source {j+1}</strong> <span style="color: #666;">(Relevance: {ctx['score']:.3f})</span><br>
                            <strong style="color: #1565c0;">From:</strong> <span style="color: #424242;">{ctx['source']} - {ctx['section']}</span>
                            <em>{display_text}...</em>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            answer_text = chat['answer'].replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ AI Assistant:</strong>
                <div style="margin-top: 10px;">
                    {answer_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

# Query input
query_input = st.text_input(
    "Ask your question about Indian laws:",
    value=query if query else "",
    placeholder="E.g., What is Section 420 IPC? or Explain Article 370",
    key="query_input"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    submit_button = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    if st.button("🎲 Random Example", use_container_width=True):
        examples = [
            "What is Section 302 IPC?",
            "Explain Article 21",
            "What are fundamental rights?",
            "Difference between cognizable and non-cognizable offenses",
            "What is the Right to Education?"
        ]
        import random
        st.session_state.random_query = random.choice(examples)
        st.rerun()

# Handle random query
if 'random_query' in st.session_state:
    query_input = st.session_state.random_query
    del st.session_state.random_query
    submit_button = True

# Process query
if submit_button and query_input:
    with st.spinner("🔍 Searching legal database and generating response..."):
        # Query the RAG pipeline
        result = st.session_state.rag_pipeline.query(query_input, top_k=top_k)
        
        # Add to chat history
        st.session_state.chat_history.append({
            'query': query_input,
            'answer': result['answer'],
            'contexts': result['contexts'],
            'is_valid_query': result['is_valid_query']
        })
        
        # Rerun to display the new message
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #424242; padding: 20px;'>
    <p style='font-size: 16px; font-weight: 500;'>⚖️ Indian Laws AI Assistant | Powered by Pinecone & Groq | RAG Technology</p>
    <p style='font-size: 14px; color: #666; margin-top: 10px;'>
        <strong>Disclaimer:</strong> This is an AI assistant for informational purposes only. 
        Always consult a qualified legal professional for legal advice.
    </p>
</div>
""", unsafe_allow_html=True)