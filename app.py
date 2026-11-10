"""
app.py
------
RAG-Based Multilingual University Helpdesk
Clean, Minimal, Modern AI Chatbot Interface (Inspired by ChatGPT & Linear)

Features:
- Minimal, distraction-free conversational layout
- Clean white/light theme with subtle borders and refined typography
- Gemini-powered natural-language synthesis with transparent citations
- Local extractive fallback when Gemini API is unconfigured or rate-limited
- Expandable supporting evidence with source and page preservation
- BM25 + FAISS hybrid multilingual retrieval (English, Hindi, Telugu, Punjabi)
- Conversational memory for contextual follow-up inquiries
- Compact sidebar for document management and language settings
"""

import os
import html
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

# Load .env before importing modules that read GEMINI_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Seamlessly interoperate with Streamlit Community Cloud st.secrets
try:
    if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            os.environ["GEMINI_API_KEY"] = str(st.secrets["GEMINI_API_KEY"]).strip()
    if "GEMINI_MODEL" not in os.environ or not os.environ["GEMINI_MODEL"]:
        if hasattr(st, "secrets") and "GEMINI_MODEL" in st.secrets:
            os.environ["GEMINI_MODEL"] = str(st.secrets["GEMINI_MODEL"]).strip()
except Exception:
    pass

from utils.document_processor import (
    process_document,
    load_or_build_document_index,
    compute_content_hash,
    scan_all_documents
)
from utils.text_search import BM25SearchEngine
from utils.vector_search import VectorSearchEngine
from utils.answer_generator import AnswerGenerator, format_citation, detect_language, test_gemini_connection
from utils.query_translator import expand_query_multilingual
from utils.context_rewriter import rewrite_conversational_query, get_recent_conversation_topic
from utils.category_manager import (
    CATEGORIES,
    ALL_CATEGORIES,
    CATEGORY_ICONS,
    classify_document,
    classify_question
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="University Helpdesk",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Clean, Minimal & Modern CSS (ChatGPT & Linear Inspired)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Clean Font & Layout without horizontal overflow */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #111827;
        background-color: #ffffff;
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    /* Clean container spacing */
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 4rem !important;
        max-width: 860px !important;
        width: 100% !important;
        margin: 0 auto !important;
        overflow-x: hidden !important;
        box-sizing: border-box !important;
        min-width: 0 !important;
    }

    /* Top Compact Header Bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0 1.25rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #f3f4f6;
        width: 100%;
        box-sizing: border-box;
    }
    .top-navbar-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .top-navbar-sub {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 400;
        margin-left: 0.5rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.80rem;
        font-weight: 500;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
    }
    .status-pill-green {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .status-pill-amber {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot-green { background-color: #16a34a; }
    .status-dot-amber { background-color: #d97706; }

    /* Streamlit Chat Message Native Full-Width Overrides */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 0.85rem 0 !important;
        max-width: 100% !important;
        min-width: 0 !important;
        overflow-x: hidden !important;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        min-width: 0 !important;
        max-width: 100% !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li {
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }

    /* Assistant Metadata Bar */
    .assistant-meta-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.5rem;
    }
    .chat-badge {
        font-size: 0.72rem;
        font-weight: 500;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    .chat-badge-gemini {
        background-color: #faf5ff;
        color: #6b21a8;
        border: 1px solid #e9d5ff;
    }
    .chat-badge-fallback {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .chat-badge-category {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }
    .chat-citation-bar {
        font-size: 0.84rem;
        color: #374151;
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
        padding: 0.35rem 0.7rem;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        display: inline-block;
        max-width: 100%;
        overflow-wrap: anywhere;
    }
    .fallback-note {
        font-size: 0.78rem;
        color: #9a3412;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 6px;
        padding: 0.35rem 0.65rem;
        margin-top: 0.4rem;
        max-width: 100%;
        overflow-wrap: anywhere;
    }

    /* Result Card for Evidence Inspector & Test Compatibility */
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        max-width: 100%;
        box-sizing: border-box;
        overflow-wrap: anywhere;
    }
    .result-card-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .source-badge {
        font-size: 0.78rem;
        font-weight: 600;
        color: #1e40af;
        background-color: #eff6ff;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        border: 1px solid #dbeafe;
    }
    .page-badge {
        font-size: 0.78rem;
        font-weight: 500;
        color: #374151;
        background-color: #f3f4f6;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        border: 1px solid #e5e7eb;
    }
    .scope-badge {
        background-color: #eef2ff !important;
        color: #3730a3 !important;
        border-color: #c7d2fe !important;
    }
    .score-badge {
        font-size: 0.78rem;
        font-weight: 500;
        color: #047857;
        background-color: #ecfdf5;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        border: 1px solid #a7f3d0;
    }
    .chunk-text {
        font-size: 0.88rem;
        line-height: 1.6;
        color: #334155;
        background-color: #f8fafc;
        padding: 0.65rem 0.85rem;
        border-radius: 6px;
        border: 1px solid #f1f5f9;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        word-break: break-word;
        max-width: 100%;
        box-sizing: border-box;
    }

    /* Minimal Button Styling */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 500 !important;
        font-size: 0.90rem !important;
        border: 1px solid #e5e7eb !important;
        background-color: #ffffff !important;
        color: #1f2937 !important;
        padding: 0.55rem 0.9rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton > button:hover {
        border-color: #cbd5e1 !important;
        background-color: #f8fafc !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
    }

    /* ChatGPT-style Pill Input Form */
    div[data-testid="stForm"] {
        border: 1px solid #e5e7eb !important;
        border-radius: 24px !important;
        padding: 0.35rem 0.6rem !important;
        background-color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stForm"]:focus-within {
        border-color: #10a37f !important;
        box-shadow: 0 4px 14px rgba(16, 163, 127, 0.15) !important;
    }
    div[data-testid="stForm"] div[data-testid="stTextInput"] input {
        border: none !important;
        background: transparent !important;
        padding-left: 0.8rem !important;
        font-size: 0.96rem !important;
        color: #111827 !important;
    }
    div[data-testid="stForm"] div[data-testid="stTextInput"] input:focus {
        border: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stForm"] button[kind="primary"] {
        border-radius: 9999px !important;
        background-color: #10a37f !important;
        border-color: #10a37f !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 0.45rem 1rem !important;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover {
        background-color: #0e8f6e !important;
        border-color: #0e8f6e !important;
    }

    /* Welcome Hero Container (ChatGPT Style) */
    .welcome-container {
        text-align: center;
        padding: 3rem 1rem 1.8rem 1rem;
        max-width: 680px;
        margin: 0 auto;
    }
    .welcome-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background-color: #ecfdf5;
        color: #10a37f;
        font-size: 1.6rem;
        margin-bottom: 1.25rem;
        border: 1px solid #a7f3d0;
        box-shadow: 0 2px 6px rgba(16, 163, 127, 0.12);
    }
    .welcome-title {
        font-size: 1.85rem;
        font-weight: 600;
        color: #111827;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    .welcome-subtitle {
        font-size: 1.0rem;
        color: #6b7280;
        line-height: 1.5;
        margin-bottom: 2rem;
    }

    /* Subtle footer disclaimer */
    .chat-disclaimer {
        text-align: center;
        font-size: 0.76rem;
        color: #9ca3af;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# State Initialization (Persistent Across Runs)
# ---------------------------------------------------------
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = []

if "search_engine" not in st.session_state:
    st.session_state.search_engine = BM25SearchEngine()

if "vector_engine" not in st.session_state:
    st.session_state.vector_engine = VectorSearchEngine()

if "answer_generator" not in st.session_state:
    st.session_state.answer_generator = AnswerGenerator(confidence_threshold=0.25)

if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = set()

if "search_input" not in st.session_state:
    st.session_state.search_input = ""

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "last_chat_query" not in st.session_state:
    st.session_state.last_chat_query = ""

if "active_language" not in st.session_state:
    st.session_state.active_language = "Auto-detect"

if "search_mode_option" not in st.session_state:
    st.session_state.search_mode_option = "Hybrid (Semantic + BM25)"

if "doc_categories" not in st.session_state:
    st.session_state.doc_categories = {}

if "selected_category" not in st.session_state:
    st.session_state.selected_category = ALL_CATEGORIES


# ---------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------
def select_example_query(query_text: str):
    """Callback triggered when an example question button is clicked."""
    st.session_state.search_input = query_text
    st.session_state.pending_query = query_text


def clear_search_query():
    """Callback to clear the search input box."""
    st.session_state.search_input = ""
    st.session_state.pending_query = ""


def clear_conversation():
    """Clear conversation history and reset query state."""
    st.session_state.conversation_history = []
    st.session_state.search_input = ""
    st.session_state.pending_query = ""
    st.session_state.last_chat_query = ""


def load_sample_documents(force_reindex: bool = False):
    """Loads and persists all university documents (DOCX, PDF, TXT) recursively from sample_documents/docs_raw."""
    samples_dir = Path("sample_documents/docs_raw")
    if not samples_dir.exists():
        samples_dir = Path("sample_documents")
    if not samples_dir.exists():
        st.warning("University policy documents directory not found.")
        return

    with st.spinner("Loading university policy knowledge base & semantic index..."):
        all_chunks, doc_categories = load_or_build_document_index(
            samples_dir=samples_dir,
            vector_engine=st.session_state.vector_engine,
            search_engine=st.session_state.search_engine,
            force_reindex=force_reindex
        )
        st.session_state.indexed_chunks = all_chunks
        st.session_state.doc_categories.update(doc_categories)
        for c in all_chunks:
            st.session_state.loaded_files.add(c["source"])

    st.success(f"Loaded {len(st.session_state.loaded_files)} genuine policy documents ({len(st.session_state.indexed_chunks)} passages).")


def remove_document(filename: str):
    """Remove a specific document from the active index and re-index."""
    st.session_state.indexed_chunks = [
        c for c in st.session_state.indexed_chunks if c["source"] != filename
    ]
    st.session_state.loaded_files.discard(filename)
    st.session_state.doc_categories.pop(filename, None)
    st.session_state.search_engine.index_chunks(st.session_state.indexed_chunks)
    if st.session_state.indexed_chunks:
        with st.spinner(f"Updating index without {filename}..."):
            st.session_state.vector_engine.index_chunks(st.session_state.indexed_chunks)
    else:
        st.session_state.vector_engine = VectorSearchEngine()
    st.success(f"Removed {filename}.")


def format_chunk_display_html(raw_text: str) -> str:
    """Formats chunk text, cleanly converting markdown table blocks into styled HTML tables."""
    lines = raw_text.splitlines()
    html_parts = []
    in_table = False
    table_rows = []

    def flush_table(rows):
        if not rows:
            return ""
        t_html = ['<table class="evidence-table" style="width:100%; border-collapse:collapse; margin:8px 0; font-size:0.82rem;">']
        is_header = True
        for row in rows:
            cols = [c.strip() for c in row.strip().strip('|').split('|')]
            if all(set(c).issubset({'-', ':', ' '}) for c in cols if c):
                is_header = False
                continue
            t_html.append('<tr>')
            for col in cols:
                tag = 'th' if is_header else 'td'
                bg = 'background:#f8fafc; font-weight:600;' if is_header else 'background:#ffffff;'
                t_html.append(f'<{tag} style="border:1px solid #cbd5e1; padding:4px 8px; {bg}">{html.escape(col)}</{tag}>')
            t_html.append('</tr>')
            is_header = False
        t_html.append('</table>')
        return "".join(t_html)

    for line in lines:
        s_line = line.strip()
        if s_line.startswith('|') and s_line.endswith('|'):
            in_table = True
            table_rows.append(s_line)
        else:
            if in_table:
                html_parts.append(flush_table(table_rows))
                table_rows = []
                in_table = False
            if s_line:
                html_parts.append(html.escape(s_line))

    if in_table:
        html_parts.append(flush_table(table_rows))

    return "<br>".join(html_parts)


def execute_query(query_text: str, conf_threshold: float, search_mode: str, category_filter: Optional[str] = None):
    """Executes search retrieval, grounded answer generation, and updates conversation history."""
    student_query = query_text.strip()
    if not student_query:
        return

    # Ensure documents are loaded
    if len(st.session_state.indexed_chunks) == 0:
        load_sample_documents()

    # 1. Feature 1: Context-Aware Conversational Query Rewriting
    history_turns = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.conversation_history[-8:]
        if m.get("role") in ("user", "assistant")
    ]

    search_query, was_rewritten, rewrite_method = rewrite_conversational_query(
        student_query,
        history_turns,
        api_key=st.session_state.answer_generator.api_key
    )

    # 2. Feature 2: Category Detection & Filtering
    active_topic, _ = get_recent_conversation_topic(history_turns) if was_rewritten else (None, None)
    detected_cat, cat_conf = classify_question(search_query, context_topic=active_topic)

    # Determine active category filter
    user_chosen_cat = category_filter or st.session_state.get("selected_category", ALL_CATEGORIES)
    if user_chosen_cat and user_chosen_cat != ALL_CATEGORIES:
        retrieval_category = user_chosen_cat
        final_category_tag = user_chosen_cat
    elif detected_cat and cat_conf >= 0.65:
        retrieval_category = detected_cat
        final_category_tag = detected_cat
    else:
        # Fall back to All Categories for uncertain classification
        retrieval_category = None
        final_category_tag = detected_cat or "General / Other"

    # Multilingual detection & expansion
    lang_info = detect_language(search_query)
    trans_res = expand_query_multilingual(search_query, lang_info["code"])
    translated_q = trans_res["english_query"] if trans_res["is_translated"] else None

    # Retrieval based on search mode with category filtering
    if "Hybrid" in search_mode:
        bm25_r = st.session_state.search_engine.search(
            search_query,
            top_k=35,
            translated_query=translated_q,
            category=retrieval_category
        )
        results = st.session_state.vector_engine.hybrid_search(
            search_query,
            bm25_r,
            top_k=5,
            translated_query=translated_q,
            category=retrieval_category
        )
    elif "Vector" in search_mode:
        results = st.session_state.vector_engine.search(
            search_query,
            top_k=5,
            translated_query=translated_q,
            category=retrieval_category
        )
    else:
        results = st.session_state.search_engine.search(
            search_query,
            top_k=5,
            translated_query=translated_q,
            category=retrieval_category
        )

    # Synthesize grounded response using retrieved evidence
    # Preserves student's original query for response synthesis while search used rewritten query
    ans_res = st.session_state.answer_generator.generate(
        query=student_query,
        retrieved_results=results,
        confidence_threshold=conf_threshold,
        conversation_history=history_turns
    )

    # Append user turn (Preserving student's original query!)
    st.session_state.conversation_history.append({
        "role": "user",
        "content": student_query,
        "rewritten_query": search_query if was_rewritten else None,
        "lang": lang_info["name"]
    })

    # Append assistant turn
    model_name = ans_res.get("model", "")
    gen_source = ans_res.get(
        "generation_source",
        "Gemini Generated" if "Gemini" in model_name else "Local Extractive Fallback"
    )

    # Synchronize session generator status with actual generator used for this answer
    is_gem = "Gemini" in gen_source or "Gemini" in model_name
    if is_gem:
        st.session_state["gemini_conn"] = (True, model_name or "Gemini Flash", "Active")
    else:
        err_msg = ans_res.get("gemini_error") or "Local fallback used"
        st.session_state["gemini_conn"] = (False, "Local Fallback", err_msg)

    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": ans_res["answer"],
        "citation": ans_res.get("primary_citation", ""),
        "source": ans_res.get("source", ""),
        "page": ans_res.get("page", ""),
        "model": model_name,
        "generation_source": gen_source,
        "gemini_error": ans_res.get("gemini_error"),
        "lang": ans_res.get("language", ""),
        "status": ans_res.get("status", ""),
        "confidence": ans_res.get("confidence", 0),
        "category": final_category_tag,
        "was_rewritten": was_rewritten,
        "search_query": search_query,
        "evidence": results[:3]
    })


# ---------------------------------------------------------
# Handle Any Pending Query (From Suggested Buttons or Inputs)
# Executed before UI rendering so state is consistent across sidebar & main view
# ---------------------------------------------------------
pending = st.session_state.get("pending_query", "").strip()
curr_threshold = st.session_state.get("conf_threshold", 0.25)
curr_search_mode = st.session_state.get("search_mode_option", "Hybrid (Semantic + BM25)")

if pending and pending != st.session_state.get("last_chat_query"):
    st.session_state.last_chat_query = pending
    st.session_state.pending_query = ""
    with st.spinner("Searching university policies..."):
        execute_query(
            pending,
            curr_threshold,
            curr_search_mode,
            category_filter=st.session_state.get("selected_category")
        )


# ---------------------------------------------------------
# Sidebar: Compact & Organized
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ University Helpdesk")
    st.caption("Multilingual AI Policy Assistant")

    # 1. New Chat
    if st.button("＋ New Chat", use_container_width=True, help="Start a fresh conversation"):
        clear_conversation()
        st.rerun()

    st.markdown("---")

    # 2. Upload Documents & Document List
    st.markdown("**📂 Document Knowledge Base**")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX or TXT documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload official university policies or syllabi."
    )

    if uploaded_files:
        newly_processed = 0
        for uploaded_file in uploaded_files:
            try:
                content = uploaded_file.read()
                if content:
                    f_hash = compute_content_hash(content)
                    # Check if document content hash is already indexed
                    already_indexed = any(c.get("doc_hash") == f_hash for c in st.session_state.indexed_chunks)
                    if already_indexed or uploaded_file.name in st.session_state.loaded_files:
                        continue
                    chunks = process_document(content, uploaded_file.name, doc_hash=f_hash)
                    if chunks:
                        primary_c = chunks[0].get("category", "General / Other")
                        st.session_state.doc_categories[uploaded_file.name] = chunks[0].get("categories", [primary_c])
                        st.session_state.indexed_chunks.extend(chunks)
                        st.session_state.loaded_files.add(uploaded_file.name)
                        newly_processed += len(chunks)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        if newly_processed > 0:
            st.session_state.search_engine.index_chunks(st.session_state.indexed_chunks)
            with st.spinner("Generating embeddings for uploaded files..."):
                st.session_state.vector_engine.index_chunks(st.session_state.indexed_chunks)
            st.success(f"Added {newly_processed} passages.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("📁 Load Samples", use_container_width=True, help="Load built-in university policy handbooks"):
            load_sample_documents()
    with col_s2:
        if st.button("Reset", use_container_width=True, help="Clear documents and index"):
            st.session_state.indexed_chunks = []
            st.session_state.loaded_files = set()
            st.session_state.doc_categories = {}
            st.session_state.search_engine = BM25SearchEngine()
            st.session_state.vector_engine = VectorSearchEngine()
            clear_conversation()
            st.rerun()

    # Uploaded document list with categories & manual override option
    if st.session_state.loaded_files:
        st.caption(f"**Indexed Documents ({len(st.session_state.loaded_files)}):**")
        for fname in sorted(st.session_state.loaded_files):
            cur_cats = st.session_state.doc_categories.get(fname, ["General / Other"])
            primary_c = cur_cats[0] if cur_cats else "General / Other"
            cat_icon = CATEGORY_ICONS.get(primary_c, "📋")

            col_doc, col_del = st.columns([4, 1])
            with col_doc:
                st.markdown(f"<span style='font-size:0.82rem;color:#1e293b;'>📄 <b>{fname}</b></span>", unsafe_allow_html=True)
            with col_del:
                if st.button("✕", key=f"rm_{fname}", help=f"Remove {fname}"):
                    remove_document(fname)
                    st.rerun()

            # Manual category override selectbox
            new_cat = st.selectbox(
                f"Category for {fname}",
                CATEGORIES,
                index=CATEGORIES.index(primary_c) if primary_c in CATEGORIES else 5,
                key=f"cat_override_{fname}",
                label_visibility="collapsed",
                help=f"Manual category override for {fname}"
            )
            if new_cat != primary_c:
                st.session_state.doc_categories[fname] = [new_cat]
                for c in st.session_state.indexed_chunks:
                    if c.get("source") == fname:
                        c["category"] = new_cat
                        c["categories"] = [new_cat]
                st.rerun()

    st.markdown("---")

    # 3. Language & Category Retrieval Settings
    st.markdown("**⚙️ Preferences**")
    cat_choices = [ALL_CATEGORIES] + CATEGORIES
    selected_cat = st.selectbox(
        "Policy Category",
        cat_choices,
        index=cat_choices.index(st.session_state.selected_category) if st.session_state.selected_category in cat_choices else 0,
        help="Filter search to a specific university category (e.g., Academics, Hostel), or choose 'All Categories' for universal search."
    )
    st.session_state.selected_category = selected_cat

    lang_choice = st.selectbox(
        "Preferred Language",
        ["Auto-detect", "English", "हिंदी (Hindi)", "తెలుగు (Telugu)", "ਪੰਜਾਬੀ (Punjabi)"],
        index=0,
        help="Language for chatbot responses and query detection."
    )
    st.session_state.active_language = lang_choice

    search_mode = st.selectbox(
        "Search Mode",
        ["Hybrid (Semantic + BM25)", "Semantic Vector", "Keyword (BM25)"],
        index=0,
        help="Retrieval engine architecture."
    )
    st.session_state.search_mode_option = search_mode

    conf_threshold = st.slider(
        "Grounding Threshold",
        min_value=0.10,
        max_value=0.60,
        value=0.25,
        step=0.05,
        help="Minimum retrieval score required to verify policy answers."
    )

    st.markdown("---")

    # 4. Gemini Connection Status (Never reveals API key)
    env_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_api_key:
        st.session_state.answer_generator.api_key = env_api_key
        # Check connection once per session
        if st.session_state.get("last_tested_key") != env_api_key:
            conn_ok, active_model, conn_msg = test_gemini_connection(env_api_key)
            st.session_state["gemini_conn"] = (conn_ok, active_model, conn_msg)
            st.session_state["last_tested_key"] = env_api_key

        conn_ok, active_model, conn_msg = st.session_state.get("gemini_conn", (False, "", ""))
        # If conversation exists, reflect the actual generator used for the latest answer
        latest_assistant_msg = next((m for m in reversed(st.session_state.conversation_history) if m.get("role") == "assistant"), None)
        if latest_assistant_msg:
            msg_gen = latest_assistant_msg.get("generation_source", "")
            msg_mod = latest_assistant_msg.get("model", "")
            conn_ok = "Gemini" in msg_gen or "Gemini" in msg_mod
            if msg_mod:
                active_model = msg_mod

        if conn_ok:
            st.markdown(
                '<div class="status-pill status-pill-green">'
                '<span class="status-dot status-dot-green"></span>'
                f'Gemini Connected ({active_model})</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-pill status-pill-amber">'
                '<span class="status-dot status-dot-amber"></span>'
                'Local Extractive Mode (Gemini Fallback)</div>',
                unsafe_allow_html=True
            )
    else:
        st.session_state.answer_generator.api_key = ""
        st.markdown(
            '<div class="status-pill status-pill-amber">'
            '<span class="status-dot status-dot-amber"></span>'
            'Local Extractive Mode</div>',
            unsafe_allow_html=True
        )

    # 5. Suggested Inquiries when in active chat (retains button keys for test suite compatibility)
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("**💡 Suggested Topics**")
        if st.button(
            "Attendance requirements",
            key="btn_q1",
            use_container_width=True,
            on_click=select_example_query,
            args=("What is the minimum attendance required for exams?",)
        ):
            st.rerun()
        if st.button(
            "Hostel curfew policies",
            key="btn_q6",
            use_container_width=True,
            on_click=select_example_query,
            args=("What are the hostel curfew timings on weekdays?",)
        ):
            st.rerun()
        if st.button(
            "Semester Abroad eligibility",
            key="btn_welcome_abroad_side",
            use_container_width=True,
            on_click=select_example_query,
            args=("What is the passport validity required for Semester Abroad?",)
        ):
            st.rerun()
        if st.button(
            "Academic benefits",
            key="btn_welcome_benefits_side",
            use_container_width=True,
            on_click=select_example_query,
            args=("What are the academic benefits and CGPA requirements for attendance?",)
        ):
            st.rerun()

    # 6. Clear Chat Button
    if st.session_state.conversation_history:
        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            clear_conversation()
            st.rerun()


# ---------------------------------------------------------
# Compact Top Header
# ---------------------------------------------------------
conn_tuple = st.session_state.get("gemini_conn")
is_gemini_online = conn_tuple[0] if conn_tuple else False
active_model_name = conn_tuple[1] if conn_tuple else "Flash"

latest_assistant_msg = next((m for m in reversed(st.session_state.conversation_history) if m.get("role") == "assistant"), None)
if latest_assistant_msg:
    msg_gen = latest_assistant_msg.get("generation_source", "")
    msg_mod = latest_assistant_msg.get("model", "")
    is_gemini_online = "Gemini" in msg_gen or "Gemini" in msg_mod
    if msg_mod:
        active_model_name = msg_mod

status_badge_html = (
    f'<div class="status-pill status-pill-green"><span class="status-dot status-dot-green"></span> Gemini Active</div>'
    if is_gemini_online else
    '<div class="status-pill status-pill-amber"><span class="status-dot status-dot-amber"></span> Local Fallback</div>'
)

st.markdown(f"""
<div class="top-navbar">
    <div class="top-navbar-title">
        🏛️ University Helpdesk
        <span class="top-navbar-sub">Multilingual AI Assistant</span>
    </div>
    <div>
        {status_badge_html}
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------
has_conversation = len(st.session_state.conversation_history) > 0

if not has_conversation:
    # -----------------------------------------------------
    # WELCOME SCREEN (When no conversation exists)
    # -----------------------------------------------------
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">🎓</div>
        <div class="welcome-title">University Helpdesk</div>
        <div class="welcome-subtitle">
            Your AI assistant for university policies, academic rules and student services.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Clean Suggested Question Buttons (2x2 Grid)
    # Note: btn_q1 and btn_q6 keys are preserved for full automated test suite compatibility
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        if st.button(
            "📋 Attendance requirements",
            key="btn_q1",
            use_container_width=True,
            help="Check minimum required class attendance for exams",
            on_click=select_example_query,
            args=("What is the minimum attendance required for exams?",)
        ):
            st.rerun()

        if st.button(
            "🏢 Hostel policies",
            key="btn_q6",
            use_container_width=True,
            help="View weekday and weekend hostel curfew rules",
            on_click=select_example_query,
            args=("What are the hostel curfew timings on weekdays?",)
        ):
            st.rerun()

    with q_col2:
        if st.button(
            "✈️ Semester Abroad eligibility",
            key="btn_welcome_abroad",
            use_container_width=True,
            help="Passport validity and eligibility criteria for semester exchange",
            on_click=select_example_query,
            args=("What is the passport validity required for Semester Abroad?",)
        ):
            st.rerun()

        if st.button(
            "🎯 Academic benefits",
            key="btn_welcome_benefits",
            use_container_width=True,
            help="Attendance concessions and CGPA merit criteria",
            on_click=select_example_query,
            args=("What are the academic benefits and CGPA requirements for attendance?",)
        ):
            st.rerun()

    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

else:
    # -----------------------------------------------------
    # ACTIVE CONVERSATION HISTORY
    # -----------------------------------------------------
    for msg in st.session_state.conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            lang = msg.get("lang", "")
            with st.chat_message("user"):
                st.markdown(content)
                if lang and lang != "English":
                    st.caption(f"Language: {lang}")
        else:
            gen_source = msg.get("generation_source", "Gemini Generated")
            model_used = msg.get("model", "")
            cit = msg.get("citation", "")
            gemini_err = msg.get("gemini_error")
            evidence = msg.get("evidence", [])

            is_gem = "Gemini" in gen_source
            badge_class = "chat-badge-gemini" if is_gem else "chat-badge-fallback"
            badge_text = f"🤖 {gen_source}" if is_gem else f"⚙️ {gen_source}"

            cat_tag = msg.get("category", "General / Other")
            cat_icon = CATEGORY_ICONS.get(cat_tag, "📋")
            was_rew = msg.get("was_rewritten", False)
            rew_query = msg.get("search_query", "")

            with st.chat_message("assistant"):
                # 1. Top Metadata Bar (Tight single-line HTML to prevent markdown code block formatting)
                rew_html = f'<span style="font-size:0.75rem; color:#6b7280; margin-left:0.35rem;" title="Context query">🔍 <i>{html.escape(rew_query)}</i></span>' if (was_rew and rew_query) else ''
                model_html = f'<span style="font-size:0.75rem; color:#94a3b8;">{html.escape(model_used)}</span>' if model_used else ''
                meta_html = (
                    f'<div class="assistant-meta-bar">'
                    f'<span class="chat-badge {badge_class}">{badge_text}</span>'
                    f'<span class="chat-badge chat-badge-category">{cat_icon} {cat_tag}</span>'
                    f'{model_html}'
                    f'{rew_html}'
                    f'</div>'
                )
                st.markdown(meta_html, unsafe_allow_html=True)

                # 2. Complete Gemini Answer (Full-width block level, complete Markdown formatting)
                st.markdown(content)

                # 3. Source & Citation
                if cit:
                    st.markdown(f'<div class="chat-citation-bar">📑 <b>Source:</b> <code>{html.escape(cit)}</code></div>', unsafe_allow_html=True)

                # 4. Fallback Note
                if gemini_err:
                    st.markdown(f'<div class="fallback-note">⚠️ <b>Fallback notice:</b> {html.escape(gemini_err)}</div>', unsafe_allow_html=True)

                # 5. Supporting Evidence (Clean Expandable Section)
                if evidence:
                    with st.expander(f"Supporting Evidence ({len(evidence)} passage{'s' if len(evidence) != 1 else ''})"):
                        for rank, res in enumerate(evidence, start=1):
                            chunk = res.get("chunk", {})
                            score = res.get("score", "")
                            src_name = chunk.get("source", "Document")
                            page_val = chunk.get("page")
                            sec_val = chunk.get("section") or chunk.get("section_title")
                            scope_val = chunk.get("programme_scope")
                            raw_text = chunk.get("text", "").strip()

                            page_str = f"Page: {page_val}" if page_val is not None else (f"Section: {sec_val}" if sec_val else "")
                            scope_str = f"Scope: {scope_val}" if (scope_val and scope_val != "General") else ""
                            score_str = f"Score: {score:.3f}" if isinstance(score, (int, float)) else (f"Score: {score}" if score else "")

                            # Format text cleanly, converting markdown tables into clean HTML tables
                            safe_text = format_chunk_display_html(raw_text)

                            page_badge = f'<span class="page-badge">{page_str}</span>' if page_str else ''
                            scope_badge = f'<span class="page-badge scope-badge">{scope_str}</span>' if scope_str else ''
                            score_badge = f'<span class="score-badge">{score_str}</span>' if score_str else ''

                            card_html = (
                                f'<div class="result-card">'
                                f'<div class="result-card-meta">'
                                f'<span class="source-badge">Source: {src_name}</span>'
                                f'{page_badge}'
                                f'{scope_badge}'
                                f'{score_badge}'
                                f'</div>'
                                f'<div class="chunk-text">{safe_text}</div>'
                                f'</div>'
                            )
                            st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------
# Chat Input (Clean, Bottom-Anchored)
# ---------------------------------------------------------
with st.form(key="chat_input_form", clear_on_submit=True):
    col_input, col_submit = st.columns([5, 1])
    with col_input:
        user_typed_query = st.text_input(
            "Ask anything about university policies...",
            placeholder="Ask anything about university policies, attendance, fees or hostels...",
            key="search_input",
            label_visibility="collapsed"
        )
    with col_submit:
        submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

if submitted and user_typed_query.strip():
    q_str = user_typed_query.strip()
    with st.spinner("Generating answer..."):
        execute_query(
            q_str,
            conf_threshold,
            st.session_state.search_mode_option,
            category_filter=st.session_state.selected_category
        )
    st.rerun()

st.markdown("""
<div class="chat-disclaimer">
    Grounded strictly in verified university handbooks. Multilingual retrieval supports English, Hindi, Telugu, and Punjabi.
</div>
""", unsafe_allow_html=True)
