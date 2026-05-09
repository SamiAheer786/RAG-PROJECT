"""
Part 4: Streamlit User Interface
Medical FAQ Assistant RAG System

Run with:  streamlit run src/app.py
"""

import os
import sys

# ── Path setup: works both locally and on Streamlit Cloud ──────────────────
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))   # .../src
PROJECT_DIR = os.path.dirname(SRC_DIR)                      # .../rag_project

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

CHUNKS_PATH = os.path.join(PROJECT_DIR, "corpus", "chunks", "chunks.json")

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical FAQ RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached resource loaders ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base…")
def load_tfidf_retriever():
    from part2_retrieval import load_chunks, TFIDFRetriever
    chunks = load_chunks(CHUNKS_PATH)
    r = TFIDFRetriever()
    r.build_index(chunks)
    return r, chunks


@st.cache_resource(show_spinner="Building dense retriever (LSA)…")
def load_dense_retriever(_chunks):
    from part2_retrieval import DenseRetriever
    r = DenseRetriever()
    r.load_model()
    cache = os.path.join(PROJECT_DIR, "corpus", "chunks", "embeddings.npy")
    r.build_index(_chunks, cache_path=cache)
    return r


@st.cache_resource(show_spinner="Loading language model…")
def load_llm(backend: str):
    from part3_generation import HuggingFaceLLM, TemplateLLM
    if backend == "HuggingFace (flan-t5-base)":
        try:
            llm = HuggingFaceLLM()
            llm.load()
            return llm
        except Exception:
            st.warning("HuggingFace model unavailable; using template fallback.")
            return TemplateLLM()
    return TemplateLLM()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    retrieval_method = st.selectbox(
        "Retrieval Method",
        ["Dense (LSA)", "TF-IDF"],
        help="Dense uses TF-IDF + SVD (LSA); TF-IDF is sparse keyword retrieval.",
    )
    llm_backend = st.selectbox(
        "Language Model",
        ["Template (Demo)", "HuggingFace (flan-t5-base)"],
    )
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=6, value=3)
    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "Medical FAQ assistant built with a RAG pipeline. "
        "Retrieves passages from a curated medical knowledge base "
        "and generates grounded answers."
    )
    st.markdown("---")
    show_sources = st.checkbox("Show retrieved evidence", value=True)
    show_scores  = st.checkbox("Show retrieval scores",   value=True)


# ── Main ───────────────────────────────────────────────────────────────────
st.title("🏥 Medical FAQ Assistant")
st.caption("Powered by RAG · NUCES Chiniot-Faisalabad · AI/NLP Project")
st.info(
    "⚠️ General medical information only. "
    "Always consult a qualified healthcare professional for personal advice.",
    icon="ℹ️",
)

# Load resources
tfidf_ret, chunks = load_tfidf_retriever()

if retrieval_method == "Dense (LSA)":
    try:
        retriever = load_dense_retriever(chunks)
    except Exception as e:
        st.warning(f"Dense retriever unavailable ({e}). Using TF-IDF.")
        retriever = tfidf_ret
else:
    retriever = tfidf_ret

llm = load_llm(llm_backend)

# ── Chat history ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("📄 Retrieved Evidence", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    score_str = f" (score: {src['score']:.4f})" if show_scores else ""
                    st.markdown(f"**[{i}] {src['source']}{score_str}**")
                    st.markdown(src["chunk_text"])
                    st.markdown("---")

# Example buttons
st.markdown("**Try an example question:**")
cols = st.columns(3)
examples = [
    "What is type 2 diabetes?",
    "How do vaccines work?",
    "What are the side effects of antibiotics?",
]
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["prefill"] = ex

# Chat input
query = st.chat_input("Ask a medical question…")
if "prefill" in st.session_state and not query:
    query = st.session_state.pop("prefill")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.history.append({"role": "user", "content": query, "sources": []})

    with st.spinner("Searching knowledge base…"):
        context_chunks = retriever.retrieve(query, top_k=top_k)

    with st.spinner("Generating answer…"):
        from part3_generation import build_rag_prompt
        prompt = build_rag_prompt(query, context_chunks)
        answer = llm.generate(prompt)

    with st.chat_message("assistant"):
        st.markdown(answer)
        if show_sources:
            with st.expander("📄 Retrieved Evidence", expanded=True):
                for i, c in enumerate(context_chunks, 1):
                    score_str = f" (score: {c['score']:.4f})" if show_scores else ""
                    st.markdown(f"**[{i}] {c['source']}{score_str}**")
                    st.markdown(c["chunk_text"])
                    st.markdown("---")

    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "sources": context_chunks,
    })

if st.session_state.history:
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.rerun()
