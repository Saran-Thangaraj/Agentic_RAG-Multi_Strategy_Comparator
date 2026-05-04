# agentic_main.py
# Entry point — Streamlit UI
# User types query here → run_agent handles everything internally
# No node logic here — only UI and pipeline setup

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.ingestion import load_pdf
from src.embeddings import get_embedding_model, store_embeddings
from src.chunkers import fixed_chunker, header_chunker, parent_child_chunker
from Agentic_RAG.run_agent import run_agent

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")
st.title("🤖 Adaptive Multi-Strategy RAG")
st.caption("Powered by dynamic grading, retry logic, and multi-strategy retrieval")

# ── Sidebar — PDF upload ──────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Classify query type")
    st.markdown("2. Find relevant sections")
    st.markdown("3. Decompose or rewrite query")
    st.markdown("4. Retrieve → Rerank → Grade")
    st.markdown("5. Retry with different strategy if needed")
    st.markdown("6. Generate answer from best chunks")

# ── Build pipeline (cached so it only runs once) ──────────────
@st.cache_resource
def build_pipeline(pdf_path: str):
    pages         = load_pdf(pdf_path)
    embedding     = get_embedding_model()
    fixed_chunks  = fixed_chunker.chunk(pages)
    header_chunks = header_chunker.chunk(pages)
    child_chunks  = parent_child_chunker.create_child_chunks(header_chunks)
    parent_chunks = parent_child_chunker.create_parent_chunks(child_chunks)

    fixed_embedding  = store_embeddings(fixed_chunks,  "Fixed_Chunks",  embedding)
    header_embedding = store_embeddings(header_chunks, "Header_Chunks", embedding)
    child_embedding  = store_embeddings(child_chunks,  "child_chunks",  embedding)
    parent_child_chunker.store_parent_chunks(parent_chunks)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

    return (embedding, llm,
            fixed_embedding, fixed_chunks,
            header_embedding, header_chunks,
            child_embedding, child_chunks)


# ── Main UI ───────────────────────────────────────────────────
if uploaded_file:
    # save uploaded file temporarily
    pdf_path = f"/tmp/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Building pipeline... (only once per document)"):
        (embedding, llm,
         fixed_embedding, fixed_chunks,
         header_embedding, header_chunks,
         child_embedding, child_chunks) = build_pipeline(pdf_path)

    st.success("Pipeline ready. Ask your question below.")

    # ── Query input ───────────────────────────────────────────
    query = st.text_input("Ask a question about your document:",
                          placeholder="e.g. Explain hybrid search and reranking")

    if query:
        with st.spinner("Agent is thinking..."):
            state = run_agent(
                query=query,
                llm=llm,
                embedding=embedding,
                header_chunks=header_chunks,
                header_embedding=header_embedding,
                child_embedding=child_embedding,
                fixed_embedding=fixed_embedding,
                fixed_chunks=fixed_chunks,
                child_chunks=child_chunks
            )

        # ── Display answer ────────────────────────────────────
        st.subheader("Answer")
        st.markdown(state["answer"])

        # ── Debug info (expandable) ───────────────────────────
        with st.expander("🔍 Agent Debug Info"):
            st.write(f"**Strategy used:** `{state['current_strategy']}`")
            st.write(f"**Retries:** `{state['retry_count']}`")
            st.write(f"**Grade:** `{state['grade']}`")
            st.write(f"**Query type:** `{'Multi-topic' if state['classifier'] else 'Single-topic'}`")

            if state["classifier"]:
                st.write("**Sub-questions generated:**")
                for i, q in enumerate(state["sub_questions"], 1):
                    st.write(f"  {i}. {q}")
            else:
                st.write(f"**Rewritten query:** `{state['direct_query']}`")

            st.write("**Chunks used:**")
            for doc in state.get("reranked_chunks", []):
                section = (doc.metadata.get("Header 2") or
                           doc.metadata.get("Header 1") or "Unknown")
                score   = doc.metadata.get("relevance_score", "N/A")
                st.write(f"  - {section} → score: {score}")
else:
    st.info("👈 Upload a PDF from the sidebar to get started.")
