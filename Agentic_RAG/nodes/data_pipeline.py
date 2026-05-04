# nodes/data_pipeline.py
# Data fetching agents
# retrieve_node  → fetch chunks using hybrid retrieval (BM25 + vector)
# reranker_node  → score each chunk against the query

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from src.reranker import rerank
import time

STRATEGIES = ["header", "child", "fixed"]


# ── Retrieve Node ─────────────────────────────────────────────
# One job: fetch chunks from the correct vector store
# Uses HYBRID retrieval (BM25 + vector) — fixes pure vector search bug
# BM25 catches keyword matches, vector catches semantic meaning
# Picks store based on state["current_strategy"]
# Writes: state["chunks"] = list of retrieved documents

def retrieve_node(state: dict) -> dict:
    strategy = state["current_strategy"]

    # pick the right store based on current strategy
    if strategy == "header":
        embedding_store = state["header_embedding"]
        chunk_store     = state["header_chunks"]
    elif strategy == "child":
        embedding_store = state["child_embedding"]
        chunk_store     = state["child_chunks"]
    else:  # fixed
        embedding_store = state["fixed_embedding"]
        chunk_store     = state["fixed_chunks"]

    # use sub_questions for multi-topic, direct_query for single-topic
    queries = state["sub_questions"] or [state["direct_query"]]

    # hybrid retrieval for each query
    seen, all_chunks = set(), []
    for q in queries:
        bm25   = BM25Retriever.from_documents(chunk_store)
        bm25.k = 3
        vector = embedding_store.as_retriever(search_kwargs={"k": 3})
        hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.4, 0.6])

        for doc in hybrid.invoke(q):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_chunks.append(doc)

    state["chunks"] = all_chunks
    return state


# ── Reranker Node ─────────────────────────────────────────────
# One job: score chunks against query and keep the best ones
# Multi-topic → rerank per sub-question to cover all aspects
# Single-topic → rerank against rewritten query for precision
# Writes: state["reranked_chunks"] = scored and filtered documents

def reranker_node(state: dict) -> dict:
    chunks = state["chunks"]

    if state["classifier"]:
        # multi-topic: rerank each sub-question separately
        seen, reranked_all = set(), []
        for sub_q in state["sub_questions"]:
            time.sleep(10)  # avoid rate limits
            reranked = rerank(sub_q, chunks, top_n=3)
            for doc in reranked:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    reranked_all.append(doc)
        state["reranked_chunks"] = reranked_all
    else:
        # single-topic: rerank against rewritten query
        top_n = 3 if state["current_strategy"] == "child" else 2
        state["reranked_chunks"] = rerank(state["direct_query"], chunks, top_n=top_n)

    return state