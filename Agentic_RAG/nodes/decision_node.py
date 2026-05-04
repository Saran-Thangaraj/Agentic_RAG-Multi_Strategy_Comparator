# nodes/decision_node.py
# Decision and answer agents
# grade_node   → are the chunks good enough? pass / fail / exhausted
# answer_node  → generate final answer from good chunks

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

STRATEGIES = ["header", "child", "fixed"]


# ── Grade Node ────────────────────────────────────────────────
# One job: decide if retrieved chunks are good enough to answer
#
# Strategy — minimum quality gate + dynamic threshold:
#   1. If avg score < MIN_QUALITY_THRESHOLD (0.30) → fail immediately,
#      switch strategy (prevents passing poor context to LLM)
#   2. If avg score >= 0.30 → keep chunks above average → pass
#      (max score always >= avg, so good_chunks is never empty here)
#   3. If retry_count >= max_retry → exhausted, give up gracefully
#
# Writes: state["grade"]            = "pass" / "fail" / "exhausted"
#         state["reranked_chunks"]   = filtered good chunks (on pass)
#         state["retry_count"]       = incremented (on fail)
#         state["current_strategy"]  = next strategy to try (on fail)

MIN_QUALITY_THRESHOLD = 0.35

def grade_node(state: dict) -> dict:
    chunks = state["reranked_chunks"]

    if state["retry_count"] >= state["max_retry"]:
        state["grade"]  = "exhausted"
        state["answer"] = (
            "I could not find relevant information for your query. "
            "Please try rephrasing your question."
        )
        return state

    scores    = [doc.metadata.get("relevance_score", 0) for doc in chunks]
    print("Relevance scores:", scores)

    avg_score = sum(scores) / len(scores) if scores else 0
    print("Average score:", avg_score)

    if avg_score < MIN_QUALITY_THRESHOLD:
        next_retry           = state["retry_count"] + 1
        state["retry_count"] = next_retry

        if next_retry >= len(STRATEGIES):
            state["grade"]  = "exhausted"
            state["answer"] = (
                "I could not find relevant information for your query. "
                "Please try rephrasing your question."
            )
            print(f"Grade: EXHAUSTED — all {len(STRATEGIES)} strategies tried, none passed threshold")
            return state

        state["grade"]            = "fail"
        state["current_strategy"] = STRATEGIES[next_retry]
        print(f"Grade: FAIL (avg {avg_score:.3f} < {MIN_QUALITY_THRESHOLD}) — switching to {state['current_strategy']}, retry {next_retry}")
        return state

    good_chunks = [doc for doc in chunks if doc.metadata.get("relevance_score", 0) >= avg_score]
    if good_chunks:
        state["grade"]           = "pass"
        state["reranked_chunks"] = good_chunks
    return state


# ── Answer Node ───────────────────────────────────────────────
# One job: generate final answer from good chunks
# Only called when grade = "pass"
# Writes: state["answer"] = final answer string

answer_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""Answer the question using ONLY the context below. Do NOT use any outside knowledge.

Rules:
- Include complete code examples exactly as they appear in context
- Do not summarize or shorten code blocks
- Structure your answer clearly
- If the context does not contain enough information to answer the question, respond with exactly:
  "I could not find a relevant answer to your question in the provided document."
- Always wrap all code examples in triple backticks ```python ... ```

Context:
{context}

Question: {query}
Answer:"""
)

def answer_node(state: dict, llm) -> dict:
    query   = state["query"]
    chunks  = state["reranked_chunks"]
    context = "\n\n".join([doc.page_content for doc in chunks])

    chain           = answer_prompt | llm | StrOutputParser()
    state["answer"] = chain.invoke({"query": query, "context": context})
    return state
