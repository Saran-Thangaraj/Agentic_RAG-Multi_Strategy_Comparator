# Agentic RAG — Self-Correcting Document Q&A

A RAG pipeline that thinks before it answers. Instead of retrieving once and hoping for the best, this agent grades its own retrieval, switches strategies when results are poor, and retries up to 4 times before falling back gracefully.

---

## The Problem with Traditional RAG

Traditional RAG is a one-shot pipeline — query → retrieve → answer. No checking, no correction.

| Problem | What went wrong |
|---|---|
| GIGO | Wrong chunks passed to LLM → wrong answer, silently |
| Fixed threshold | `relevance_score > 0.3` filtered out good chunks on hard queries |
| Pure vector search | Keyword coincidence beat semantic relevance |
| One function does everything | `decompose_and_retrieve()` had 5 jobs — impossible to debug |
| No fallback | System returned confident wrong answers instead of "I don't know" |

---

## What Agentic RAG Fixes

| Traditional RAG | Agentic RAG |
|---|---|
| No shared memory | **State** — shared dictionary passed between every node |
| One function does all | **Each node has one job** — easier to debug and improve |
| No routing logic | **Conditional edges** — smart routing based on query type |
| One shot, no checking | **Grade node** — checks relevance scores after every retrieval |
| Fixed threshold (0.3) | **Dynamic threshold** — cuts at average score, adapts per query |
| Pure vector search | **Hybrid retrieval** — BM25 + vector, catches keyword + semantic matches |
| No retry | **Retry loop** — switches strategy (header → child → fixed) on failure |
| Max 1 attempt | **Max 4 retries** — exhausts all strategies before giving up |
| No fallback | **Graceful fallback** — "I could not find relevant information" |

---

## Issues Found and Fixes Applied

### Issue 1 — Irrelevant chunks ranked higher than relevant ones
**Cause:** Pure vector search matches words, not meaning. A chunk mentioning "chunking" twice outranked a chunk explaining it properly.
**Fix:** Replaced `retrieval_embedding.as_retriever()` with `EnsembleRetriever` combining BM25 (keyword) and vector (semantic) at 40/60 weight.

### Issue 2 — Fixed threshold 0.3 filtered good chunks
**Cause:** On difficult queries, even correct chunks scored below 0.3. Setting threshold too low let noise through. Setting it high lost good chunks.
**Fix:** Dynamic threshold — compute average score across all retrieved chunks, keep only chunks that score at or above the average. Threshold adapts to each query.

### Issue 3 — `decompose_and_retrieve()` was one giant function
**Cause:** One function doing classify + section finding + decompose + retrieve + deduplicate made it impossible to test or fix one step without breaking others.
**Fix:** Split into individual nodes — `classifier_node`, `section_node`, `decompose_node`, `rewrite_node`, `retrieve_node` — each with one job.

### Issue 4 — No recovery when retrieval failed
**Cause:** Traditional RAG had no concept of "this retrieval was bad, try again."
**Fix:** `grade_node` reads relevance scores, decides pass/fail, and triggers strategy switch on failure. Agent retries with header → child → fixed stores in sequence.

---

## Project Structure

```
agentic_rag/
├── nodes/
│   ├── state.py            → shared memory — stores every node's result
│   ├── query_pipeline.py   → classifier, section, decompose, rewrite, query_planner
│   ├── data_pipeline.py    → retrieve (hybrid), reranker
│   └── decision_node.py    → grade (dynamic threshold), answer
├── run_agent.py            → orchestrator — controls the flow and retry loop
└── agentic_main.py         → Streamlit UI — user input and answer display
```

---

## How the Agent Works

```
User Query
    ↓
[classifier_node]   → is this simple or complex?
    ↓
[section_node]      → which sections exist in the document?
    ↓
[query_planner_node]
    ├── complex → [decompose_node]   → break into 3-5 sub-questions
    └── simple  → [rewrite_node]    → rewrite for precision
    ↓
[retrieve_node]     → hybrid retrieval (BM25 + vector) from current strategy store
    ↓
[reranker_node]     → score each chunk against query using Cohere
    ↓
[grade_node]        → dynamic threshold check
    ├── PASS       → [answer_node] → generate final answer → done
    ├── FAIL       → switch strategy → retry (max 4 times)
    └── EXHAUSTED  → graceful fallback → "I could not find relevant information"
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API keys to .env
GROQ_API_KEY=your_key_here
COHERE_API_KEY=your_key_here

# 4. Run the app
streamlit run agentic_main.py
```

---

## Key Concepts Learned

**State** — a shared dictionary that every node reads from and writes to. Nothing is lost between steps. This is what makes the retry loop possible.

**Node** — a Python function with one job. Takes state as input, returns updated state as output.

**Conditional Edge** — an if/else decision that routes the flow to the next node based on what's in state. The grade node's pass/fail decision is a conditional edge.

**Dynamic Threshold** — instead of a fixed score cutoff, compute the average relevance score of retrieved chunks and keep only those at or above average. Adapts to each query.

**Hybrid Retrieval** — combines BM25 keyword matching with vector semantic search. BM25 catches exact keyword matches that vector search misses. Vector search catches meaning that keyword search misses.
