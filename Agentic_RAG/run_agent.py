# run_agent.py
# Orchestrator — master agent that controls the flow
# Calls each specialist node in the correct order
# Manages the retry loop when grade = "fail"
# No UI here — pure Python logic only

from nodes.state import create_state
from nodes.query_pipeline import classifier_node, section_node, query_planner_node
from nodes.data_pipeline import retrieve_node, reranker_node
from nodes.decision_node import grade_node, answer_node


def run_agent(query: str, llm, embedding,
              header_chunks, header_embedding,
              child_embedding, fixed_embedding,
              fixed_chunks, child_chunks) -> dict:

    # ── Step 1: initialise shared memory ──────────────────────
    state = create_state(
        query=query,
        header_chunks=header_chunks,
        embedding=embedding,
        header_embedding=header_embedding,
        child_embedding=child_embedding,
        fixed_embedding=fixed_embedding,
        fixed_chunks=fixed_chunks,
        child_chunks=child_chunks
    )

    # ── Step 2: classify the query ────────────────────────────
    # Conditional Edge: multi-topic → decompose / single → rewrite
    state = classifier_node(state)

    # ── Step 3: find relevant sections ───────────────────────
    # LLM needs real section names before generating sub-questions
    state = section_node(state)

    # ── Step 4: understand the query (decompose or rewrite) ──
    # query_planner_node is the Conditional Edge:
    # multi-topic → decompose_node / single-topic → rewrite_node
    state = query_planner_node(state, llm)

    # ── Step 5: retrieve → rerank → grade loop ───────────────
    # Agent retries with a different strategy if grade = "fail"
    # Stops when grade = "pass" or "exhausted"
    while True:
        state = retrieve_node(state)
        state = reranker_node(state)
        state = grade_node(state)

        if state["grade"] == "pass":
            # good chunks found → generate answer
            state = answer_node(state, llm)
            break

        elif state["grade"] == "exhausted":
            # max retries reached → fallback answer already set
            break

        else:
            # grade = "fail" → grade_node already switched strategy
            # loop back and try again with new strategy
            print(f"Retrying with strategy: {state['current_strategy']}...")
            continue

    return state