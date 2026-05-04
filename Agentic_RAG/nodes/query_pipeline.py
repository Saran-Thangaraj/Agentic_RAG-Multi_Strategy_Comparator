# nodes/query_pipeline.py
# Query understanding agents
# classifier_node  → is this simple or complex?
# section_node     → which sections exist in the document?
# decompose_node   → break complex query into sub-questions
# rewrite_node     → rewrite simple query to be more specific

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from src.retriever import is_multi_topic, get_relevant_sections


# ── Classifier Node ───────────────────────────────────────────
# One job: decide if the query is multi-topic or single-topic
# Writes: state["classifier"] = True/False

def classifier_node(state: dict) -> dict:
    query = state["query"]
    state["classifier"] = is_multi_topic(query)
    return state


# ── Section Node ──────────────────────────────────────────────
# One job: find which sections in the document are relevant
# Why: LLM needs real section names before generating sub-questions
#      Without this, LLM invents topics not in the document
# Writes: state["sections"] = list of relevant section names

def section_node(state: dict) -> dict:
    query         = state["query"]
    header_chunks = state["header_chunks"]
    embedding     = state["embedding"]

    sections = get_relevant_sections(query, header_chunks, embedding)
    state["sections"] = sections
    return state


# ── Decompose Node ────────────────────────────────────────────
# One job: break a complex multi-topic query into sub-questions
# Only called when classifier = True
# Writes: state["sub_questions"] = list of sub-questions

decompose_prompt = PromptTemplate(
    input_variables=["question", "sections"],
    template="""
You are an expert search query planner for a RAG system.

Available sections in the document:
{sections}

User question: {question}

Step 1 — For each section above, decide: is this section topically relevant to the user question?
         A section is relevant ONLY if its title directly addresses a concept asked in the question.
         Discard any section whose title does not relate to the question.

Step 2 — From the relevant sections identified in Step 1, generate 3-5 specific sub-queries.
         Each sub-query must be directly answerable from one of those relevant sections.

Rules:
- Sub-queries must be grounded in the user question, not invented from section titles alone
- Do NOT generate a sub-query for a section that is irrelevant to the user question
- Do NOT invent topics not present in the user question
- Return ONLY a comma-separated list of sub-queries

Sub-questions:"""
)

def decompose_node(state: dict, llm) -> dict:
    parser   = CommaSeparatedListOutputParser()
    question = state["query"]
    sections = state["sections"]

    chain         = decompose_prompt | llm | parser
    sub_questions = chain.invoke({
        "question": question,
        "sections": "\n".join(sections)
    })

    state["sub_questions"] = sub_questions
    return state


# ── Rewrite Node ──────────────────────────────────────────────
# One job: rewrite a simple single-topic query to be more specific
# Only called when classifier = False
# Writes: state["direct_query"] = rewritten query string

rewrite_prompt = PromptTemplate(
    input_variables=["question", "sections"],
    template="""
Rewrite this question as a specific search query.
Use the available sections as context.

Available sections: {sections}
Question: {question}
Rewritten query (one line only):"""
)

# ── Query Planner Node ────────────────────────────────────────
# One job: route to decompose_node or rewrite_node based on classifier
# This is the Conditional Edge — one function, one decision
# Writes: state["sub_questions"] or state["direct_query"]

def query_planner_node(state: dict, llm) -> dict:
    if state["classifier"]:
        return decompose_node(state, llm)   # multi-topic path
    else:
        return rewrite_node(state, llm)     # single-topic path


def rewrite_node(state: dict, llm) -> dict:
    question = state["query"]
    sections = state["sections"]

    chain   = rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({
        "question": question,
        "sections": "\n".join(sections)
    })

    print("Rewritten query:", rewritten)
    state["direct_query"] = rewritten
    return state