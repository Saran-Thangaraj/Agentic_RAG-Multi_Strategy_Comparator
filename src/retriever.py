import re
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser,StrOutputParser
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


# Chapters that consistently return noise — excluded for all queries
NOISE_CHAPTERS = {
    "Chapter 14: API Reference",
    "Chapter 2: Python Environment Setup",
    "Appendix E: Testing Your RAG Pipeline",
    "Appendix F: RAG Interview Questions & Model Answers"
}


def is_multi_topic(query: str) -> bool:
    has_bullets = re.search(r'[-•]\s', query)
    has_commas = re.search(r'[,]\s', query)
    has_vs = re.search(r'\b(vs\.?)', query)
    has_compare = re.search(r'\b(compare|comparison|difference between|contrast)\b', query, re.IGNORECASE)
    has_ampersand = re.search(r'\s&\s', query)
    has_multiple = re.search(
    r'\b(different|various)\b.*\b(methods|types|strategies|kinds)\b',
    query,
    re.IGNORECASE
)
    return bool(has_bullets or has_commas or has_vs or has_compare or has_ampersand or has_multiple)


def extract_sections(chunks: list) -> list:
    seen = set()
    sections = []
    for chunk in chunks:
          label = (chunk.metadata.get("parent_id") or
            chunk.metadata.get("header_id") or
            chunk.metadata.get("Header 1", "") or
            chunk.metadata.get("page_chapter", "")
          )
          if label and label not in seen:
            seen.add(label)
            sections.append(label)
    return sections


def get_relevant_sections(query: str, chunks: list, embedding,k=4) -> list:
    sections = extract_sections(chunks)
    section_docs = [Document(page_content=section) for section in sections]

    bm25 = BM25Retriever.from_documents(section_docs)
    bm25.k = k

    mini_vectorstore = Chroma.from_documents(
        documents=section_docs,
        embedding=embedding
    )
    vector = mini_vectorstore.as_retriever(search_kwargs={"k": k})

    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.2, 0.8]
    )

    results = hybrid.invoke(query)
    seen = set()
    unique = []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return [doc.page_content for doc in unique]


decompose_prompt = PromptTemplate(
    input_variables=["question", "sections"],
    template="""
You are an expert search query planner for a RAG system.

Available sections in the document:
{sections}

Generate 4-5 natural sub-queries based ONLY on the above sections.
Rules:
- Use section names as guidance
- Do NOT invent topics not in sections
- Return ONLY comma-separated list

User question:
{question}

Sub-questions:"""
)

def decompose_and_retrieve(query: str, embedding, chunks: list, llm, retrieval_embedding) -> list:
    parser = CommaSeparatedListOutputParser()

    if is_multi_topic(query):
        qe = get_relevant_sections(query, chunks, embedding)


        allowed_chapters = set()
        for section in qe:
            parts = section.split("-")
            if len(parts) > 1:
                allowed_chapters.add(parts[1])

        # Remove noise chapters
        allowed_chapters = allowed_chapters - NOISE_CHAPTERS


        chain = decompose_prompt | llm | parser
        sub_questions = chain.invoke({
            "question": query,
            "sections": "\n".join(qe)
        })
    else:
        # Single topic → find relevant section → rewrite query
        qe = get_relevant_sections(query, chunks, embedding, k=1)
        rewrite_prompt = PromptTemplate(
            input_variables=["question", "sections"],
            template="""
         Rewrite this question as a specific search query.
         Use the available sections as context.

        Available sections: {sections}
        Question: {question}
        Rewritten query (one line only):"""
        )
        chain = rewrite_prompt | llm | StrOutputParser()
        rewritten = chain.invoke({
            "question": query,
            "sections": "\n".join(qe)
        })

        print("Rewritten query:", rewritten)
        sub_questions = [rewritten]
        allowed_chapters = None


    print("Sub-questions:", sub_questions)

    sub_results = []
    for sub_q in sub_questions:
        search_kwargs = {"k": 5}

        if allowed_chapters:
            search_kwargs["k"] = 3
            search_kwargs["filter"] = {"Header 1": {"$in": list(allowed_chapters)}}
        else:
            # Single topic: exclude noise chapters
            search_kwargs["filter"] = {"Header 1": {"$nin": list(NOISE_CHAPTERS)}}

        retriever = retrieval_embedding.as_retriever(search_kwargs=search_kwargs)
        answer = retriever.invoke(sub_q)
        sub_results.append((sub_q, answer))

    # Deduplicate
    seen = set()
    unique_context = []
    for sub_q, docs in sub_results:
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_context.append(doc)

    return unique_context
