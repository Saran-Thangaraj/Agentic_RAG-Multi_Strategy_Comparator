from langchain_cohere import CohereRerank


def rerank(query: str, docs: list, top_n: int = 5) -> list:
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=top_n
    )
    reranked_docs = reranker.compress_documents(
        query=query,
        documents=docs
    )
    # print(reranked_docs)
    # threshold = 0.3
    # # Filter low score chunks
    # reranked_docs =  [doc for doc in reranked_docs 
    #         if doc.metadata.get('relevance_score', 0) > threshold]

    return reranked_docs
