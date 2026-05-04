import hashlib
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': True}
    )


def generate_hash(chunks):
    hashes = []
    for chunk in chunks:
        content = chunk.page_content
        metadata = chunk.metadata

        meta_string = (
            str(metadata.get("source", "")) +
            str(metadata.get("page_chapter", "")) +
            str(metadata.get("strategy", "")) +
            str(metadata.get("Header 1", "")) +
            str(metadata.get("Header 2", '')) +
            str(metadata.get("chunk_type", '')) +
            str(metadata.get("parent_id", ''))
        )

        hash_input = content + '||' + meta_string
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        chunk.metadata['hash'] = hash_value
        hashes.append(hash_value)

    return chunks, hashes


def store_embeddings(chunks, collection_name, embedding):
    chunks, hashes = generate_hash(chunks)

    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory="./chroma_chunks_db"
    )

    existing_ids = set(vector_db.get(ids=hashes)['ids'])
    new_hash_set = set(hashes) - existing_ids

    new_chunks = []
    new_hashes = []

    for chunk, hash_id in zip(chunks, hashes):
        if hash_id in new_hash_set:
            new_chunks.append(chunk)
            new_hashes.append(hash_id)

    if new_chunks:
        vector_db.add_documents(documents=new_chunks, ids=new_hashes)

    return vector_db
