import json
import os
from copy import deepcopy
from langchain_core.documents import Document


def create_child_chunks(header_chunks: list) -> list:
    child_chunks = []
    parent_child_input = deepcopy(header_chunks)

    for idx, chunk in enumerate(parent_child_input):
        chunk.metadata['chunk_type'] = 'child'
        chunk.metadata['strategy'] = 'parent_child'
        chunk.metadata.pop('header_id',None)
        chunk.metadata['parent_id'] = (
            chunk.metadata.get('source') + '-' +
            chunk.metadata.get('Header 1', 'MISSING') + "-" +
            chunk.metadata.get('Header 2', 'MISSING') + "-" + str(idx)
        )
        chunk.metadata['chunk_index'] = idx
        child_chunks.append(chunk)

    return child_chunks


def create_parent_chunks(child_chunks: list) -> list:
    parent_chunks = []

    for i, chunk in enumerate(child_chunks):
        current_chunk = child_chunks[i]
        previous_chunk = None
        next_chunk = None

        if i > 0 and child_chunks[i-1].metadata['Header 1'] == current_chunk.metadata['Header 1']:
            previous_chunk = child_chunks[i-1]

        if i < len(child_chunks)-1 and child_chunks[i+1].metadata['Header 1'] == current_chunk.metadata['Header 1']:
            next_chunk = child_chunks[i+1]

        chunks_to_combine = []
        if previous_chunk:
            chunks_to_combine.append(previous_chunk.page_content)
        chunks_to_combine.append(current_chunk.page_content)
        if next_chunk:
            chunks_to_combine.append(next_chunk.page_content)

        combined_chunks = '\n\n'.join(chunks_to_combine)

        parent_id = (
            current_chunk.metadata.get('source') + '-' +
            current_chunk.metadata.get('Header 1', 'MISSING') + "-" +
            current_chunk.metadata.get('Header 2', 'MISSING') + '-' + str(i)
        )

        parent_doc = Document(
            page_content=combined_chunks,
            metadata={
                "parent_id": parent_id,
                "chapter_id": current_chunk.metadata.get('Header 1', ''),
                "section_id": current_chunk.metadata.get('Header 2', ''),
                "chunk_type": "parent",
                "strategy": "parent_child",
                "source": current_chunk.metadata.get('source', ''),
                "chunk_index": i
            }
        )
        parent_chunks.append(parent_doc)

    return parent_chunks


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_json(path, data):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def store_parent_chunks(parent_chunks, data_path="Parent_chunk_data.json"):
    file_exists = os.path.exists(data_path)
    existing_data = load_json(data_path) if file_exists else {}

    incoming_map = {doc.metadata['parent_id']: doc for doc in parent_chunks}
    new_parentid = set(incoming_map.keys()) - set(existing_data.keys())

    new_docs = {
        h: {
            "page_content": incoming_map[h].page_content,
            "metadata": incoming_map[h].metadata,
            "parent_id": h
        }
        for h in new_parentid
    }

    all_data = {**existing_data, **new_docs}
    save_json(data_path, all_data)

    return {"new_added": len(new_docs), "total": len(all_data)}


def load_parent_store(path="Parent_chunk_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_parent_chunks(child_docs, parent_store):
    parent_docs = []
    seen = set()

    for doc in child_docs:
        parent_id = doc.metadata.get('parent_id')
        if parent_id and parent_id not in seen:
            parent_chunk = parent_store.get(parent_id)
            if parent_chunk:
                parent_docs.append(Document(
                    page_content=parent_chunk['page_content'],
                    metadata=parent_chunk['metadata']
                ))
                seen.add(parent_id)

    return parent_docs
