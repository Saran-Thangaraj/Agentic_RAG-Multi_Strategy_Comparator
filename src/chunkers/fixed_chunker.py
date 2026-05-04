import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion import validate_page


def chunk(pages: list) -> list:
    fixed_size = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n###", "\n##", "\n#", "\n", " "]
    )

    fixed_chunks = []
    previous_page = 'Introduction'

    for d in pages:
        is_valid, reason = validate_page(d.page_content)
        if not is_valid:
            continue

        if "page" in d.metadata:
            d.metadata['page'] += 1

        page_header = d.page_content.split('\n')[0].strip()
        page_header = re.findall(
            r'^(Chapter \d+:.*|Appendix [A-Z]:.*)$',
            page_header, re.MULTILINE
        )

        result = page_header[0] if page_header else 'Unknown'

        if result != 'Unknown':
            d.metadata['page_chapter'] = result
            previous_page = result
        else:
            d.metadata['page_chapter'] = previous_page

        d.metadata['strategy'] = 'Fixed Size'

        chunks = fixed_size.split_documents([d])
        fixed_chunks.extend(chunks)

    return fixed_chunks
