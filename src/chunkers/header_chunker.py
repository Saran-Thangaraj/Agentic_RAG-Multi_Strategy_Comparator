import re
from langchain_text_splitters import MarkdownHeaderTextSplitter
from src.ingestion import validate_page


def chunk(pages: list) -> list:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#=", "Header 1"),
        ("##=", "Header 2")
    ])

    header_texts = []

    for d in pages:
        is_valid, reason = validate_page(d.page_content)
        if not is_valid:
            continue

        lines = d.page_content.split("\n")

        for line in lines:
            line = line.strip()

            if '#' in line:
                header_texts.append(line)
                continue

            if line.startswith("Chapter") or line.startswith("Appendix"):
                header_texts.append(f"#= {line}\n")
            elif re.match(r"^\d+\.\d+", line):
                header_texts.append(f"##= {line}\n")
            elif re.match(r"^[A-Z]\.\d+", line):
                header_texts.append(f"##= {line}\n")
            else:
                header_texts.append(line)

    header_text = "\n".join(header_texts)
    all_chunks = splitter.split_text(header_text)

    source = pages[-1].metadata.get('source', '') if pages else ''

    for idx, chunk in enumerate(all_chunks):
        chunk.metadata['strategy'] = 'header'
        chunk.metadata['source'] = source
        chunk.metadata['Chunk_id'] = f"header_{idx}"
        chunk.metadata['total_chunks'] = len(all_chunks)

        header = chunk.metadata.get("Header 2") or chunk.metadata.get("Header 1") or ""
        chunk.page_content = header + "-" + chunk.page_content 

        chunk.metadata['header_id'] = (source + "-" + chunk.metadata.get("Header 1", "MISSING") + "-" +
                                       chunk.metadata.get("Header 2", "MISSING") + "-" +str(idx))

    return all_chunks
