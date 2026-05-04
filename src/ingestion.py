import re
import os
from langchain_community.document_loaders import PyPDFLoader


def validate_page(d: str):
    if d.strip() == '':
        return False, "blank page"
    elif len(d.strip()) < 50:
        return False, "content too short"
    elif d.startswith('Table of Contents'):
        return False, "table of contents"
    elif d.startswith('LangChain & RAG\n Complete Technical Documentation'):
        return False, 'Introduction'
    return True, "valid page"


def load_pdf(path: str) -> list:
    loader = PyPDFLoader(path)
    pages = list(loader.lazy_load())
    
    #Normalize source to just filename
    filename = os.path.basename(path)
    for page in pages:
        page.metadata['source'] = filename
    return pages
