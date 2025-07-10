# 📁 data_processing.py
# Functions for PDF discovery, text extraction, and document chunking.

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def get_file_hash(file_path: Path) -> str:
    """Calculates the SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""

def extract_text_from_pdf(pdf_path: Path) -> List[Document]:
    """Extracts text from each page of a PDF and returns a list of Documents."""
    documents = []
    logger.info(f"Processing PDF: {pdf_path.name}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            proceeding_match = pdf_path.parent.name
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    metadata = {
                        "source": pdf_path.name,
                        "proceeding": proceeding_match,
                        "page": i + 1,
                        "file_path": str(pdf_path),
                        "last_modified": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
                    }
                    documents.append(Document(page_content=text, metadata=metadata))
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
    logger.info(f"Extracted {len(documents)} pages from {pdf_path.name}")
    return documents

def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split documents into chunks with overlap"""
    if not documents:
        logger.warning("No documents to chunk")
        return []
    logger.info(f"Chunking {len(documents)} documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunked_docs = splitter.split_documents(documents)
    for i, doc in enumerate(chunked_docs):
        doc.metadata["chunk_id"] = i
        doc.metadata["chunk_length"] = len(doc.page_content)
    logger.info(f"Created {len(chunked_docs)} chunks")
    return chunked_docs
