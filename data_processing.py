# 📁 data_processing.py
# Functions for PDF discovery, text extraction, and document chunking.

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

from utils import extract_text_from_image
from config import CHUNK_SIZE, CHUNK_OVERLAP

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
    """
    Extracts text and tables from a PDF, along with OCR of images.
    Returns a list of Documents, each representing a chunk of text or a table.
    """
    documents = []
    logger.info(f"Processing PDF for text, tables, and images (OCR): {pdf_path.name}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            proceeding_match = pdf_path.parent.name
            for page_num, page in enumerate(pdf.pages):
                # 1. Extract Tables (same logic as before)
                tables = page.find_tables()
                for table_index, table in enumerate(tables):
                    try:
                        table_data = table.extract()
                        if table_data:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            table_content = df.to_markdown(index=False)
                            metadata = _create_metadata(pdf_path, page_num + 1, content_type="table", part=table_index)
                            documents.append(Document(page_content=table_content, metadata=metadata))
                    except Exception as e:
                        logger.warning(f"Error extracting table: {e} on page {page_num + 1} of {pdf_path.name}")

                # 2. Extract Text (using a more robust method, better for headings)
                text_content = page.extract_text(x_tolerance=3, y_tolerance=3)  # Increased tolerance

                # 3. Extract text from images via OCR
                for img_idx, img in enumerate(page.images):
                    try:
                        if "stream" in img and hasattr(img["stream"], "get_data"):
                            image_bytes = img["stream"].get_data()
                            if not image_bytes:
                                continue  # Skip if the image data is empty
                            ocr_text = extract_text_from_image(image_bytes)
                            if ocr_text:
                                content = f"Text from image on page {page_num}:\n\n{ocr_text}"
                                metadata = _create_metadata(pdf_path, page_num, "image_ocr", img_idx)
                                documents.append(Document(page_content=content, metadata=metadata))
                    except Exception as e:
                        logger.warning(f"Could not process an image on page {page_num} of {pdf_path.name}: {e}")

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
    logger.info(f"Extracted {len(documents)} text and table chunks from {pdf_path.name}")
    return documents


def _identify_and_process_headings(text: str) -> str:
    """Identifies and highlights headings in the text (optional)"""
    # Add some basic heading patterns (adjust as needed)
    heading_patterns = [
        (r'^\s*\d+(\.\d+)*\s+(.*)$', '## '),  # Headings like "1.2.3 Title"
        (r'^(.*?\n){0,2}(CHAPTER|SECTION)\s+\d+\s+.*', '### '),  # Chapter or Section
        (r'^\s*[A-Z][A-Z]+\s*$', '## ')  # ALL CAPS headings
        # Add more patterns as needed
    ]

    processed_text = text
    for pattern, replacement in heading_patterns:
        processed_text = re.sub(pattern, r'\n' + replacement + r'\2\n', processed_text, flags=re.MULTILINE)
    return processed_text


def _create_metadata(pdf_path: Path, page_num: int, content_type: str, part: int = 0) -> dict:
    """Helper to create consistent metadata dictionaries, including the chunk_id."""
    chunk_id = f"{pdf_path.name}_p{page_num}_{content_type}_{part}"
    return {
        "source": pdf_path.name,
        "proceeding": pdf_path.parent.name if pdf_path.parent.name != '.' else 'Unknown',
        "page": page_num,
        "file_path": str(pdf_path),
        "last_modified": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat(),
        "chunk_id": chunk_id,
        "relevance_score": 0.0  # Initialize the relevance score.
    }


def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Splits a list of Documents into smaller chunks."""
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
        doc.metadata["chunk_id"] = f"{doc.metadata.get('source', 'unknown')}_chunk_{i}"
        doc.metadata["chunk_length"] = len(doc.page_content)
    logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents.")
    return chunked_docs
