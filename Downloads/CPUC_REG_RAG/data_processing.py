# 📁 data_processing.py
# Functions for PDF discovery, text extraction, and document chunking.

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import re

import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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


def chunk_documents(self, documents: List[Document]) -> List[Document]:
    """Split documents into chunks with overlap"""
    if not documents:
        logger.warning("No documents to chunk")
        return []

    logger.info(f"Chunking {len(documents)} documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=self.chunk_size,
        chunk_overlap=self.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    chunked_docs = splitter.split_documents(documents)

    # Add chunk metadata
    for i, doc in enumerate(chunked_docs):
        doc.metadata["chunk_id"] = i
        doc.metadata["chunk_length"] = len(doc.page_content)

    logger.info(f"Created {len(chunked_docs)} chunks")
    return chunked_docs


def _create_hierarchical_chunks(self, documents: List[Document]) -> List[Document]:
    """Create hierarchical chunks that respect regulatory document structure"""

    hierarchical_chunks = []

    for doc in documents:
        text = doc.page_content

        # Identify regulatory sections
        sections = _identify_regulatory_sections(text)

        if sections:
            # Create parent chunks for complete sections
            for section in sections:
                parent_chunk = Document(
                    page_content=section['content'],
                    metadata={
                        **doc.metadata,
                        'chunk_type': 'parent',
                        'section_type': section['type'],
                        'section_title': section['title']
                    }
                )
                hierarchical_chunks.append(parent_chunk)

                # Create child chunks within section if it's long
                if len(section['content']) > self.chunk_size:
                    child_chunks = self._create_child_chunks(section['content'], doc.metadata, section)
                    hierarchical_chunks.extend(child_chunks)
        else:
            # Fall back to regular chunking
            regular_chunks = self.chunk_documents([doc])
            hierarchical_chunks.extend(regular_chunks)

    return hierarchical_chunks


def _create_child_chunks(self, section_content: str, base_metadata: Dict, section_info: Dict) -> List[Document]:
    """Create child chunks within a regulatory section"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=self.chunk_size,
        chunk_overlap=self.chunk_overlap,
        length_function=len
    )

    chunks = splitter.split_text(section_content)
    child_documents = []

    for i, chunk in enumerate(chunks):
        child_doc = Document(
            page_content=chunk,
            metadata={
                **base_metadata,
                'chunk_type': 'child',
                'parent_section': section_info['type'],
                'parent_title': section_info['title'],
                'child_index': i
            }
        )
        child_documents.append(child_doc)

    return child_documents


def process_proceeding(self, proceeding_dir: Path) -> List[Document]:
    """Process all PDFs in a proceeding directory"""
    all_documents = []

    # FIXED: Handle both directory structure and single directory
    if proceeding_dir.is_dir():
        pdf_files = list(proceeding_dir.glob("*.pdf"))
    else:
        # If base_dir is not a directory, check if it's a file pattern
        pdf_files = list(self.base_dir.glob("*.pdf"))

    logger.info(f"Processing {len(pdf_files)} PDFs in {proceeding_dir}")

    if not pdf_files:
        logger.warning(f"No PDF files found in {proceeding_dir}")
        return []

    for pdf_path in pdf_files:
        if self._needs_update(pdf_path):
            logger.info(f"Processing updated file: {pdf_path.name}")
            documents = extract_text_chunks(pdf_path)
            if documents:
                all_documents.extend(documents)
                # Update hash after successful processing
                self.doc_hashes[str(pdf_path)] = _get_file_hash(pdf_path)
            else:
                logger.warning(f"No documents extracted from {pdf_path.name}")
        else:
            logger.info(f"Skipping unchanged file: {pdf_path.name}")

    logger.info(f"Total documents processed: {len(all_documents)}")
    return all_documents


def _identify_regulatory_sections(text: str) -> List[Dict]:
    """Identify regulatory sections in document text"""
    sections = []

    section_patterns = {
        'order': r'(?:IT IS ORDERED|ORDERS?):?\s*\n(.*?)(?=\n\n|\n[A-Z]{2,}|\Z)',
        'finding': r'(?:FINDINGS?):?\s*\n(.*?)(?=\n\n|\n[A-Z]{2,}|\Z)',
        'rule': r'(?:Rule|Section)\s+(\d+[.\d]*)\s*[-:]?\s*(.*?)(?=\n(?:Rule|Section)|\n\n|\Z)',
        'definition': r'(?:Definition|means|shall mean)\s*(.*?)(?=\n\n|\n[A-Z]|\Z)',
        'requirement': r'(?:shall|must|required to|obligated to)\s*(.*?)(?=\n\n|\n[A-Z]|\Z)'
    }

    for section_type, pattern in section_patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            content = match.group(0).strip()
            if len(content) > 100:  # Only include substantial sections
                sections.append({
                    'type': section_type,
                    'title': content.split('\n')[0][:100],
                    'content': content,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })

    # Sort by position in document
    sections.sort(key=lambda x: x['start_pos'])
    return sections


def extract_text_chunks(pdf_path: Path) -> List[Document]:
    """Extract text from PDF with enhanced metadata and error handling"""
    documents = []

    logger.info(f"Processing PDF: {pdf_path.name}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"PDF has {len(pdf.pages)} pages")

            # Extract metadata from first page or filename
            proceeding_match = pdf_path.parent.name

            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        # Enhanced metadata
                        metadata = {
                            "source": pdf_path.name,
                            "proceeding": proceeding_match,
                            "page": i + 1,
                            "file_path": str(pdf_path),
                            "last_modified": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
                        }

                        documents.append(Document(
                            page_content=text,
                            metadata=metadata
                        ))

                        # Log progress every 10 pages
                        if (i + 1) % 10 == 0:
                            logger.info(f"Processed {i + 1} pages from {pdf_path.name}")
                    else:
                        logger.warning(f"No text extracted from page {i + 1} of {pdf_path.name}")

                except Exception as e:
                    logger.error(f"Error processing page {i + 1} of {pdf_path.name}: {e}")
                    continue

            logger.info(f"Successfully extracted {len(documents)} pages from {pdf_path.name}")

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")

    return documents


def _get_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""
