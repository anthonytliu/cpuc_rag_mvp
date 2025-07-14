# 📁 data_processing.py
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import DocItem, TableItem
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

# Initialize the converter once. This object can be reused.
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(backend=DoclingParseV4DocumentBackend)
    }
)


def extract_and_chunk_with_docling(pdf_path: Path) -> List[Document]:
    """
    Processes a PDF using the Docling DocumentConverter and traverses the
    document tree to extract structured elements as LangChain Documents.
    """
    logger.info(f"Processing with correct Docling workflow: {pdf_path.name}")
    langchain_documents = []

    try:
        # 1. Use the converter to process the file into a DoclingDocument object
        conv_results = doc_converter.convert_all([pdf_path], raises_on_error=False)
        conv_res = next(iter(conv_results), None)

        if not conv_res or conv_res.status == ConversionStatus.FAILURE:
            logger.error(f"Docling failed to convert document: {pdf_path.name}")
            return []

        docling_doc = conv_res.document

        # 2. Use the iterate_items() method to traverse the document tree
        # This is the correct way to access all content elements in reading order.
        logger.info(f"Traversing document tree for {pdf_path.name}...")

        # We iterate through all content items (text, tables, etc.)
        for item, level in docling_doc.iterate_items(with_groups=False):

            # Skip items that are not DocItem (e.g., pure GroupItems if with_groups=True)
            if not isinstance(item, DocItem):
                continue

            content = ""
            # For tables, get the clean Markdown representation
            if isinstance(item, TableItem):
                content = item.export_to_markdown(doc=docling_doc)
            # For all other text-based items, get the text content
            elif hasattr(item, 'text'):
                content = item.text

            if content and content.strip():
                # Get the primary provenance (location) information
                page_num = item.prov[0].page_no + 1 if item.prov else 0

                # Create rich metadata
                metadata = {
                    "source": pdf_path.name,
                    "page": page_num,
                    "content_type": item.label.value,  # e.g., 'paragraph', 'title', 'table'
                    "chunk_id": f"{pdf_path.name}_{item.get_ref().cref.replace('#/', '').replace('/', '_')}",
                    # Create a unique ID
                    "file_path": str(pdf_path),
                    "last_modified": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat(),
                }
                langchain_documents.append(Document(page_content=content, metadata=metadata))

    except Exception as e:
        logger.error(f"An unexpected error occurred during Docling processing for {pdf_path.name}: {e}", exc_info=True)
        return []

    logger.info(
        f"Successfully extracted {len(langchain_documents)} structured chunks from {pdf_path.name} via Docling.")
    return langchain_documents


# The old chunking logic is no longer needed as Docling provides semantic chunks.
# The old get_file_hash can remain if you still use it in rag_core.py for the _needs_update check.

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
