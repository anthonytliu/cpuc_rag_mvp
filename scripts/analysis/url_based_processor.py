# 📁 url_based_processor.py
# URL-based PDF processor for CPUC documents using Docling

import json
import logging
import os
import re
from datetime import datetime
from typing import List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup

from rag_core import CPUCRAGSystem
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PROCEEDING_LIST = ["R2207005"]
DOWNLOAD_DIR = os.path.abspath("./cpuc_csvs")
URL_STORAGE_FILE = os.path.join(DOWNLOAD_DIR, "cpuc_pdf_urls.json")

def sanitize_filename(filename_to_sanitize):
    """Sanitize filename to remove invalid characters"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename_to_sanitize)

def extract_pdf_urls_from_csv(csv_file_path):
    """Extract PDF URLs from CSV file"""
    pdf_url_list = []
    try:
        dataframe = pd.read_csv(csv_file_path)
        logger.info(f"📊 Processing CSV with {len(dataframe)} rows...")

        for row_index, row_data in dataframe.iterrows():
            doc_type_cell_content = row_data.get("Document Type")
            if isinstance(doc_type_cell_content, str) and "<a href=" in doc_type_cell_content:
                # Parse the HTML to extract the actual URL
                try:
                    cell_soup = BeautifulSoup(doc_type_cell_content, "html.parser")
                    link_element = cell_soup.find("a")
                    if link_element and link_element.get("href"):
                        document_url = link_element["href"]
                        pdf_url_list.append(document_url)
                except Exception as parse_error:
                    logger.warning(f"Error parsing HTML in cell: {parse_error}")

    except Exception as csv_error:
        logger.error(f"Error reading CSV: {csv_error}")

    return pdf_url_list

def extract_pdf_urls_and_metadata(document_urls: List[str]) -> List[Dict[str, str]]:
    """
    Extract PDF URLs and metadata from CPUC document pages.
    
    This function processes document page URLs to extract the actual PDF URLs
    and document metadata without downloading the PDFs.
    
    Args:
        document_urls: List of CPUC document page URLs
        
    Returns:
        List of dictionaries with PDF URL, title, and metadata
    """
    pdf_data = []
    
    for i, doc_url in enumerate(document_urls, 1):
        logger.info(f"🔗 Processing document {i}/{len(document_urls)}: {doc_url}")
        
        try:
            doc_page = requests.get(doc_url, timeout=30)
            doc_soup = BeautifulSoup(doc_page.text, "html.parser")

            # Get the title for metadata
            title_td = doc_soup.find("td", class_="ResultTitleTD")
            if title_td:
                title_text = title_td.get_text(strip=True)
                # Extract just the document ID and description (before "Proceeding:")
                title_parts = title_text.split("Proceeding:")
                if title_parts:
                    title = title_parts[0].strip()
                    # Remove line breaks and extra spaces
                    title = re.sub(r'\s+', ' ', title)
                else:
                    title = f"document_{i}"
            else:
                title = f"document_{i}"

            # Find PDF download link
            result_td = doc_soup.find("td", class_="ResultLinkTD")
            if result_td:
                pdf_link = result_td.find("a", string="PDF")
                if pdf_link and pdf_link.get("href"):
                    pdf_url = "https://docs.cpuc.ca.gov" + pdf_link["href"]
                    
                    pdf_data.append({
                        'url': pdf_url,
                        'title': title,
                        'source_page': doc_url,
                        'extracted_date': datetime.now().isoformat()
                    })
                    
                    logger.info(f"✅ Extracted PDF URL: {title}")
                else:
                    logger.warning(f"⚠️ No PDF link found for {doc_url}")
            else:
                logger.warning(f"⚠️ No result link section found for {doc_url}")

        except Exception as e:
            logger.error(f"⚠️ Error processing {doc_url}: {e}")

    return pdf_data

def save_pdf_urls(pdf_data: List[Dict[str, str]], filename: str):
    """Save PDF URL data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(pdf_data, f, indent=2)
    logger.info(f"💾 Saved {len(pdf_data)} PDF URLs to {filename}")

def load_pdf_urls(filename: str) -> List[Dict[str, str]]:
    """Load PDF URL data from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def process_cpuc_documents_url_based():
    """
    Main function to process CPUC documents using URL-based approach.
    
    This function:
    1. Reads the CSV file to get document page URLs
    2. Extracts PDF URLs and metadata from document pages
    3. Processes PDFs directly from URLs using Docling
    4. Updates the vector store with URL-based processing
    """
    logger.info("🚀 Starting URL-based CPUC document processing...")
    
    for proceeding in PROCEEDING_LIST:
        logger.info(f"📋 Processing proceeding: {proceeding}")
        
        # Load existing URLs
        existing_urls = load_pdf_urls(URL_STORAGE_FILE)
        existing_pdf_urls = {item['url'] for item in existing_urls}
        
        # Read CSV file
        csv_file = os.path.join(DOWNLOAD_DIR, f"{proceeding.lower()}_resultCSV.csv")
        if not os.path.exists(csv_file):
            logger.error(f"CSV file not found: {csv_file}")
            continue
            
        # Extract document page URLs from CSV
        document_urls = extract_pdf_urls_from_csv(csv_file)
        logger.info(f"📄 Found {len(document_urls)} document URLs in CSV")
        
        # Extract PDF URLs and metadata
        logger.info("🔍 Extracting PDF URLs from document pages...")
        pdf_data = extract_pdf_urls_and_metadata(document_urls)
        
        # Filter for new URLs
        new_pdf_data = [item for item in pdf_data if item['url'] not in existing_pdf_urls]
        logger.info(f"🆕 Found {len(new_pdf_data)} new PDF URLs")
        
        # Combine with existing data
        all_pdf_data = existing_urls + new_pdf_data
        
        # Save updated URL data
        save_pdf_urls(all_pdf_data, URL_STORAGE_FILE)
        
        # Process with RAG system
        logger.info("🔄 Processing URLs with RAG system...")
        rag_system = CPUCRAGSystem()
        
        # Convert to the format expected by build_vector_store_from_urls
        url_list = [{'url': item['url'], 'title': item['title']} for item in all_pdf_data]
        
        # Build/update vector store from URLs
        rag_system.build_vector_store_from_urls(url_list)
        
        logger.info(f"✅ Completed processing for proceeding {proceeding}")

if __name__ == "__main__":
    process_cpuc_documents_url_based()