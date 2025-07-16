#!/usr/bin/env python3
# Test URL-based processing with a sample CPUC document

import logging
import requests
from bs4 import BeautifulSoup
from data_processing import extract_and_chunk_with_docling_url, validate_pdf_url

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_pdf_url_from_page(page_url: str) -> str:
    """Extract the direct PDF URL from a CPUC document page"""
    try:
        response = requests.get(page_url, timeout=30)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find PDF download link
        result_td = soup.find("td", class_="ResultLinkTD")
        if result_td:
            pdf_link = result_td.find("a", string="PDF")
            if pdf_link and pdf_link.get("href"):
                pdf_url = "https://docs.cpuc.ca.gov" + pdf_link["href"]
                return pdf_url
    except Exception as e:
        logger.error(f"Failed to extract PDF URL from {page_url}: {e}")
    
    return None

def test_url_processing():
    """Test the URL-based processing pipeline with a sample document"""
    
    # Test with a public ArXiv PDF for reliable testing (CPUC site may be slow/restricted)
    test_pdf_url = "https://arxiv.org/pdf/2408.09869"
    
    logger.info(f"🧪 Testing URL processing with ArXiv PDF: {test_pdf_url}")
    
    # Step 1: Validate the PDF URL
    logger.info("Step 1: Validating PDF URL...")
    if not validate_pdf_url(test_pdf_url):
        logger.error("❌ PDF URL validation failed")
        return False
    
    logger.info("✅ PDF URL validation passed")
    
    # Step 2: Process the PDF using Docling
    logger.info("Step 2: Processing PDF with Docling...")
    try:
        documents = extract_and_chunk_with_docling_url(
            test_pdf_url, 
            "Docling Research Paper"
        )
        
        if documents:
            logger.info(f"✅ Successfully processed PDF:")
            logger.info(f"   - Extracted {len(documents)} chunks")
            logger.info(f"   - Sample chunk length: {len(documents[0].page_content) if documents else 0}")
            logger.info(f"   - Document type: {documents[0].metadata.get('document_type', 'Unknown') if documents else 'N/A'}")
            logger.info(f"   - Proceeding: {documents[0].metadata.get('proceeding_number', 'Unknown') if documents else 'N/A'}")
            
            # Show sample metadata
            if documents:
                sample_metadata = documents[0].metadata
                logger.info("Sample metadata:")
                for key, value in sample_metadata.items():
                    logger.info(f"   {key}: {value}")
            
            # Show sample content
            if documents:
                logger.info(f"Sample content (first 200 chars):")
                logger.info(f"   {documents[0].page_content[:200]}...")
            
            return True
        else:
            logger.error("❌ No chunks extracted from PDF")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to process PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cpuc_url_extraction():
    """Test CPUC URL extraction separately with a longer timeout"""
    
    test_page_url = "https://docs.cpuc.ca.gov/SearchRes.aspx?DocFormat=ALL&DocID=571985189"
    
    logger.info(f"🔍 Testing CPUC URL extraction with: {test_page_url}")
    
    try:
        response = requests.get(test_page_url, timeout=60)  # Longer timeout
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find PDF download link
        result_td = soup.find("td", class_="ResultLinkTD")
        if result_td:
            pdf_link = result_td.find("a", string="PDF")
            if pdf_link and pdf_link.get("href"):
                pdf_url = "https://docs.cpuc.ca.gov" + pdf_link["href"]
                logger.info(f"✅ Successfully extracted PDF URL: {pdf_url}")
                return pdf_url
            else:
                logger.warning("⚠️ PDF link not found in ResultLinkTD")
        else:
            logger.warning("⚠️ ResultLinkTD not found")
            
        # Debug: print page structure
        logger.info("Page structure:")
        for td in soup.find_all("td", class_=True):
            logger.info(f"   Found TD with class: {td.get('class')}")
            
    except Exception as e:
        logger.error(f"❌ CPUC URL extraction failed: {e}")
    
    return None

if __name__ == "__main__":
    # Test URL processing with ArXiv PDF
    success = test_url_processing()
    if success:
        logger.info("🎉 URL processing test completed successfully!")
    else:
        logger.error("💥 URL processing test failed!")
    
    # Also test CPUC URL extraction
    logger.info("\n" + "="*50)
    cpuc_pdf_url = test_cpuc_url_extraction()
    if cpuc_pdf_url:
        logger.info("🎉 CPUC URL extraction test completed successfully!")
    else:
        logger.error("💥 CPUC URL extraction test failed!")