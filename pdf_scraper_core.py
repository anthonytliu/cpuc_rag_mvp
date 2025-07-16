#!/usr/bin/env python3
"""
Core PDF Scraper Functions for CPUC RAG System

This module provides the core functionality for checking and downloading
new PDFs from CPUC proceedings, designed to be used by both the scheduler
and manual scraping operations.

Author: Claude Code
"""

import hashlib
import json
import logging
import os
import re
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)

# Configuration
PROCEEDING_LIST = ["R2207005"]
BASE_URL = "https://apps.cpuc.ca.gov/apex/f?p=401:1:0"
DOWNLOAD_DIR = Path("./cpuc_csvs")
# PDF_BASE_DIR = Path("./cpuc_pdfs")  # DEPRECATED - moved to URL-based processing


class CPUCPDFScraper:
    """Core PDF scraper for CPUC proceedings"""
    
    def __init__(self, headless: bool = True):
        """
        Initialize the PDF scraper
        
        Args:
            headless: Whether to run Chrome in headless mode
        """
        self.headless = headless
        self.driver = None
        self.download_dir = DOWNLOAD_DIR
        # self.pdf_base_dir = PDF_BASE_DIR  # DEPRECATED - moved to URL-based processing
        
        # Ensure directories exist
        self.download_dir.mkdir(exist_ok=True)
        # self.pdf_base_dir.mkdir(exist_ok=True)  # DEPRECATED
        
        logger.info("CPUC PDF Scraper initialized")
    
    def _setup_driver(self):
        """Set up Chrome WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True
        })
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Chrome WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {e}")
            raise
    
    def _cleanup_driver(self):
        """Clean up Chrome WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Chrome WebDriver closed")
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {e}")
            finally:
                self.driver = None
    
    def check_for_new_pdfs(self, proceeding: str) -> Tuple[List[str], Dict]:
        """
        Check for new PDFs for a given proceeding
        
        Args:
            proceeding: The proceeding number (e.g., "R2207005")
            
        Returns:
            Tuple of (new_urls, metadata)
        """
        logger.info(f"Checking for new PDFs in proceeding {proceeding}")
        
        try:
            # Load existing download history
            download_history = self._load_download_history(proceeding)
            
            # Get current URLs from the website
            current_urls = self._get_current_pdf_urls(proceeding)
            
            # Compare with download history to find new URLs
            new_urls, updated_urls = self._compare_urls(current_urls, download_history)
            
            # Combine new and updated URLs
            all_new_urls = new_urls + updated_urls
            
            # Extract metadata
            metadata = self._extract_proceeding_metadata(proceeding)
            
            logger.info(f"Found {len(all_new_urls)} new/updated PDFs for {proceeding}")
            
            return all_new_urls, metadata
            
        except Exception as e:
            logger.error(f"Error checking for new PDFs in {proceeding}: {e}")
            raise
    
    def _get_current_pdf_urls(self, proceeding: str) -> List[str]:
        """Get current PDF URLs from the CPUC website"""
        try:
            # First, try to get URLs from existing CSV
            csv_file = self.download_dir / f"{proceeding.lower()}_resultCSV.csv"
            if csv_file.exists():
                # Check if CSV is recent (less than 1 hour old)
                csv_age = datetime.now() - datetime.fromtimestamp(csv_file.stat().st_mtime)
                if csv_age.total_seconds() < 3600:  # 1 hour
                    logger.info(f"Using existing CSV file: {csv_file}")
                    return self._extract_pdf_urls_from_csv(str(csv_file))
            
            # If no recent CSV, scrape fresh data
            logger.info(f"Scraping fresh data for proceeding {proceeding}")
            return self._scrape_fresh_urls(proceeding)
            
        except Exception as e:
            logger.error(f"Error getting current PDF URLs for {proceeding}: {e}")
            raise
    
    def _scrape_fresh_urls(self, proceeding: str) -> List[str]:
        """Scrape fresh PDF URLs from the CPUC website"""
        self._setup_driver()
        
        try:
            self.driver.get(BASE_URL)
            wait = WebDriverWait(self.driver, 10)
            
            # Search for proceeding
            logger.info(f"Searching for proceeding {proceeding}...")
            input_box = wait.until(ec.presence_of_element_located((By.ID, "P1_PROCEEDING_NUM")))
            input_box.clear()
            input_box.send_keys(proceeding)
            self.driver.find_element(By.ID, "P1_SEARCH").click()
            
            # Click result link
            logger.info("Clicking result link...")
            wait.until(ec.presence_of_element_located(
                (By.XPATH, f"//td[@headers='PROCEEDING_STATUS_DESC']/a[contains(@href, '{proceeding}')]")
            )).click()
            
            # Navigate to Documents tab
            logger.info("Navigating to Documents tab...")
            wait.until(ec.presence_of_element_located((By.LINK_TEXT, "Documents"))).click()
            time.sleep(2)
            
            # Download CSV
            logger.info("Downloading CSV...")
            download_btn = wait.until(ec.presence_of_element_located((By.XPATH, "//input[@value='Download']")))
            download_btn.click()
            
            # Wait for download to complete
            self._wait_for_download()
            
            # Find and rename the CSV file
            csv_files = sorted(
                [f for f in self.download_dir.iterdir() if f.suffix == '.csv'],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if csv_files:
                latest_csv = csv_files[0]
                new_csv_name = f"{proceeding.lower()}_resultCSV.csv"
                new_csv_path = self.download_dir / new_csv_name
                
                latest_csv.rename(new_csv_path)
                logger.info(f"Downloaded and renamed CSV to: {new_csv_name}")
                
                # Extract URLs from the CSV
                return self._extract_pdf_urls_from_csv(str(new_csv_path))
            else:
                logger.warning("No CSV file found after download")
                return []
                
        finally:
            self._cleanup_driver()
    
    def _extract_pdf_urls_from_csv(self, csv_file_path: str) -> List[str]:
        """Extract PDF URLs from CSV file"""
        pdf_urls = []
        
        try:
            df = pd.read_csv(csv_file_path)
            logger.info(f"Processing CSV with {len(df)} rows...")
            
            for index, row in df.iterrows():
                doc_type_content = row.get("Document Type")
                if isinstance(doc_type_content, str) and "<a href=" in doc_type_content:
                    try:
                        soup = BeautifulSoup(doc_type_content, "html.parser")
                        link = soup.find("a")
                        if link and link.get("href"):
                            pdf_urls.append(link["href"])
                    except Exception as e:
                        logger.warning(f"Error parsing HTML in CSV row {index}: {e}")
                        
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_file_path}: {e}")
            raise
        
        logger.info(f"Extracted {len(pdf_urls)} PDF URLs from CSV")
        return pdf_urls
    
    def _extract_proceeding_metadata(self, proceeding: str) -> Dict:
        """DEPRECATED: Extract metadata for a proceeding - moved to URL-based processing"""
        # metadata_file = self.pdf_base_dir / proceeding / "metadata.json"
        logger.warning("_extract_proceeding_metadata is deprecated - moved to URL-based processing")
        return {"proceeding": proceeding, "last_updated": datetime.now().isoformat()}
    
    def _load_download_history(self, proceeding: str) -> Dict:
        """Load download history for a proceeding"""
        history_file = self.download_dir / f"{proceeding.lower()}_download_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading download history: {e}")
        
        return {}
    
    def _save_download_history(self, proceeding: str, history: Dict):
        """Save download history for a proceeding"""
        history_file = self.download_dir / f"{proceeding.lower()}_download_history.json"
        
        try:
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving download history: {e}")
    
    def _compare_urls(self, current_urls: List[str], download_history: Dict) -> Tuple[List[str], List[str]]:
        """Compare current URLs with download history"""
        new_urls = []
        updated_urls = []
        
        for url in current_urls:
            url_hash = self._create_url_hash(url)
            
            if url_hash not in download_history:
                new_urls.append(url)
            else:
                # Check if we need to re-download (could add date checking here)
                existing_entry = download_history[url_hash]
                if existing_entry.get('status') == 'error':
                    updated_urls.append(url)
        
        return new_urls, updated_urls
    
    def download_pdfs(self, proceeding: str, urls: List[str]) -> int:
        """DEPRECATED: Download PDFs from URLs - moved to URL-based processing"""
        logger.warning("download_pdfs is deprecated - moved to URL-based processing")
        if not urls:
            logger.info("No URLs to download")
            return 0
        
        # DEPRECATED: No longer downloading PDFs locally
        # download_count = 0
        # pdf_dir = self.pdf_base_dir / proceeding
        # pdf_dir.mkdir(exist_ok=True)
        return 0
    
    def _update_download_history(self, proceeding: str, url: str, filename: str, status: str):
        """Update download history with new entry"""
        history = self._load_download_history(proceeding)
        url_hash = self._create_url_hash(url)
        
        history[url_hash] = {
            "url": url,
            "filename": filename,
            "status": status,
            "download_date": datetime.now().isoformat(),
            "last_checked": datetime.now().isoformat()
        }
        
        self._save_download_history(proceeding, history)
    
    def _wait_for_download(self, timeout: int = 30):
        """Wait for download to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for .crdownload files (Chrome partial downloads)
            crdownload_files = list(self.download_dir.glob("*.crdownload"))
            if not crdownload_files:
                return True
            
            time.sleep(1)
        
        logger.warning("Download timeout reached")
        return False
    
    @staticmethod
    def _create_url_hash(url: str) -> str:
        """Create a hash of the URL for unique identification"""
        return hashlib.md5(url.encode()).hexdigest()
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename to remove invalid characters"""
        return re.sub(r'[<>:"/\\|?*]', '_', filename)


def check_for_new_pdfs(proceeding: str = "R2207005", headless: bool = True) -> Tuple[List[str], Dict]:
    """
    Convenience function to check for new PDFs
    
    Args:
        proceeding: The proceeding number
        headless: Whether to run in headless mode
        
    Returns:
        Tuple of (new_urls, metadata)
    """
    scraper = CPUCPDFScraper(headless=headless)
    return scraper.check_for_new_pdfs(proceeding)


def download_new_pdfs(proceeding: str = "R2207005", headless: bool = True) -> int:
    """
    Convenience function to check for and download new PDFs
    
    Args:
        proceeding: The proceeding number
        headless: Whether to run in headless mode
        
    Returns:
        Number of PDFs downloaded
    """
    scraper = CPUCPDFScraper(headless=headless)
    
    # Check for new PDFs
    new_urls, metadata = scraper.check_for_new_pdfs(proceeding)
    
    # Download them
    if new_urls:
        return scraper.download_pdfs(proceeding, new_urls)
    else:
        logger.info(f"No new PDFs found for {proceeding}")
        return 0