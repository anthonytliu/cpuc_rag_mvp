#!/usr/bin/env python3
"""
Simplified CPUC Scraper - Single-threaded, straightforward implementation

This scraper follows a simple 4-step process:
1. Create proceeding folder and fetch Documents.csv
2. Analyze CSV and scrape PDF metadata using PDF analysis
3. Google search for additional PDFs from cpuc.ca.gov
4. Real-time progress tracking with dynamic loading bar

Author: Claude Code
"""

import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil  # For setting process priority

import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

# PDF analysis libraries - using alternative approach
from io import BytesIO
import tempfile

logger = logging.getLogger(__name__)

class ProgressBar:
    """Simple progress bar for real-time updates"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.discovered_pdfs = 0
        self.scraped_pdfs = 0
        
    def update(self, step: int = None, discovered: int = None, scraped: int = None):
        """Update progress bar with current status"""
        if step is not None:
            self.current_step = step
        if discovered is not None:
            self.discovered_pdfs = discovered
        if scraped is not None:
            self.scraped_pdfs = scraped
            
        # Calculate percentage
        progress = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * progress // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Update display
        status = f"\r{self.description} |{bar}| {progress:.1f}% | Discovered: {self.discovered_pdfs} | Scraped: {self.scraped_pdfs}"
        print(status, end='', flush=True)
        
        if progress >= 100:
            print()  # New line when complete


class CPUCSimplifiedScraper:
    """
    Simplified CPUC document scraper - single-threaded, straightforward approach
    """
    
    def __init__(self, headless: bool = True):
        """Initialize the simplified scraper"""
        self.headless = headless
        self.driver = None
        
        # Set up download directory
        self.download_dir = Path.cwd().resolve()
        
        # Thread-safe lock for history file updates
        self.history_lock = threading.Lock()
        
        # Set process to low priority to avoid interfering with user activities
        self._set_background_priority()
        
        # Chrome options - configured for downloads
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Configure download preferences
        prefs = {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True  # Avoid opening PDFs in browser
        }
        self.chrome_options.add_experimental_option("prefs", prefs)
    
    def _setup_driver(self):
        """Initialize Chrome driver with background processing settings"""
        if self.driver is None:
            # Use background WebDriver settings for all operations to prevent desktop interference
            self.driver = self._create_background_webdriver()
            logger.info("Chrome driver initialized in background mode")
    
    def _create_background_webdriver(self):
        """
        Create a WebDriver instance optimized for background processing
        Ensures minimal interference with user's computer activities
        """
        # Create enhanced options for background processing
        background_options = Options()
        
        # Essential background settings
        background_options.add_argument("--headless")  # Run in background
        background_options.add_argument("--no-sandbox")
        background_options.add_argument("--disable-dev-shm-usage")
        background_options.add_argument("--disable-gpu")
        
        # Minimize resource usage and interference - ensure NO visible windows
        background_options.add_argument("--disable-web-security")
        background_options.add_argument("--disable-features=VizDisplayCompositor")
        background_options.add_argument("--disable-extensions")
        background_options.add_argument("--disable-plugins")
        background_options.add_argument("--disable-images")  # Don't load images for faster processing
        background_options.add_argument("--disable-background-timer-throttling")
        background_options.add_argument("--disable-renderer-backgrounding")
        background_options.add_argument("--disable-backgrounding-occluded-windows")
        background_options.add_argument("--disable-client-side-phishing-detection")
        background_options.add_argument("--disable-default-apps")
        background_options.add_argument("--disable-hang-monitor")
        background_options.add_argument("--disable-prompt-on-repost")
        background_options.add_argument("--disable-sync")
        background_options.add_argument("--disable-translate")
        background_options.add_argument("--disable-web-resources")
        background_options.add_argument("--hide-scrollbars")
        background_options.add_argument("--metrics-recording-only")
        background_options.add_argument("--mute-audio")
        background_options.add_argument("--no-first-run")
        background_options.add_argument("--safebrowsing-disable-auto-update")
        background_options.add_argument("--disable-background-networking")
        
        # Additional options to ensure complete background operation
        background_options.add_argument("--disable-infobars")
        background_options.add_argument("--disable-notifications")
        background_options.add_argument("--disable-popup-blocking")
        background_options.add_argument("--no-default-browser-check")
        background_options.add_argument("--disable-blink-features=AutomationControlled")
        background_options.add_argument("--remote-debugging-port=0")  # Disable remote debugging
        background_options.add_argument("--silent")  # Run silently
        
        # Set low priority process
        background_options.add_argument("--process-per-tab")
        background_options.add_argument("--max_old_space_size=512")  # Limit memory usage
        
        # Configure for minimal window interference
        background_options.add_argument("--window-position=-2000,-2000")  # Position off-screen
        background_options.add_argument("--window-size=800,600")  # Small window size
        
        # Configure download preferences (copy from main options)
        prefs = {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False,  # Disable for background processing
            "plugins.always_open_pdf_externally": True,
            "profile.default_content_setting_values": {
                "notifications": 2,  # Block notifications
                "media_stream": 2,   # Block media access
            }
        }
        background_options.add_experimental_option("prefs", prefs)
        
        # Create and return the background WebDriver
        return webdriver.Chrome(options=background_options)
    
    def _set_background_priority(self):
        """
        Set the current process to low priority to minimize interference with user activities
        """
        try:
            # Get current process
            current_process = psutil.Process()
            
            # Set to low priority (different methods for different OS)
            if os.name == 'nt':  # Windows
                current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:  # Unix-like (macOS, Linux)
                current_process.nice(10)  # Lower priority (higher nice value)
                
            logger.info("🔽 Set process to background priority to minimize user interference")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not set background priority: {e}")
    
    def _cleanup_driver(self):
        """Clean up Chrome driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def scrape_proceeding(self, proceeding: str) -> Dict:
        """
        Main entry point for scraping a proceeding
        
        Args:
            proceeding: Proceeding number (e.g., 'R2207005')
            
        Returns:
            Dictionary with scraping results
        """
        logger.info(f"Starting simplified scraping for proceeding {proceeding}")
        
        try:
            # Initialize progress tracking
            progress = ProgressBar(4, f"Scraping {proceeding}")
            
            # Step 1: Create proceeding folder and fetch CSV
            progress.update(1, 0, 0)
            proceeding_folder, csv_path = self._create_folder_and_fetch_csv(proceeding)
            
            # Initialize PDF history file early for incremental updates
            self._initialize_pdf_history_file(proceeding_folder)
            
            # Step 2: Analyze CSV and scrape PDF metadata
            progress.update(2, 0, 0)
            pdf_metadata = self._analyze_csv_and_scrape_pdfs(proceeding, csv_path, progress, proceeding_folder)
            
            # Step 3: Google search for additional PDFs
            progress.update(3, len(pdf_metadata), len(pdf_metadata))
            additional_pdfs = self._google_search_for_pdfs(proceeding, pdf_metadata, progress)
            
            # Step 4: Save results
            progress.update(4, len(pdf_metadata) + len(additional_pdfs), len(pdf_metadata) + len(additional_pdfs))
            self._save_scraped_history(proceeding_folder, pdf_metadata + additional_pdfs)
            
            logger.info(f"Scraping completed for {proceeding}: {len(pdf_metadata) + len(additional_pdfs)} PDFs found")
            
            return {
                'proceeding': proceeding,
                'total_pdfs': len(pdf_metadata) + len(additional_pdfs),
                'csv_pdfs': len(pdf_metadata),
                'google_pdfs': len(additional_pdfs),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error scraping proceeding {proceeding}: {e}")
            return {
                'proceeding': proceeding,
                'total_pdfs': 0,
                'status': 'error',
                'error': str(e)
            }
        finally:
            self._cleanup_driver()
    
    def _create_folder_and_fetch_csv(self, proceeding: str) -> tuple[Path, Path]:
        """
        Step 1: Create proceeding folder and fetch Documents.csv
        
        Returns:
            Tuple of (proceeding_folder_path, csv_file_path)
        """
        logger.info(f"Step 1: Creating folder and fetching CSV for {proceeding}")
        
        # Create cpuc_proceedings directory structure
        cpuc_proceedings_dir = Path("cpuc_proceedings")
        cpuc_proceedings_dir.mkdir(exist_ok=True)
        
        # Create proceeding folder within cpuc_proceedings
        proceeding_folder = cpuc_proceedings_dir / proceeding
        proceeding_folder.mkdir(exist_ok=True)
        logger.info(f"Created/verified proceeding folder: {proceeding_folder}")
        
        # Create documents subdirectory for CSV and related files
        documents_folder = proceeding_folder / "documents"
        documents_folder.mkdir(exist_ok=True)
        logger.info(f"Created/verified documents folder: {documents_folder}")
        
        # Fetch CSV from CPUC website
        csv_url = self._get_csv_download_url(proceeding)
        csv_path = documents_folder / f"{proceeding}_documents.csv"
        
        self._download_csv(csv_url, csv_path, proceeding)
        logger.info(f"Downloaded CSV to: {csv_path}")
        
        return proceeding_folder, csv_path
    
    def _initialize_pdf_history_file(self, proceeding_folder: Path):
        """
        Initialize PDF history JSON file early for incremental updates
        Creates empty file if it doesn't exist, preserves existing data if it does
        """
        history_file = proceeding_folder / f"{proceeding_folder.name}_scraped_pdf_history.json"
        
        if not history_file.exists():
            # Create empty JSON file
            with open(history_file, 'w') as f:
                json.dump({}, f, indent=2)
            logger.info(f"📝 Initialized PDF history file: {history_file}")
        else:
            logger.info(f"📄 Using existing PDF history file: {history_file}")
    
    def _save_single_pdf_to_history(self, proceeding_folder: Path, pdf_info: Dict):
        """
        Save a single PDF to the history file incrementally (thread-safe)
        This allows real-time updates as each PDF is processed
        """
        with self.history_lock:
            history_file = proceeding_folder / f"{proceeding_folder.name}_scraped_pdf_history.json"
            
            # Load existing history
            existing_history = self._load_existing_history_safe(history_file)
            
            # Add new PDF entry
            url_hash = self._create_url_hash(pdf_info['pdf_url'])
            
            # Create entry with required fields only
            entry = {
                'url': pdf_info['pdf_url'],
                'title': pdf_info['title'],
                'document_type': pdf_info['document_type'],
                'pdf_creation_date': pdf_info['pdf_creation_date'],
                'scrape_date': pdf_info['scrape_date'],
                'status': 'discovered',
                'source': pdf_info.get('source', 'csv'),  # Required: 'csv' or 'google search'
                'parent_url': pdf_info.get('parent_page_url', ''),  # URL of parent page
                'metadata': pdf_info.get('pdf_metadata', {})
            }
            
            # Preserve specific additional fields (excluding link_text, parent_text, raw_url)
            additional_fields = ['filing_date', 'filed_by', 'description', 'search_query']
            
            for field in additional_fields:
                if field in pdf_info:
                    entry[field] = pdf_info[field]
            
            # Update history
            existing_history[url_hash] = entry
            
            # Save immediately (create backup first)
            self._create_backup_before_write(history_file)
            
            try:
                with open(history_file, 'w') as f:
                    json.dump(existing_history, f, indent=2)
                logger.info(f"💾 Incrementally saved PDF to history: {pdf_info.get('title', 'Unknown')} -> {pdf_info['pdf_url']}")
            except Exception as e:
                logger.error(f"❌ Failed to save PDF to history: {e}")
                self._restore_from_backup_if_needed(history_file)
    
    def _get_csv_download_url(self, proceeding: str) -> str:
        """Get the base CPUC search URL for the proceeding navigation"""
        # Start at the main CPUC search page
        search_url = "https://apps.cpuc.ca.gov/apex/f?p=401:1"
        logger.info(f"Starting CPUC navigation at: {search_url}")
        return search_url
    
    def _download_csv(self, url: str, csv_path: Path, proceeding: str):
        """
        Download CSV file from CPUC website following exact navigation sequence:
        1. Go to https://apps.cpuc.ca.gov/apex/f?p=401:1
        2. Enter proceeding in 'Proceeding Number Search:' text box
        3. Click "Search"  
        4. Click on first result in "Proceeding Number" column
        5. Click on 'Documents' tab
        6. Click on 'Download' button
        7. Save as [proceeding].csv
        """
        self._setup_driver()
        
        try:
            logger.info(f"🔍 Step 1: Navigating to CPUC main search page")
            # Step 1: Start at the CPUC search page
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 15)
            logger.info(f"✅ Successfully loaded page: {self.driver.current_url}")
            
            logger.info(f"🔍 Step 2: Entering proceeding number '{proceeding}' in search box")
            # Step 2: Find and fill the proceeding number search box - using the exact field ID
            try:
                search_box = wait.until(ec.element_to_be_clickable((
                    By.ID, "P1_PROCEEDING_NUM"
                )))
                logger.info(f"✅ Found proceeding input field: P1_PROCEEDING_NUM")
                search_box.clear()
                search_box.send_keys(proceeding)
                
                # Verify the value was entered
                entered_value = search_box.get_attribute('value')
                if entered_value != proceeding:
                    logger.warning(f"⚠️ Value verification failed: expected '{proceeding}', got '{entered_value}'")
                else:
                    logger.info(f"✅ Successfully entered and verified '{proceeding}' in field P1_PROCEEDING_NUM")
                    
            except Exception as field_error:
                logger.error(f"❌ Failed to find or fill P1_PROCEEDING_NUM field: {field_error}")
                # Try to find any proceeding-related input field as fallback
                try:
                    logger.info("🔍 Attempting fallback selector for proceeding field...")
                    search_box = wait.until(ec.element_to_be_clickable((
                        By.XPATH, "//input[contains(@name, 'PROCEEDING') or contains(@id, 'PROCEEDING')]"
                    )))
                    search_box.clear()
                    search_box.send_keys(proceeding)
                    logger.info(f"✅ Successfully used fallback selector for proceeding: {proceeding}")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback selector also failed: {fallback_error}")
                    raise field_error
            
            logger.info(f"🔍 Step 3: Clicking Search button")  
            # Step 3: Click the Search button
            search_button = wait.until(ec.element_to_be_clickable((
                By.XPATH, "//input[@value='Search' or @type='submit'] | //button[text()='Search']"
            )))
            search_button.click()
            
            # Wait for search results to load
            time.sleep(3)
            
            logger.info(f"🔍 Step 4: Clicking on first proceeding result")
            # Step 4: Click on the first result in Proceeding Number column
            proceeding_link = wait.until(ec.element_to_be_clickable((
                By.XPATH, f"//a[contains(@href, 'P5_PROCEEDING_SELECT:{proceeding}') or contains(text(), '{proceeding}')]"
            )))
            proceeding_link.click()
            
            # Wait for proceeding page to load
            time.sleep(3)
            
            logger.info(f"🔍 Step 5: Clicking on Documents tab")
            # Step 5: Click on the 'Documents' tab
            documents_tab = wait.until(ec.element_to_be_clickable((
                By.XPATH, "//span[text()='Documents'] | //a[contains(text(), 'Documents')] | //tab[contains(text(), 'Documents')]"
            )))
            documents_tab.click()
            
            # Wait for documents tab to load
            time.sleep(3)
            
            logger.info(f"🔍 Step 6: Clicking Download button")
            # Step 6: Click on the 'Download' button to get CSV
            download_button = wait.until(ec.element_to_be_clickable((
                By.XPATH, "//input[@value='Download' and contains(@onclick, 'CSV')] | //button[text()='Download'] | //input[@type='button' and @value='Download']"
            )))
            
            # Get the download URL from the onclick attribute if available
            onclick_attr = download_button.get_attribute('onclick')
            if onclick_attr and 'CSV' in onclick_attr:
                logger.info(f"Found CSV download onclick: {onclick_attr}")
            
            download_button.click()
            
            # Wait for download to start
            time.sleep(5)
            
            logger.info(f"🔍 Step 7: Processing downloaded CSV file")
            # Step 7: The file should be downloaded - handle the download
            # Since Selenium downloads go to default download folder, we need to handle this
            self._handle_csv_download(csv_path, proceeding)
            
        except Exception as e:
            logger.error(f"Error in CPUC navigation sequence: {e}")
            # For testing, we should NOT create fallback - let the error propagate
            raise Exception(f"Failed to download CSV from CPUC website: {e}")
            
    def _handle_csv_download(self, csv_path: Path, proceeding: str):
        """Handle the downloaded CSV file and move it to the correct location"""
        import glob
        
        try:
            # Look for downloaded files in common download locations
            download_locations = [
                Path.home() / "Downloads",
                Path.cwd() / "downloads",
                Path.cwd()
            ]
            
            # Look for recently downloaded CSV files (including default "documents.csv")
            csv_files = []
            for location in download_locations:
                if location.exists():
                    # Look for files that might be the downloaded CSV
                    pattern_files = list(location.glob("*.csv"))
                    pattern_files.extend(location.glob("Documents*.csv"))
                    pattern_files.extend(location.glob("documents*.csv"))
                    pattern_files.extend(location.glob("export*.csv"))  # Sometimes CPUC exports as export.csv
                    csv_files.extend(pattern_files)
            
            if csv_files:
                # Sort by modification time to get the most recent
                csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_csv = csv_files[0]
                
                logger.info(f"Found downloaded CSV: {latest_csv}")
                logger.info(f"Renaming to proceeding-specific name: {csv_path.name}")
                
                # Copy to the target location with proceeding-specific name
                shutil.copy2(latest_csv, csv_path)
                logger.info(f"Renamed and moved CSV to: {csv_path}")
                
                # Delete the original download to prevent file bloat
                try:
                    latest_csv.unlink()
                    logger.info(f"🗑️ Deleted original download to prevent file bloat: {latest_csv}")
                except Exception as delete_error:
                    logger.warning(f"⚠️ Could not delete original CSV (not critical): {delete_error}")
                
                logger.info(f"✅ Successfully copied and renamed CSV")
                logger.info(f"   Original: {latest_csv} -> Target: {csv_path}")
                logger.info(f"   (Original deleted to prevent bloat)")
                    
            else:
                logger.error("No downloaded CSV file found - download failed")
                raise Exception(f"CSV download failed - no file found in download directories")
                
        except Exception as e:
            logger.error(f"Error handling CSV download: {e}")
            raise Exception(f"Failed to handle CSV download: {e}")
    
    def _create_fallback_csv(self, csv_path: Path):
        """Create a fallback CSV structure if download fails"""
        logger.info("Creating fallback CSV structure")
        
        # Create CSV with proper CPUC structure based on known format
        df = pd.DataFrame(columns=[
            'Document Type', 'Title', 'Date', 'URL', 'Filing Date', 
            'Document Category', 'Proceeding', 'Author', 'Document ID'
        ])
        df.to_csv(csv_path, index=False)
        logger.info(f"Created fallback CSV at: {csv_path}")
    
    def _extract_document_type(self, cells) -> str:
        """Extract document type from table cells"""
        # Look for common CPUC document types
        text = ' '.join([cell.get_text() for cell in cells]).lower()
        
        if 'decision' in text:
            return 'Decision'
        elif 'ruling' in text:
            return 'Ruling'
        elif 'comment' in text:
            return 'Comment'
        elif 'brief' in text:
            return 'Brief'
        elif 'application' in text:
            return 'Application'
        elif 'proposal' in text:
            return 'Proposal'
        else:
            return 'Other'
    
    def _extract_document_title(self, cells) -> str:
        """Extract document title from table cells"""
        # Usually the first or second cell contains the title
        for cell in cells[:2]:
            text = cell.get_text().strip()
            if len(text) > 10:  # Reasonable title length
                return text
        return "Unknown Title"
    
    def _extract_document_date(self, cells) -> str:
        """Extract document date from table cells"""
        # Look for date patterns in cells
        date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
        
        for cell in cells:
            text = cell.get_text()
            match = re.search(date_pattern, text)
            if match:
                return match.group()
        
        return datetime.now().strftime("%m/%d/%Y")
    
    def _extract_document_url(self, cells) -> Optional[str]:
        """Extract PDF URL from table cells"""
        for cell in cells:
            # Look for links to PDFs
            links = cell.find_all('a', href=True)
            for link in links:
                href = link['href']
                if '.pdf' in href.lower() or '.PDF' in href:
                    # Ensure it's a full URL
                    if href.startswith('http'):
                        return href
                    elif href.startswith('/'):
                        return f"https://docs.cpuc.ca.gov{href}"
        return None
    
    def _analyze_csv_and_scrape_pdfs(self, proceeding: str, csv_path: Path, progress: ProgressBar, proceeding_folder: Path) -> List[Dict]:
        """
        Step 2: Analyze CSV and scrape PDF metadata by extracting URLs from Document Type column
        Uses parallel processing for CSV document URL scraping
        """
        logger.info(f"Step 2: Analyzing CSV and extracting document URLs for {proceeding}")
        
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return []
        
        # Extract URLs from the Document Type column which contains HTML links
        document_urls = []
        for idx, row in df.iterrows():
            if pd.notna(row.get('Document Type')):
                # Parse HTML to extract URL from Document Type column
                doc_type_html = row['Document Type']
                if 'href=' in doc_type_html:
                    try:
                        soup = BeautifulSoup(doc_type_html, 'html.parser')
                        link = soup.find('a')
                        if link and link.get('href'):
                            url = link['href']
                            # Convert relative URL to absolute
                            if url.startswith('/'):
                                url = f"https://docs.cpuc.ca.gov{url}"
                            elif not url.startswith('http'):
                                url = f"https://docs.cpuc.ca.gov/{url}"
                            
                            document_urls.append({
                                'url': url,
                                'filing_date': row.get('Filing Date', 'Unknown'),
                                'filed_by': row.get('Filed By', 'Unknown'),
                                'description': row.get('Description', 'Unknown')
                            })
                    except Exception as e:
                        logger.error(f"Error parsing HTML in row {idx}: {e}")
        
        logger.info(f"Found {len(document_urls)} document URLs from CSV")
        
        # Filter out already processed URLs
        unprocessed_urls = []
        for doc_info in document_urls:
            if not self._check_if_page_already_processed(doc_info['url'], proceeding_folder):
                unprocessed_urls.append(doc_info)
            else:
                logger.info(f"⏭️ Skipping already processed page: {doc_info['url']}")
        
        logger.info(f"Processing {len(unprocessed_urls)} unprocessed document URLs with parallel workers")
        
        # Process URLs in parallel using ThreadPoolExecutor with reduced concurrency for reliability
        pdf_metadata = []
        # Reduced from 4 to 2 workers to prevent resource contention and browser interference
        max_workers = min(2, len(unprocessed_urls)) if unprocessed_urls else 1
        
        if unprocessed_urls:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit document processing tasks with staggered delays to reduce contention
                future_to_doc = {}
                for i, doc_info in enumerate(unprocessed_urls):
                    # Add small delay between worker submissions to prevent resource contention
                    if i > 0:
                        time.sleep(0.5)  # 500ms delay between worker starts
                    
                    future = executor.submit(self._process_document_url_worker, doc_info, proceeding, proceeding_folder)
                    future_to_doc[future] = doc_info
                
                # Process completed tasks with improved monitoring
                completed_count = 0
                success_count = 0
                failure_count = 0
                
                for future in as_completed(future_to_doc):
                    doc_info = future_to_doc[future]
                    completed_count += 1
                    
                    try:
                        pdfs_from_page = future.result()
                        pdf_metadata.extend(pdfs_from_page)
                        
                        # Update progress
                        progress.update(2, len(pdf_metadata), len(pdf_metadata))
                        
                        if len(pdfs_from_page) > 0:
                            success_count += 1
                            logger.info(f"✅ SUCCESS {completed_count}/{len(unprocessed_urls)}: {doc_info['url']} -> {len(pdfs_from_page)} PDFs")
                        else:
                            failure_count += 1
                            logger.warning(f"⚠️ NO PDFS {completed_count}/{len(unprocessed_urls)}: {doc_info['url']} -> No PDFs found")
                        
                        # Save each PDF incrementally for real-time updates
                        for pdf_info in pdfs_from_page:
                            self._save_single_pdf_to_history(proceeding_folder, pdf_info)
                        
                    except Exception as e:
                        failure_count += 1
                        logger.error(f"❌ FAILED {completed_count}/{len(unprocessed_urls)}: {doc_info['url']} -> {e}")
                        continue
                
                # Log final statistics
                logger.info(f"📊 Parallel processing completed: {success_count} successes, {failure_count} failures out of {len(unprocessed_urls)} total")
                if failure_count > 0:
                    logger.warning(f"⚠️ {failure_count} documents failed processing - may need manual review")
        
        logger.info(f"📊 Parallel CSV processing completed: {len(pdf_metadata)} total PDFs discovered from {len(document_urls)} document URLs")
        return pdf_metadata
    
    def _process_document_url_worker(self, doc_info: Dict, proceeding: str, proceeding_folder: Path) -> List[Dict]:
        """
        Worker method for parallel processing of individual document URLs with retry logic
        This method is called by ThreadPoolExecutor for each document URL
        Each worker gets its own WebDriver instance to avoid session conflicts
        """
        worker_driver = None
        max_retries = 2  # Retry failed URLs up to 2 times
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"🔄 Retry attempt {attempt} for: {doc_info['url']}")
                    # Add delay between retries to avoid overwhelming the server
                    time.sleep(2 * attempt)
                else:
                    logger.info(f"🔄 Worker processing: {doc_info['url']}")
                
                # Create a dedicated WebDriver instance for this worker with background settings
                worker_driver = self._create_background_webdriver()
                logger.info(f"🔧 Worker created background WebDriver instance")
                
                # Extract PDFs using the worker's dedicated driver
                pdfs_from_page = self._extract_pdfs_from_document_page_with_driver(
                    worker_driver, doc_info['url'], doc_info, proceeding
                )
                
                logger.info(f"🔄 Worker completed: {doc_info['url']} -> {len(pdfs_from_page)} PDFs")
                return pdfs_from_page
                
            except Exception as e:
                logger.warning(f"🔄 Worker attempt {attempt + 1} failed: {doc_info['url']} -> {e}")
                
                # Clean up failed driver before retry
                if worker_driver:
                    try:
                        worker_driver.quit()
                        worker_driver = None
                    except:
                        pass
                
                # If this was the last attempt, log as error
                if attempt == max_retries:
                    logger.error(f"🔄 Worker failed after {max_retries + 1} attempts: {doc_info['url']}")
                    return []
                    
            finally:
                # Clean up the worker's WebDriver instance
                if worker_driver:
                    try:
                        worker_driver.quit()
                        logger.info(f"🧹 Worker cleaned up WebDriver instance")
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ Worker WebDriver cleanup warning: {cleanup_error}")
                        
        return []
    
    def _extract_pdfs_from_document_page_with_driver(self, driver, document_url: str, doc_info: Dict, proceeding: str) -> List[Dict]:
        """
        Extract PDFs from a CPUC document page by parsing the ResultTable
        Uses provided WebDriver instance (for parallel processing)
        """
        pdfs = []
        
        try:
            logger.info(f"Visiting document page: {document_url}")
            
            driver.get(document_url)
            time.sleep(2)  # Wait for page to load
            
            # Look for the ResultTable
            try:
                result_table = driver.find_element(By.ID, "ResultTable")
                logger.info("Found ResultTable on document page")
                
                # Find data rows (skip header rows and separator rows)
                # Look for rows with td cells (not th) that contain actual document data
                # Exclude rows that contain header elements or have height:1px (separator rows)
                data_rows = result_table.find_elements(By.XPATH, ".//tbody/tr[td and count(td) >= 4 and not(contains(@style, 'height:1px'))]")
                logger.info(f"Found {len(data_rows)} data rows in ResultTable")
                
                # Process only the first data row that is not a Certificate of Service
                for row_idx, row in enumerate(data_rows):
                    try:
                        # Extract all cell data at once to avoid stale element references
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 4:  # Title, Type, Link, Date
                            
                            # Extract all text content immediately
                            title_text = cells[0].text.strip()
                            doc_type = cells[1].text.strip()
                            link_cell = cells[2]  # Keep reference for link extraction
                            filing_date = cells[3].text.strip()
                            
                            # Skip Certificate of Service documents
                            if "Certificate of Service" in title_text:
                                logger.info(f"Skipping Certificate of Service: {title_text}")
                                continue
                            
                            # Extract PDF links from link cell immediately
                            pdf_links = link_cell.find_elements(By.XPATH, ".//a[contains(@href, '.PDF') or contains(@href, '.pdf')]")
                            
                            # Collect all link data at once to avoid stale references
                            link_data = []
                            for pdf_link in pdf_links:
                                href = pdf_link.get_attribute('href')
                                link_text = pdf_link.text.strip()
                                if href:
                                    link_data.append({'href': href, 'link_text': link_text})
                            
                            # Process collected link data
                            row_pdfs = []
                            for link_info in link_data:
                                href = link_info['href']
                                link_text = link_info['link_text']
                                
                                # Convert relative URL to absolute if needed
                                if href.startswith('/'):
                                    pdf_url = f"https://docs.cpuc.ca.gov{href}"
                                elif not href.startswith('http'):
                                    pdf_url = f"https://docs.cpuc.ca.gov/{href}"
                                else:
                                    pdf_url = href
                                
                                # Create PDF info dictionary
                                pdf_info = {
                                    'pdf_url': pdf_url,
                                    'parent_page_url': document_url,
                                    'source': 'csv',
                                    'title': title_text,
                                    'document_type': doc_type,
                                    'filing_date': filing_date if filing_date else doc_info.get('filing_date', 'Unknown'),
                                    'filed_by': doc_info.get('filed_by', 'Unknown'),
                                    'description': doc_info.get('description', 'Unknown'),
                                    'scrape_date': datetime.now().strftime('%m/%d/%Y'),
                                    'pdf_creation_date': filing_date if filing_date else doc_info.get('filing_date', 'Unknown'),
                                    # Temporary fields for filtering only (not saved to history)
                                    '_link_text': link_text,
                                    '_parent_text': link_text  # Use link_text as parent_text
                                }
                                
                                row_pdfs.append(pdf_info)
                            
                            # Apply clean PDF filtering logic
                            filtered_pdfs = self._filter_clean_pdfs(row_pdfs)
                            
                            # Add filtered PDFs to results with duplicate checking
                            added_count = 0
                            for pdf_info in filtered_pdfs:
                                # Check if already scraped (skip if exists)
                                proceeding_folder = Path("cpuc_proceedings") / proceeding
                                if self._check_if_already_scraped(pdf_info['pdf_url'], proceeding_folder):
                                    continue
                                
                                # Analyze the PDF with timeout protection
                                enhanced_pdf_info = self._analyze_pdf_with_timeout(
                                    pdf_info['pdf_url'], 
                                    pdf_info['document_type'], 
                                    timeout=10
                                )
                                
                                if enhanced_pdf_info:  # Only add if analysis succeeded
                                    # Copy over the CSV-specific fields
                                    enhanced_pdf_info.update({
                                        'parent_page_url': pdf_info['parent_page_url'],
                                        'source': pdf_info['source'],
                                        'filing_date': pdf_info['filing_date'],
                                        'filed_by': pdf_info['filed_by'],
                                        'description': pdf_info['description']
                                    })
                                    pdfs.append(enhanced_pdf_info)
                                    added_count += 1
                                    logger.info(f"✅ Added PDF from first row: {enhanced_pdf_info['title']} -> {enhanced_pdf_info['pdf_url']}")
                                else:
                                    logger.warning(f"⚠️ Skipped PDF (analysis failed): {pdf_info['pdf_url']}")
                            
                            # Process only the first valid data row (not Certificate of Service)
                            logger.info(f"✅ Processed first data row, found {len(filtered_pdfs)} PDFs. Skipping remaining rows.")
                            break
                            
                    except Exception as row_error:
                        logger.error(f"Error processing row {row_idx}: {row_error}")
                        continue
                        
            except Exception as table_error:
                logger.error(f"Could not find or parse ResultTable: {table_error}")
                
        except Exception as e:
            logger.error(f"Error extracting PDFs from document page {document_url}: {e}")
        
        return pdfs
    
    def _extract_pdfs_from_document_page(self, document_url: str, doc_info: Dict, proceeding: str) -> List[Dict]:
        """
        Extract PDFs from a CPUC document page by parsing the ResultTable
        Uses the main scraper's WebDriver instance (for sequential processing)
        """
        self._setup_driver()
        return self._extract_pdfs_from_document_page_with_driver(self.driver, document_url, doc_info, proceeding)
    
    def _filter_clean_pdfs(self, pdfs: List[Dict]) -> List[Dict]:
        """
        Filter PDFs to prioritize 'clean' versions over 'redline' and original versions.
        
        Logic:
        - If a 'clean' version exists, skip 'redline' and original versions of the same document
        - Group PDFs by base document name
        - Prioritize: clean > original > redline
        """
        if not pdfs:
            return pdfs
        
        # Group PDFs by base document name
        pdf_groups = {}
        
        for pdf in pdfs:
            # Extract base document name from URL and text
            base_name = self._extract_base_document_name(pdf)
            
            if base_name not in pdf_groups:
                pdf_groups[base_name] = []
            pdf_groups[base_name].append(pdf)
        
        filtered_pdfs = []
        
        for base_name, group_pdfs in pdf_groups.items():
            if len(group_pdfs) == 1:
                # Single PDF, include it
                filtered_pdfs.extend(group_pdfs)
                continue
            
            # Multiple PDFs with same base name - apply filtering logic
            clean_pdfs = []
            original_pdfs = []
            redline_pdfs = []
            
            for pdf in group_pdfs:
                pdf_text = f"{pdf.get('_link_text', '')} {pdf.get('_parent_text', '')} {pdf.get('pdf_url', '')}".lower()
                
                if any(clean_indicator in pdf_text for clean_indicator in ['(clean)', 'clean.pdf', '-clean.pdf', '_clean.pdf']):
                    clean_pdfs.append(pdf)
                    logger.info(f"🟢 Found CLEAN PDF: {pdf.get('_link_text', 'Unknown')}")
                elif any(redline_indicator in pdf_text for redline_indicator in ['(redline)', 'redline.pdf', '-redline.pdf', '_redline.pdf']):
                    redline_pdfs.append(pdf)
                    logger.info(f"🔴 Found REDLINE PDF: {pdf.get('_link_text', 'Unknown')} (will skip if clean exists)")
                else:
                    original_pdfs.append(pdf)
                    logger.info(f"⚪ Found ORIGINAL PDF: {pdf.get('_link_text', 'Unknown')}")
            
            # Priority selection logic
            if clean_pdfs:
                # Clean versions exist - use only clean versions
                filtered_pdfs.extend(clean_pdfs)
                logger.info(f"✅ Selected {len(clean_pdfs)} CLEAN PDF(s) for '{base_name}' (skipped {len(original_pdfs)} original + {len(redline_pdfs)} redline)")
            elif original_pdfs:
                # No clean versions - use original versions
                filtered_pdfs.extend(original_pdfs)
                logger.info(f"⚪ Selected {len(original_pdfs)} ORIGINAL PDF(s) for '{base_name}' (skipped {len(redline_pdfs)} redline)")
            else:
                # Only redline versions available - include them
                filtered_pdfs.extend(redline_pdfs)
                logger.info(f"🔴 Selected {len(redline_pdfs)} REDLINE PDF(s) for '{base_name}' (no clean/original available)")
        
        logger.info(f"📊 PDF Filtering Summary: {len(pdfs)} total -> {len(filtered_pdfs)} filtered (removed {len(pdfs) - len(filtered_pdfs)} duplicates/inferior versions)")
        return filtered_pdfs
    
    def _extract_base_document_name(self, pdf: Dict) -> str:
        """
        Extract base document name for grouping similar PDFs
        """
        # Try multiple sources for document name
        sources = [
            pdf.get('_link_text', ''),
            pdf.get('title', ''),
            pdf.get('pdf_url', '').split('/')[-1]  # filename from URL
        ]
        
        for source in sources:
            if source:
                # Remove common suffixes and normalize
                base_name = source.lower()
                
                # Remove file extensions
                base_name = re.sub(r'\.(pdf|doc|docx)$', '', base_name)
                
                # Remove version indicators
                base_name = re.sub(r'\s*\(clean\)$', '', base_name)
                base_name = re.sub(r'\s*\(redline\)$', '', base_name)
                base_name = re.sub(r'\s*clean$', '', base_name)
                base_name = re.sub(r'\s*redline$', '', base_name)
                base_name = re.sub(r'[-_]clean$', '', base_name)
                base_name = re.sub(r'[-_]redline$', '', base_name)
                
                # Clean up trailing hyphens and underscores
                base_name = re.sub(r'[-_]+$', '', base_name)
                base_name = re.sub(r'^[-_]+', '', base_name)
                
                # Remove extra whitespace
                base_name = ' '.join(base_name.split())
                
                if base_name:
                    return base_name
        
        # Fallback to URL if no good name found
        return pdf.get('pdf_url', 'unknown')
    
    def _enhance_pdf_metadata(self, pdf_info: Dict) -> Dict:
        """
        Enhance PDF metadata by analyzing the actual PDF file
        """
        try:
            # Get additional metadata from PDF headers
            response = requests.head(pdf_info['pdf_url'], timeout=10)
            if response.status_code == 200:
                content_length = response.headers.get('content-length', '0')
                content_type = response.headers.get('content-type', '')
                last_modified = response.headers.get('last-modified', '')
                
                pdf_info['pdf_metadata'] = {
                    'file_size': content_length,
                    'content_type': content_type,
                    'last_modified': last_modified
                }
                
                # Update creation date from headers if available
                if last_modified:
                    try:
                        from email.utils import parsedate_to_datetime
                        mod_date = parsedate_to_datetime(last_modified)
                        pdf_info['pdf_creation_date'] = mod_date.strftime('%m/%d/%Y')
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"Could not enhance metadata for {pdf_info['pdf_url']}: {e}")
            pdf_info['pdf_metadata'] = {}
            
        return pdf_info
    
    def _check_if_already_scraped(self, pdf_url: str, proceeding_folder: Path) -> bool:
        """
        Check if PDF has already been scraped by looking in existing JSON
        
        Returns True if already scraped, False if new
        """
        history_file = proceeding_folder / f"{proceeding_folder.name}_scraped_pdf_history.json"
        existing_history = self._load_existing_history_safe(history_file)
        
        url_hash = self._create_url_hash(pdf_url)
        already_scraped = url_hash in existing_history
        
        if already_scraped:
            logger.info(f"⏭️ Skipping already scraped PDF: {pdf_url}")
            return True
        
        return False
    
    def _check_if_page_already_processed(self, page_url: str, proceeding_folder: Path) -> bool:
        """
        Check if a page URL has already been processed by looking for any PDFs from that source_page
        
        Returns True if page already processed, False if new
        """
        history_file = proceeding_folder / f"{proceeding_folder.name}_scraped_pdf_history.json"
        existing_history = self._load_existing_history_safe(history_file)
        
        for pdf_data in existing_history.values():
            if pdf_data.get('source_page') == page_url:
                logger.info(f"⏭️ Skipping already processed page: {page_url}")
                return True
        
        return False
    
    def _analyze_pdf_with_timeout(self, pdf_url: str, document_type: str, timeout: int = 10) -> Optional[Dict]:
        """
        Analyze a PDF to extract metadata with timeout protection
        
        SAFETY: 10-second timeout prevents hanging on slow/broken URLs
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Analyzing PDF (timeout: {timeout}s): {pdf_url}")
            
            # Get PDF headers and basic info with timeout
            response = requests.head(pdf_url, timeout=timeout)
            response.raise_for_status()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ PDF analyzed successfully in {elapsed:.2f}s: {pdf_url}")
            
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            logger.warning(f"⏰ PDF analysis timed out after {elapsed:.2f}s, skipping: {pdf_url}")
            return None
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            logger.warning(f"❌ PDF analysis failed after {elapsed:.2f}s: {pdf_url} - {e}")
            return None
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Unexpected error after {elapsed:.2f}s analyzing PDF {pdf_url}: {e}")
            return None
            
        # Get file size and content type
        content_length = response.headers.get('content-length', '0')
        content_type = response.headers.get('content-type', '')
        last_modified = response.headers.get('last-modified', '')
        
        # Parse creation/modification date from headers
        creation_date = datetime.now().strftime('%m/%d/%Y')
        if last_modified:
            try:
                # Parse HTTP date format
                from email.utils import parsedate_to_datetime
                mod_date = parsedate_to_datetime(last_modified)
                creation_date = mod_date.strftime('%m/%d/%Y')
            except:
                pass
        
        # Extract title from URL or generate from document type
        url_path = Path(pdf_url)
        filename = url_path.name
        
        # Create a readable title from filename
        title = filename.replace('.pdf', '').replace('.PDF', '')
        title = title.replace('_', ' ').replace('-', ' ')
        
        # If title is too short or unclear, use document type
        if len(title) < 5:
            title = f"{document_type} Document"
        
        # Create comprehensive metadata dictionary
        pdf_info = {
            'pdf_url': pdf_url,
            'title': title,
            'document_type': document_type,
            'pdf_creation_date': creation_date,
            'scrape_date': datetime.now().strftime('%m/%d/%Y'),
            'analysis_duration': round(elapsed, 2),
            'pdf_metadata': {
                'file_size': content_length,
                'content_type': content_type,
                'last_modified': last_modified,
                'analysis_timestamp': datetime.now().isoformat(),
                'timeout_used': timeout
            },
            'status': 'successfully_analyzed',
            # Temporary fields for filtering compatibility (not saved to history)
            '_link_text': title,
            '_parent_text': title
        }
        
        return pdf_info
    
    def _analyze_pdf(self, pdf_url: str, document_type: str) -> Optional[Dict]:
        """
        Legacy method - now calls timeout-protected version
        """
        return self._analyze_pdf_with_timeout(pdf_url, document_type, timeout=10)
    
    def _google_search_for_pdfs(self, proceeding: str, existing_pdfs: List[Dict], progress: ProgressBar) -> List[Dict]:
        """
        Step 3: Google search for additional PDFs from cpuc.ca.gov
        """
        logger.info(f"Step 3: Google search for additional PDFs for {proceeding}")
        
        # Format proceeding for search (R2207005 -> R.22-07-005)
        formatted_proceeding = f"R.{proceeding[1:3]}-{proceeding[3:5]}-{proceeding[5:]}"
        search_query = f"{formatted_proceeding} site:cpuc.ca.gov filetype:pdf"
        
        existing_urls = {pdf['pdf_url'] for pdf in existing_pdfs}
        additional_pdfs = []
        google_search_urls = set()  # Track URLs found in current Google search to prevent duplicates
        
        try:
            # Perform Google search
            logger.info(f"Searching Google for: {search_query}")
            search_results = list(search(search_query, num_results=10))
            
            discovered_count = len(existing_pdfs)
            
            for idx, url in enumerate(search_results):
                progress.update(3, discovered_count + idx + 1, len(existing_pdfs) + len(additional_pdfs))
                
                # Check for duplicates within Google search results
                if url in google_search_urls:
                    logger.info(f"⏭️ Skipping duplicate Google result: {url}")
                    continue
                    
                google_search_urls.add(url)
                
                # Check if URL contains cpuc.ca.gov and is a PDF
                if 'cpuc.ca.gov' in url and url not in existing_urls:
                    if url.lower().endswith('.pdf'):
                        # Direct PDF link from Google search - parent is the search context
                        proceeding_folder = Path("cpuc_proceedings") / proceeding
                        if not self._check_if_already_scraped(url, proceeding_folder):
                            pdf_info = self._analyze_pdf_with_timeout(url, 'Google Search Result', timeout=10)
                            if pdf_info:
                                # For direct PDF links, parent_url should indicate it came from Google search
                                pdf_info['parent_page_url'] = f"Google Search: {search_query}"
                                pdf_info['source'] = 'google search'
                                pdf_info['search_query'] = search_query  # Additional context
                                additional_pdfs.append(pdf_info)
                                existing_urls.add(url)
                                logger.info(f"✅ Added Google PDF: {url}")
                            else:
                                logger.warning(f"⚠️ Skipped Google PDF (analysis failed): {url}")
                        else:
                            logger.info(f"⏭️ Skipped Google PDF (already scraped): {url}")
                    else:
                        # Webpage that might contain PDF links
                        page_pdfs = self._extract_pdfs_from_webpage(url, proceeding, existing_urls)
                        for pdf_info in page_pdfs:
                            additional_pdfs.append(pdf_info)
                            existing_urls.add(pdf_info['pdf_url'])
                
                time.sleep(1)  # Be respectful to servers
            
        except Exception as e:
            logger.error(f"Error in Google search: {e}")
        
        # Apply clean PDF filtering to Google search results
        logger.info(f"Applying clean PDF filtering to {len(additional_pdfs)} Google search results")
        filtered_additional_pdfs = self._filter_clean_pdfs(additional_pdfs)
        
        logger.info(f"Found {len(filtered_additional_pdfs)} additional PDFs from Google search (after filtering)")
        return filtered_additional_pdfs
    
    def _extract_pdfs_from_webpage(self, webpage_url: str, proceeding: str, existing_urls: set) -> List[Dict]:
        """Extract PDF links from a webpage with duplicate checking and timeout protection"""
        pdfs = []
        
        try:
            self._setup_driver()
            self.driver.get(webpage_url)
            
            # Find all PDF links
            pdf_links = self.driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf') or contains(@href, '.PDF')]")
            
            for link in pdf_links:
                href = link.get_attribute('href')
                if href and 'cpuc.ca.gov' in href and href not in existing_urls:
                    # Check if already scraped to prevent duplicates
                    proceeding_folder = Path("cpuc_proceedings") / proceeding
                    if not self._check_if_already_scraped(href, proceeding_folder):
                        # Use timeout-protected analysis
                        pdf_info = self._analyze_pdf_with_timeout(href, 'Webpage Link', timeout=10)
                        if pdf_info:
                            # Add Google search source information
                            pdf_info['parent_page_url'] = webpage_url
                            pdf_info['source'] = 'google search'
                            pdfs.append(pdf_info)
                            existing_urls.add(href)
                            logger.info(f"✅ Added webpage PDF: {href}")
                    else:
                        logger.info(f"⏭️ Skipping already scraped PDF: {href}")
            
        except Exception as e:
            logger.error(f"Error extracting PDFs from {webpage_url}: {e}")
        
        return pdfs
    
    def _save_scraped_history(self, proceeding_folder: Path, pdf_metadata: List[Dict]):
        """
        Step 4: Save scraped PDF history to JSON file (NON-DESTRUCTIVE)
        
        SAFETY: This method preserves existing data by merging new entries
        instead of overwriting the entire file.
        """
        history_file = proceeding_folder / f"{proceeding_folder.name}_scraped_pdf_history.json"
        
        # SAFETY: Load existing history first (non-destructive merge)
        existing_history = self._load_existing_history_safe(history_file)
        
        # Create new entries dictionary
        new_entries = {}
        for idx, pdf_info in enumerate(pdf_metadata):
            url_hash = self._create_url_hash(pdf_info['pdf_url'])
            
            # Create entry with required fields only
            entry = {
                'url': pdf_info['pdf_url'],
                'title': pdf_info['title'],
                'document_type': pdf_info['document_type'],
                'pdf_creation_date': pdf_info['pdf_creation_date'],
                'scrape_date': pdf_info['scrape_date'],
                'status': 'discovered',
                'source': pdf_info.get('source', 'csv'),  # Required: 'csv' or 'google search'
                'parent_url': pdf_info.get('parent_page_url', ''),  # URL of parent page
                'metadata': pdf_info.get('pdf_metadata', {})
            }
            
            # Preserve specific additional fields (excluding link_text, parent_text, raw_url)
            additional_fields = ['filing_date', 'filed_by', 'description', 'search_query']
            
            for field in additional_fields:
                if field in pdf_info:
                    entry[field] = pdf_info[field]
            
            new_entries[url_hash] = entry
        
        # SAFETY: Merge existing and new data (preserves existing entries)
        merged_history = {**existing_history, **new_entries}
        
        # SAFETY: Create backup before writing (if file exists)
        self._create_backup_before_write(history_file)
        
        # Write merged data to file
        try:
            with open(history_file, 'w') as f:
                json.dump(merged_history, f, indent=2)
                
            new_count = len(new_entries)
            total_count = len(merged_history)
            preserved_count = len(existing_history)
            
            logger.info(f"💾 Non-destructively saved PDF history:")
            logger.info(f"   • New entries: {new_count}")
            logger.info(f"   • Preserved entries: {preserved_count}") 
            logger.info(f"   • Total entries: {total_count}")
            logger.info(f"   • File: {history_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save history: {e}")
            # SAFETY: Restore from backup if write failed
            self._restore_from_backup_if_needed(history_file)
            raise
    
    def _load_existing_history_safe(self, history_file: Path) -> Dict:
        """
        SAFETY: Load existing history file safely (non-destructive)
        Returns empty dict if file doesn't exist or is corrupted
        """
        if not history_file.exists():
            logger.info(f"📄 No existing history file found at {history_file}")
            return {}
        
        try:
            with open(history_file, 'r') as f:
                existing_data = json.load(f)
            
            logger.info(f"📄 Loaded existing history: {len(existing_data)} entries from {history_file}")
            return existing_data
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"⚠️ Could not load existing history (will preserve by backup): {e}")
            return {}
    
    def _create_backup_before_write(self, history_file: Path):
        """
        SAFETY: Create backup of existing file before overwriting
        """
        if not history_file.exists():
            return
        
        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = history_file.with_suffix(f'.backup_{timestamp}.json')
        
        try:
            shutil.copy2(history_file, backup_file)
            logger.info(f"🔒 Created safety backup: {backup_file}")
            
            # Keep only last 5 backups to prevent disk bloat
            self._cleanup_old_backups(history_file.parent, history_file.stem)
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create backup (proceeding anyway): {e}")
    
    def _cleanup_old_backups(self, backup_dir: Path, file_stem: str):
        """
        SAFETY: Keep only the 5 most recent backups to prevent disk bloat
        """
        try:
            backup_pattern = f"{file_stem}.backup_*.json"
            backup_files = list(backup_dir.glob(backup_pattern))
            
            if len(backup_files) > 5:
                # Sort by modification time (oldest first)
                backup_files.sort(key=lambda x: x.stat().st_mtime)
                
                # Remove oldest backups (keep newest 5)
                for old_backup in backup_files[:-5]:
                    old_backup.unlink()
                    logger.info(f"🗑️ Cleaned up old backup: {old_backup}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not cleanup old backups: {e}")
    
    def _restore_from_backup_if_needed(self, history_file: Path):
        """
        SAFETY: Restore from most recent backup if main file is corrupted
        """
        try:
            # Find most recent backup
            backup_pattern = f"{history_file.stem}.backup_*.json"
            backup_files = list(history_file.parent.glob(backup_pattern))
            
            if backup_files:
                # Get most recent backup
                latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
                
                # Restore from backup
                shutil.copy2(latest_backup, history_file)
                logger.info(f"🔧 Restored from backup: {latest_backup} -> {history_file}")
            else:
                logger.warning(f"⚠️ No backup available to restore from")
                
        except Exception as e:
            logger.error(f"❌ Could not restore from backup: {e}")
    
    def _create_url_hash(self, url: str) -> str:
        """Create a hash from URL for use as dictionary key"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()


# Convenience functions to maintain compatibility with existing code
def scrape_proceeding_pdfs(proceeding: str, headless: bool = True) -> Dict:
    """
    Convenience function to scrape a single proceeding
    """
    scraper = CPUCSimplifiedScraper(headless=headless)
    return scraper.scrape_proceeding(proceeding)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        proceeding = sys.argv[1]
    else:
        proceeding = "R2207005"  # Default for testing
    
    result = scrape_proceeding_pdfs(proceeding)
    print(f"\nScraping Results:")
    print(f"Proceeding: {result['proceeding']}")
    print(f"Total PDFs: {result['total_pdfs']}")
    print(f"Status: {result['status']}")