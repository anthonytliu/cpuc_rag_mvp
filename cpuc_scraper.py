#!/usr/bin/env python3
"""
Unified CPUC Scraper

Consolidates CPUC document scraping functionality into a single, comprehensive scraper.
Handles both CSV fetching from CPUC website and Google search integration.

Features:
- Fetch CSV containing document table from CPUC website
- Fetch top 10 Google results for proceeding
- Extract PDF URLs from both sources
- Update scraped PDF history with new naming convention
- Compare with document_hashes to find new PDFs
- Process new PDFs into vector database

Author: Claude Code
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import googleapiclient.discovery
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://apps.cpuc.ca.gov/apex/f?p=401:1:0"
DOWNLOAD_DIR = Path("./cpuc_csvs")
MAX_GOOGLE_RESULTS = 10
DEFAULT_TIMEOUT = 30

# Thread safety for file operations
history_file_lock = threading.Lock()


class CPUCUnifiedScraper:
    """
    Unified CPUC document scraper that combines CSV fetching and Google search.
    
    This class implements the complete scraping workflow:
    1. Fetch CSV for proceeding from CPUC website
    2. Fetch top 10 Google results for proceeding
    3. Extract PDF URLs from both sources
    4. Update scraped PDF history
    5. Compare with document_hashes to find new PDFs
    """
    
    def __init__(self, headless: bool = True, max_workers: int = 4, proceedings: List[str] = None):
        """
        Initialize the unified CPUC scraper.
        
        Args:
            headless: Whether to run Chrome in headless mode
            max_workers: Number of parallel workers for processing
            proceedings: List of proceedings to scrape (defaults to config values)
        """
        self.headless = headless
        self.max_workers = max_workers
        self.download_dir = DOWNLOAD_DIR.resolve()  # Resolve to absolute path
        self.download_dir.mkdir(exist_ok=True)
        logger.info(f"Download directory set to: {self.download_dir}")
        
        # Set proceedings list
        if proceedings is None:
            import config
            self.proceedings = config.SCRAPER_PROCEEDINGS
        else:
            self.proceedings = proceedings
        
        # Chrome options for Selenium - optimized for speed
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")
        
        # Core performance optimizations
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-web-security")
        self.chrome_options.add_argument("--disable-features=TranslateUI")
        self.chrome_options.add_argument("--disable-ipc-flooding-protection")
        
        # Speed optimizations - disable unnecessary features
        self.chrome_options.add_argument("--disable-images")  # Don't load images
        self.chrome_options.add_argument("--disable-javascript")  # Most CPUC pages work without JS
        self.chrome_options.add_argument("--disable-plugins")
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-default-apps")
        self.chrome_options.add_argument("--disable-background-timer-throttling")
        self.chrome_options.add_argument("--disable-renderer-backgrounding")
        self.chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        
        # Network and loading optimizations
        self.chrome_options.add_argument("--aggressive-cache-discard")
        self.chrome_options.add_argument("--memory-pressure-off")
        self.chrome_options.add_argument("--max_old_space_size=4096")
        
        # Additional performance flags
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.chrome_options.add_argument("--disable-dev-tools")
        self.chrome_options.add_argument("--no-first-run")
        self.chrome_options.add_argument("--disable-infobars")
        self.chrome_options.add_argument("--disable-logging")
        self.chrome_options.add_argument("--disable-login-animations")
        self.chrome_options.add_argument("--disable-notifications")
        self.chrome_options.add_argument("--disable-password-generation")
        self.chrome_options.add_argument("--disable-save-password-bubble")
        self.chrome_options.add_argument("--disable-translate")
        self.chrome_options.add_argument("--hide-scrollbars")
        self.chrome_options.add_argument("--mute-audio")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-setuid-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Page load strategy optimization
        self.chrome_options.page_load_strategy = 'eager'  # Don't wait for all resources
        
        self.chrome_options.add_experimental_option("prefs", {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False,  # Disable for speed
            "plugins.always_open_pdf_externally": True,
            "profile.default_content_setting_values.notifications": 2,  # Block notifications
            "profile.managed_default_content_settings.images": 2  # Block images
        })
        
        self.driver = None
        self.driver_pool = Queue()  # Pool of drivers for parallel processing
        self.driver_lock = threading.Lock()
        
        logger.info(f"Unified CPUC scraper initialized (headless: {headless})")
    
    def scrape_proceeding_pdfs(self, proceeding: str) -> Dict:
        """
        Main scraping method that implements the complete workflow.
        
        Args:
            proceeding: Proceeding number (e.g., "R2207005")
            
        Returns:
            Dict containing:
                - csv_urls: List of URLs from CSV
                - google_urls: List of URLs from Google search
                - new_pdfs: List of new PDF URLs to process
                - total_scraped: Total number of URLs scraped
                - history_file: Path to scraped PDF history file
        """
        logger.info(f"🚀 Starting unified scraping for proceeding {proceeding}")
        
        try:
            # Step 1: Fetch CSV for proceeding
            logger.info("📋 Step 1: Fetching CSV from CPUC website...")
            csv_urls = self._fetch_csv_urls(proceeding)
            logger.info(f"✅ Found {len(csv_urls)} URLs from CSV")
            
            # Step 2: Fetch Google results
            logger.info("🔍 Step 2: Fetching Google search results...")
            google_urls = self._fetch_google_urls(proceeding)
            logger.info(f"✅ Found {len(google_urls)} URLs from Google search")
            
            # Step 3: Update scraped PDF history
            logger.info("💾 Step 3: Updating scraped PDF history...")
            all_urls = csv_urls + google_urls
            history_file = self._update_scraped_pdf_history(proceeding, all_urls)
            logger.info(f"✅ Updated history file: {history_file}")
            
            # Step 4: Compare with document_hashes
            logger.info("🔍 Step 4: Comparing with document_hashes...")
            new_pdfs = self._find_new_pdfs(proceeding, all_urls)
            logger.info(f"✅ Found {len(new_pdfs)} new PDFs to process")
            
            # Step 5: Trigger vector store building if new PDFs found
            if len(new_pdfs) > 0:
                logger.info("🔨 Step 5: Triggering vector store building...")
                build_success = self._trigger_vector_store_build(proceeding)
                if build_success:
                    logger.info("✅ Vector store building completed")
                else:
                    logger.warning("⚠️ Vector store building failed")
            
            # Step 6: Return results
            results = {
                'csv_urls': csv_urls,
                'google_urls': google_urls,
                'new_pdfs': new_pdfs,
                'total_scraped': len(all_urls),
                'history_file': str(history_file),
                'proceeding': proceeding,
                'vector_store_built': len(new_pdfs) > 0
            }
            
            logger.info(f"🎉 Scraping completed for {proceeding}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Scraping failed for {proceeding}: {e}")
            raise
        finally:
            self._cleanup_driver()
    
    def _fetch_csv_urls(self, proceeding: str) -> List[Dict[str, str]]:
        """
        Fetch CSV from CPUC website and extract PDF URLs.
        
        Args:
            proceeding: Proceeding number
            
        Returns:
            List of dictionaries with url, title, and source_type
        """
        try:
            self._setup_driver()
            
            # Navigate to CPUC website
            self.driver.get(BASE_URL)
            wait = WebDriverWait(self.driver, DEFAULT_TIMEOUT)
            
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
            
            # Find and process the CSV file
            csv_file = self._find_latest_csv()
            if csv_file:
                # Rename CSV to proceeding format
                renamed_csv = self._rename_csv_file(csv_file, proceeding)
                logger.info(f"Processing CSV file: {renamed_csv}")
                return self._extract_pdf_urls_from_csv(renamed_csv)
            else:
                logger.warning("No CSV file found after download")
                return []
                
        except Exception as e:
            logger.error(f"Failed to fetch CSV URLs: {e}")
            return []
    
    def _fetch_google_urls(self, proceeding: str) -> List[Dict[str, str]]:
        """
        Fetch Google search results using multiple methods with fallbacks.
        
        Args:
            proceeding: Proceeding number
            
        Returns:
            List of dictionaries with url, title, and source_type
        """
        logger.info(f"Starting Google search for {proceeding}...")
        
        # Try methods in order of preference
        methods = [
            ("Google Custom Search API", self._google_custom_search),
            ("Google Search with enhanced rate limiting", self._google_search_with_retry),
            ("CPUC website direct search", self._cpuc_website_search)
        ]
        
        for method_name, method_func in methods:
            try:
                logger.info(f"Trying {method_name}...")
                results = method_func(proceeding)
                
                if results:
                    logger.info(f"✅ {method_name} found {len(results)} URLs")
                    return results
                else:
                    logger.info(f"⚠️ {method_name} returned no results, trying next method...")
                    
            except Exception as e:
                logger.warning(f"❌ {method_name} failed: {e}")
                continue
        
        logger.warning(f"All Google search methods failed for {proceeding}")
        
        # Final fallback: return empty list but don't fail the entire scraping process
        # The CSV scraping part should still work to get the main document list
        logger.info("🔄 Google search failed, but CSV scraping should still provide document URLs")
        return []
    
    def _google_custom_search(self, proceeding: str) -> List[Dict[str, str]]:
        """
        Use Google Custom Search API (most reliable method).
        
        Args:
            proceeding: Proceeding number
            
        Returns:
            List of dictionaries with url, title, and source_type
        """
        try:
            import config
            
            # Check if API credentials are available
            if not config.GOOGLE_API_KEY or not config.GOOGLE_CSE_ID:
                logger.info("Google Custom Search API credentials not configured, skipping...")
                return []
            
            # Format proceeding for search (R2207005 -> R.22-07-005)
            search_term_formatted = f"R.{proceeding[1:3]}-{proceeding[3:5]}-{proceeding[5:]}"
            
            # Build the search service
            service = googleapiclient.discovery.build("customsearch", "v1", developerKey=config.GOOGLE_API_KEY)
            
            # Perform the search
            query = f'"{search_term_formatted}" site:cpuc.ca.gov'
            result = service.cse().list(
                q=query,
                cx=config.GOOGLE_CSE_ID,
                num=min(10, MAX_GOOGLE_RESULTS)
            ).execute()
            
            results = []
            found_urls = set()
            
            if 'items' in result:
                for item in result['items']:
                    url = item.get('link', '')
                    title = item.get('title', '')
                    
                    if url in found_urls:
                        continue
                    found_urls.add(url)
                    
                    if url.lower().endswith('.pdf'):
                        # Direct PDF link
                        if 'cpuc.ca.gov' in url.lower():
                            filename = self._sanitize_filename(title) or self._sanitize_filename(os.path.basename(url)) or f"api_pdf_{self._create_url_hash(url)}.pdf"
                            if not filename.lower().endswith('.pdf'):
                                filename += '.pdf'
                            results.append({
                                'url': url,
                                'title': filename,
                                'source_type': 'google_api_direct_pdf'
                            })
                    else:
                        # Scrape webpage for PDF links
                        webpage_pdfs = self._scrape_webpage_for_pdfs(url, proceeding)
                        results.extend(webpage_pdfs)
                        
                        # Add small delay between webpage scraping
                        time.sleep(config.GOOGLE_SEARCH_DELAY_SECONDS / 2)
            
            logger.info(f"Google Custom Search API found {len(results)} URLs")
            return results
            
        except Exception as e:
            logger.error(f"Google Custom Search API failed: {e}")
            raise
    
    def _google_search_with_retry(self, proceeding: str) -> List[Dict[str, str]]:
        """
        Use googlesearch-python with enhanced retry logic.
        
        Args:
            proceeding: Proceeding number
            
        Returns:
            List of dictionaries with url, title, and source_type
        """
        try:
            import config
            
            # Format proceeding for search (R2207005 -> R.22-07-005)
            search_term_formatted = f"R.{proceeding[1:3]}-{proceeding[3:5]}-{proceeding[5:]}"
            query = f'"{search_term_formatted}"'
            
            found_urls = set()
            results = []
            
            for attempt in range(config.GOOGLE_SEARCH_MAX_RETRIES):
                try:
                    logger.info(f"Google search attempt {attempt + 1}/{config.GOOGLE_SEARCH_MAX_RETRIES}: {query}")
                    
                    # Add progressive delay
                    if attempt > 0:
                        delay = config.GOOGLE_SEARCH_RETRY_DELAY * (attempt + 1)
                        logger.info(f"Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                    else:
                        time.sleep(config.GOOGLE_SEARCH_DELAY_SECONDS)
                    
                    count = 0
                    for url in search(query, num_results=MAX_GOOGLE_RESULTS):
                        if url in found_urls:
                            continue
                        found_urls.add(url)
                        
                        if url.lower().endswith('.pdf'):
                            # Direct PDF link - only add if it's from CPUC
                            if 'cpuc.ca.gov' in url.lower():
                                filename = self._sanitize_filename(os.path.basename(url)) or f"google_pdf_{self._create_url_hash(url)}.pdf"
                                results.append({
                                    'url': url,
                                    'title': filename,
                                    'source_type': 'google_search_direct_pdf'
                                })
                        else:
                            # Only scrape webpages from cpuc.ca.gov for PDF links
                            if 'cpuc.ca.gov' in url.lower():
                                webpage_pdfs = self._scrape_webpage_for_pdfs(url, proceeding)
                                results.extend(webpage_pdfs)
                                # Add small delay between webpage scraping
                                time.sleep(config.GOOGLE_SEARCH_DELAY_SECONDS / 2)
                            # Ignore non-CPUC URLs to save time
                        
                        count += 1
                        if count >= MAX_GOOGLE_RESULTS:
                            break
                    
                    # If we got results, break out of retry loop
                    if results:
                        break
                        
                except Exception as e:
                    logger.warning(f"Google search attempt {attempt + 1} failed: {e}")
                    
                    # If rate limited, use longer delay
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        if attempt < config.GOOGLE_SEARCH_MAX_RETRIES - 1:
                            delay = config.GOOGLE_SEARCH_RETRY_DELAY * (attempt + 2)
                            logger.info(f"Rate limited - waiting {delay} seconds before retry...")
                            time.sleep(delay)
                        else:
                            logger.error("Max retries reached for Google search")
                            raise
                    else:
                        raise
            
            logger.info(f"Google search with retry found {len(results)} URLs")
            return results
            
        except Exception as e:
            logger.error(f"Google search with retry failed: {e}")
            raise
    
    def _cpuc_website_search(self, proceeding: str) -> List[Dict[str, str]]:
        """
        Fallback to direct CPUC website search.
        
        Args:
            proceeding: Proceeding number
            
        Returns:
            List of dictionaries with url, title, and source_type
        """
        try:
            import config
            
            logger.info("Using CPUC website direct search as fallback...")
            
            # Format proceeding for CPUC search
            search_term_formatted = f"R.{proceeding[1:3]}-{proceeding[3:5]}-{proceeding[5:]}"
            
            # Use requests to search CPUC website directly
            search_url = f"{config.CPUC_SEARCH_BASE}?category=proceeding&proceeding={proceeding}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the search results
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            found_urls = set()
            
            # Find PDF links in the search results
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                
                if not href:
                    continue
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = f"{config.CPUC_BASE_URL}{href}"
                elif not href.startswith('http'):
                    continue
                
                # Check if it's a PDF link
                if (href.lower().endswith('.pdf') or 
                    'filetype=pdf' in href.lower() or
                    'docformat=pdf' in href.lower()):
                    
                    if href in found_urls:
                        continue
                    found_urls.add(href)
                    
                    # Only add PDFs from CPUC
                    if 'cpuc.ca.gov' in href.lower():
                        # Extract title from link text
                        title = link.get_text(strip=True)
                        if not title:
                            title = os.path.basename(href) if href.lower().endswith('.pdf') else f"cpuc_search_{self._create_url_hash(href)}.pdf"
                        
                        # Clean up title
                        title = self._sanitize_filename(title)
                        if not title.lower().endswith('.pdf'):
                            title += '.pdf'
                        
                        results.append({
                            'url': href,
                            'title': title,
                            'source_type': 'cpuc_website_search_pdf'
                        })
            
            logger.info(f"CPUC website search found {len(results)} URLs")
            return results
            
        except Exception as e:
            logger.error(f"CPUC website search failed: {e}")
            raise
    
    def _scrape_webpage_for_pdfs(self, webpage_url: str, proceeding: str) -> List[Dict[str, str]]:
        """
        Scrape a webpage for PDF links.
        
        Args:
            webpage_url: URL of the webpage to scrape
            proceeding: Proceeding number for context
            
        Returns:
            List of PDF URLs found on the webpage
        """
        pdfs_found = []
        try:
            logger.info(f"Scraping webpage for PDFs: {webpage_url}")
            
            # Use requests to get the webpage content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(webpage_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links that might be PDFs
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                
                # Skip if not a link
                if not href:
                    continue
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = f"https://docs.cpuc.ca.gov{href}"
                elif href.startswith('../'):
                    # Handle relative paths
                    from urllib.parse import urljoin
                    href = urljoin(webpage_url, href)
                elif not href.startswith('http'):
                    continue
                
                # Check if it's a PDF link
                if (href.lower().endswith('.pdf') or 
                    'filetype=pdf' in href.lower() or
                    'docformat=pdf' in href.lower() or
                    ('SearchRes.aspx' in href and 'DocID=' in href)):
                    
                    # Only add PDFs from CPUC
                    if 'cpuc.ca.gov' in href.lower():
                        # Extract title from link text or href
                        title = link.get_text(strip=True)
                        if not title:
                            title = os.path.basename(href) if href.lower().endswith('.pdf') else f"document_{self._create_url_hash(href)}.pdf"
                        
                        # Clean up title
                        title = self._sanitize_filename(title)
                        if not title.lower().endswith('.pdf'):
                            title += '.pdf'
                        
                        pdfs_found.append({
                            'url': href,
                            'title': title,
                            'source_type': 'google_webpage_pdf'
                        })
            
            # Remove duplicates
            seen_urls = set()
            unique_pdfs = []
            for pdf in pdfs_found:
                if pdf['url'] not in seen_urls:
                    seen_urls.add(pdf['url'])
                    unique_pdfs.append(pdf)
            
            logger.info(f"Found {len(unique_pdfs)} PDFs on webpage: {webpage_url}")
            return unique_pdfs
            
        except Exception as e:
            logger.warning(f"Failed to scrape webpage {webpage_url}: {e}")
            return []
    
    def _update_scraped_pdf_history(self, proceeding: str, urls: List[Dict[str, str]]) -> Path:
        """
        Update the scraped PDF history file with new naming convention.
        
        Args:
            proceeding: Proceeding number
            urls: List of URL dictionaries
            
        Returns:
            Path to the updated history file
        """
        try:
            # New naming convention: <proceeding>_scraped_pdf_history.json
            history_file = self.download_dir / f"{proceeding.lower()}_scraped_pdf_history.json"
            
            # Load existing history
            history = self._load_scraped_pdf_history(proceeding)
            
            # Process each URL
            for url_data in urls:
                url = url_data['url']
                title = url_data.get('title', '')
                source_type = url_data.get('source_type', 'unknown')
                
                # Create URL hash for deduplication
                url_hash = self._create_url_hash(url)
                
                # Update history entry
                history[url_hash] = {
                    'url': url,
                    'title': title,
                    'source_type': source_type,
                    'scraped_date': datetime.now().isoformat(),
                    'last_checked': datetime.now().isoformat()
                }
            
            # Save updated history
            self._save_scraped_pdf_history(proceeding, history)
            
            logger.info(f"Updated scraped PDF history with {len(urls)} URLs")
            return history_file
            
        except Exception as e:
            logger.error(f"Failed to update scraped PDF history: {e}")
            raise
    
    def _find_new_pdfs(self, proceeding: str, urls: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Compare scraped URLs with document_hashes to find new PDFs.
        
        Args:
            proceeding: Proceeding number
            urls: List of scraped URL dictionaries
            
        Returns:
            List of new PDF URLs that need processing
        """
        try:
            # Load document_hashes from proceeding-specific RAG system
            document_hashes_file = Path(f"local_chroma_db/{proceeding}/document_hashes.json")
            
            if document_hashes_file.exists():
                with open(document_hashes_file, 'r') as f:
                    document_hashes = json.load(f)
                existing_hashes = set(document_hashes.keys())
            else:
                existing_hashes = set()
            
            # Find new URLs
            new_pdfs = []
            for url_data in urls:
                url = url_data['url']
                url_hash = self._create_url_hash(url)
                
                if url_hash not in existing_hashes:
                    new_pdfs.append(url_data)
            
            logger.info(f"Found {len(new_pdfs)} new PDFs out of {len(urls)} total URLs")
            return new_pdfs
            
        except Exception as e:
            logger.error(f"Failed to find new PDFs: {e}")
            return []
    
    def _setup_driver(self):
        """Initialize Chrome driver if not already initialized."""
        if self.driver is None:
            try:
                self.driver = webdriver.Chrome(options=self.chrome_options)
                logger.info("Chrome driver initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Chrome driver: {e}")
                raise
    
    def _get_driver_from_pool(self):
        """Get a driver from the pool or create a new one."""
        try:
            return self.driver_pool.get_nowait()
        except:
            # Create new driver if pool is empty
            return webdriver.Chrome(options=self.chrome_options)
    
    def _return_driver_to_pool(self, driver):
        """Return a driver to the pool."""
        try:
            # Check if driver is still functional
            driver.current_url  # This will throw if driver is dead
            self.driver_pool.put(driver)
        except:
            # Driver is dead, quit it
            try:
                driver.quit()
            except:
                pass
    
    def _initialize_driver_pool(self, pool_size=4):
        """Initialize a pool of drivers for parallel processing."""
        logger.info(f"Initializing driver pool with {pool_size} drivers...")
        for i in range(pool_size):
            try:
                driver = webdriver.Chrome(options=self.chrome_options)
                self.driver_pool.put(driver)
            except Exception as e:
                logger.warning(f"Failed to create driver {i+1}: {e}")
        logger.info(f"Driver pool initialized with {self.driver_pool.qsize()} drivers")
    
    def _cleanup_driver_pool(self):
        """Clean up all drivers in the pool."""
        while not self.driver_pool.empty():
            try:
                driver = self.driver_pool.get_nowait()
                driver.quit()
            except:
                pass
    
    def _cleanup_driver(self):
        """Clean up Chrome driver."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.info("Chrome driver cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up driver: {e}")
    
    def _wait_for_download(self, timeout: int = DEFAULT_TIMEOUT):
        """Wait for file download to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            files = list(self.download_dir.glob("*"))
            if any(f.suffix == '.crdownload' for f in files):
                time.sleep(1)
                continue
            if any(f.suffix == '.csv' and not f.name.endswith('.crdownload') for f in files):
                return True
            time.sleep(1)
        return False
    
    def _find_latest_csv(self) -> Optional[Path]:
        """Find the most recently downloaded CSV file."""
        csv_files = list(self.download_dir.glob("*.csv"))
        if csv_files:
            return max(csv_files, key=lambda p: p.stat().st_mtime)
        return None
    
    def _rename_csv_file(self, csv_file: Path, proceeding: str) -> Path:
        """Rename CSV file to proceeding format."""
        new_name = f"{proceeding.lower()}.csv"
        new_path = self.download_dir / new_name
        
        # Check if the file already has the correct name
        if csv_file.name == new_name:
            logger.info(f"CSV file already has correct name: {csv_file}")
            return csv_file
        
        # If target file already exists and it's not the source file, remove it
        if new_path.exists() and new_path != csv_file:
            new_path.unlink()
            logger.info(f"Removed existing target file: {new_path}")
        
        # Only rename if source and target are different
        if csv_file != new_path:
            csv_file.rename(new_path)
            logger.info(f"Renamed CSV file from {csv_file.name} to: {new_path}")
        else:
            logger.info(f"CSV file already in correct location: {new_path}")
            
        return new_path
    
    def _extract_pdf_urls_from_csv(self, csv_file_path: Path) -> List[Dict[str, str]]:
        """
        Extract PDF URLs from CSV file using parallel processing for speed.
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            List of dictionaries with url, title, and source_type
        """
        pdf_urls = []
        try:
            df = pd.read_csv(csv_file_path)
            logger.info(f"Processing CSV with {len(df)} rows using parallel processing...")
            
            # Collect document URLs to process
            document_urls = []
            for index, row in df.iterrows():
                # Skip rows with 'Certificate Of Service'
                skip_row = False
                for cell_content in row.values:
                    if isinstance(cell_content, str) and 'Certificate Of Service' in cell_content:
                        skip_row = True
                        break
                
                if skip_row:
                    continue
                
                # Extract URL from Document Type column
                doc_type_cell = row.get("Document Type")
                if isinstance(doc_type_cell, str) and "<a href=" in doc_type_cell:
                    try:
                        soup = BeautifulSoup(doc_type_cell, "html.parser")
                        link = soup.find("a")
                        if link and link.get("href"):
                            document_urls.append(link["href"])
                    except Exception as e:
                        logger.warning(f"Error parsing HTML in CSV row {index}: {e}")
                        continue
            
            if not document_urls:
                logger.info("No document URLs found in CSV")
                return pdf_urls
            
            logger.info(f"Found {len(document_urls)} document URLs to process")
            
            # Initialize driver pool for parallel processing - optimize based on document count
            if len(document_urls) <= 10:
                max_workers = min(4, len(document_urls))  # Small batches: use fewer workers
            elif len(document_urls) <= 100:
                max_workers = min(8, len(document_urls))  # Medium batches: moderate workers
            else:
                max_workers = min(12, len(document_urls))  # Large batches: more workers
            
            self._initialize_driver_pool(max_workers)
            
            try:
                # Process URLs in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_url = {
                        executor.submit(self._extract_document_info_parallel, url): url 
                        for url in document_urls
                    }
                    
                    # Collect results as they complete
                    processed_count = 0
                    for future in as_completed(future_to_url):
                        url = future_to_url[future]
                        try:
                            result = future.result()
                            if result:
                                pdf_urls.append(result)
                            processed_count += 1
                            
                            # Log progress every 50 documents
                            if processed_count % 50 == 0:
                                logger.info(f"Processed {processed_count}/{len(document_urls)} documents")
                                
                        except Exception as e:
                            logger.warning(f"Error processing {url}: {e}")
                            continue
                
                logger.info(f"Parallel processing completed: {len(pdf_urls)} PDFs found from {len(document_urls)} documents")
                
            finally:
                # Clean up driver pool
                self._cleanup_driver_pool()
        
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_file_path}: {e}")
        
        return pdf_urls
    
    def _extract_document_info_parallel(self, document_page_url: str) -> Optional[Dict[str, str]]:
        """
        Thread-safe version of _extract_document_info for parallel processing.
        
        Args:
            document_page_url: URL of the document page
            
        Returns:
            Dictionary with url, title, and source_type, or None if failed
        """
        driver = None
        try:
            # Get driver from pool
            driver = self._get_driver_from_pool()
            
            # Load page with minimal wait time
            driver.get(document_page_url)
            
            # Use minimal wait time for speed (2 seconds for parallel processing)
            wait = WebDriverWait(driver, 2)
            
            # Extract title and PDF link in a single pass using optimized CSS selectors
            title = "Unknown Document"
            pdf_url = None
            
            try:
                # Combined CSS selector for faster title extraction
                title_element = wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, 
                    "#ResultTitleTD, .ResultTitleTD, h1, h2, .title, .document-title")))
                title = title_element.text.strip()
            except Exception:
                # Fallback: try without wait to get any available title
                try:
                    title_element = driver.find_element(By.CSS_SELECTOR, 
                        "#ResultTitleTD, .ResultTitleTD, h1, h2, .title, .document-title")
                    title = title_element.text.strip()
                except:
                    pass  # Keep default title
            
            # Optimized PDF link detection - single XPath with OR conditions
            try:
                pdf_element = driver.find_element(By.XPATH, 
                    "//a[contains(@href, '.PDF') or contains(@href, '.pdf') or contains(text(), 'PDF')]")
                href = pdf_element.get_attribute('href')
                if href and 'cpuc.ca.gov' in href.lower():
                    pdf_url = href
            except Exception:
                # Fallback: try download links
                try:
                    download_element = driver.find_element(By.XPATH, 
                        "//a[contains(text(), 'Download') or contains(text(), 'View')]")
                    href = download_element.get_attribute('href')
                    if href and 'cpuc.ca.gov' in href.lower():
                        pdf_url = href
                except Exception:
                    pass
            
            if pdf_url:
                # Fast title cleanup
                title = ' '.join(title.split()).strip()
                if 'Proceeding:' in title:
                    title = title.split('Proceeding:')[0].strip()
                
                # Quick filename sanitization
                title = self._sanitize_filename(title)
                if not title.lower().endswith('.pdf'):
                    title += '.pdf'
                
                return {
                    'url': pdf_url,
                    'title': title,
                    'source_type': 'csv_document_pdf'
                }
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Failed to extract from {document_page_url}: {e}")
            return None
        finally:
            # Return driver to pool
            if driver:
                self._return_driver_to_pool(driver)

    def _extract_document_info(self, document_page_url: str) -> Optional[Dict[str, str]]:
        """
        Extract ResultTitleTD and actual PDF URL from a CPUC document page (optimized for speed).
        
        Args:
            document_page_url: URL of the document page
            
        Returns:
            Dictionary with url, title, and source_type, or None if failed
        """
        try:
            # Ensure we have a driver initialized
            self._setup_driver()
            
            # Load page with minimal wait time
            self.driver.get(document_page_url)
            
            # Use shorter wait time for speed (2 seconds instead of 10)
            wait = WebDriverWait(self.driver, 2)
            
            # Extract title and PDF link in a single pass using optimized CSS selectors
            title = "Unknown Document"
            pdf_url = None
            
            try:
                # Combined CSS selector for faster title extraction
                title_element = wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, 
                    "#ResultTitleTD, .ResultTitleTD, h1, h2, .title, .document-title")))
                title = title_element.text.strip()
            except Exception:
                # Fallback: try without wait to get any available title
                try:
                    title_element = self.driver.find_element(By.CSS_SELECTOR, 
                        "#ResultTitleTD, .ResultTitleTD, h1, h2, .title, .document-title")
                    title = title_element.text.strip()
                except:
                    pass  # Keep default title
            
            # Optimized PDF link detection - single XPath with OR conditions
            try:
                # Combined XPath for faster PDF link detection
                pdf_element = self.driver.find_element(By.XPATH, 
                    "//a[contains(@href, '.PDF') or contains(@href, '.pdf') or contains(text(), 'PDF')]")
                href = pdf_element.get_attribute('href')
                if href and 'cpuc.ca.gov' in href.lower():
                    pdf_url = href
            except Exception:
                # Fallback: try download links
                try:
                    download_element = self.driver.find_element(By.XPATH, 
                        "//a[contains(text(), 'Download') or contains(text(), 'View')]")
                    href = download_element.get_attribute('href')
                    if href and 'cpuc.ca.gov' in href.lower():
                        pdf_url = href
                except Exception:
                    pass
            
            if pdf_url:
                # Fast title cleanup
                title = ' '.join(title.split()).strip()
                if 'Proceeding:' in title:
                    title = title.split('Proceeding:')[0].strip()
                
                # Quick filename sanitization
                title = self._sanitize_filename(title)
                if not title.lower().endswith('.pdf'):
                    title += '.pdf'
                
                return {
                    'url': pdf_url,
                    'title': title,
                    'source_type': 'csv_document_pdf'
                }
            else:
                return None
                
        except Exception as e:
            # Reduced logging for speed - only log errors, not warnings
            logger.debug(f"Failed to extract from {document_page_url}: {e}")
            return None
    
    def _trigger_vector_store_build(self, proceeding: str) -> bool:
        """
        Trigger vector store building for new PDFs.
        
        Args:
            proceeding: Proceeding number
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Initializing RAG system for vector store building: {proceeding}")
            
            # Import and initialize RAG system
            from rag_core import CPUCRAGSystem
            rag_system = CPUCRAGSystem(current_proceeding=proceeding)
            
            # Build vector store
            logger.info("Building vector store from scraped PDFs...")
            success = rag_system.build_vector_store()
            
            if success:
                logger.info("Vector store building completed successfully")
                return True
            else:
                logger.error("Vector store building failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to trigger vector store building: {e}")
            return False
    
    def _load_scraped_pdf_history(self, proceeding: str) -> Dict:
        """Load scraped PDF history for a proceeding."""
        history_file = self.download_dir / f"{proceeding.lower()}_scraped_pdf_history.json"
        
        with history_file_lock:
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to load scraped PDF history: {e}")
                    return {}
            else:
                return {}
    
    def _save_scraped_pdf_history(self, proceeding: str, history: Dict):
        """Save scraped PDF history for a proceeding."""
        history_file = self.download_dir / f"{proceeding.lower()}_scraped_pdf_history.json"
        
        with history_file_lock:
            try:
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)
                logger.info(f"Saved scraped PDF history to {history_file}")
            except Exception as e:
                logger.error(f"Failed to save scraped PDF history: {e}")
                raise
    
    def _create_url_hash(self, url: str) -> str:
        """Create MD5 hash of URL for unique identification."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing invalid characters."""
        return re.sub(r'[<>:"/\\|?*]', '_', filename)


# Convenience functions for backward compatibility
def scrape_proceeding_pdfs(proceeding: str, headless: bool = True) -> Dict:
    """
    Convenience function to scrape PDFs for a proceeding.
    
    Args:
        proceeding: Proceeding number (e.g., "R2207005")
        headless: Whether to run in headless mode
        
    Returns:
        Dictionary with scraping results
    """
    scraper = CPUCUnifiedScraper(headless=headless)
    return scraper.scrape_proceeding_pdfs(proceeding)


def get_new_pdfs_for_proceeding(proceeding: str, headless: bool = True) -> List[Dict[str, str]]:
    """
    Get only the new PDFs for a proceeding (those not in document_hashes).
    
    Args:
        proceeding: Proceeding number
        headless: Whether to run in headless mode
        
    Returns:
        List of new PDF URLs that need processing
    """
    results = scrape_proceeding_pdfs(proceeding, headless)
    return results.get('new_pdfs', [])


if __name__ == "__main__":
    # Test the unified scraper
    import sys
    
    from config import DEFAULT_PROCEEDING
    proceeding = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROCEEDING
    
    logging.basicConfig(level=logging.INFO)
    
    scraper = CPUCUnifiedScraper(headless=True)
    results = scraper.scrape_proceeding_pdfs(proceeding)
    
    print(f"\n=== Scraping Results for {proceeding} ===")
    print(f"CSV URLs: {len(results['csv_urls'])}")
    print(f"Google URLs: {len(results['google_urls'])}")
    print(f"New PDFs: {len(results['new_pdfs'])}")
    print(f"Total Scraped: {results['total_scraped']}")
    print(f"History File: {results['history_file']}")