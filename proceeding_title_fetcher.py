#!/usr/bin/env python3
"""
Proceeding Title Fetcher Agent

This script fetches descriptive titles for CPUC proceedings by:
1. Taking proceeding IDs from SCRAPER_PROCEEDINGS in config.py
2. Formatting them using format_proceeding_for_search function
3. Performing Google searches for each formatted proceeding
4. Extracting titles from CPUC pages
5. Creating a JSON mapping file for UI dropdowns

Features:
- Rate limiting for Google searches
- Robust error handling and retry logic
- Progress tracking and status updates
- Graceful handling of network errors
- Validation of search results

Author: Claude Code
Generated: 2025-01-29
"""

import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import hashlib
from datetime import datetime

import requests
from bs4 import BeautifulSoup
try:
    from googlesearch import search
except ImportError:
    print("Warning: googlesearch-python not available. Install with: pip install googlesearch-python")
    search = None
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('proceeding_title_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProceedingTitleFetcher:
    """Fetches proceeding titles from CPUC website via Google search."""
    
    def __init__(self):
        """Initialize the proceeding title fetcher."""
        self.project_root = config.PROJECT_ROOT
        self.output_file = self.project_root / "proceeding_titles.json"
        self.cache_file = self.project_root / "proceeding_titles_cache.json"
        
        # Rate limiting settings
        self.search_delay = config.GOOGLE_SEARCH_DELAY_SECONDS
        self.max_retries = config.GOOGLE_SEARCH_MAX_RETRIES
        self.retry_delay = config.GOOGLE_SEARCH_RETRY_DELAY
        
        # Search configuration
        self.cpuc_domains = ['cpuc.ca.gov', 'docs.cpuc.ca.gov', 'apps.cpuc.ca.gov']
        self.max_search_results = 10
        
        # Results storage
        self.proceeding_titles = {}
        self.errors = []
        self.cache = self._load_cache()
        
        logger.info("Proceeding Title Fetcher initialized")
        logger.info(f"Output file: {self.output_file}")
        logger.info(f"Cache file: {self.cache_file}")
        logger.info(f"Search delay: {self.search_delay}s")
        logger.info(f"Max retries: {self.max_retries}")
    
    def _load_cache(self) -> Dict:
        """Load cached results to avoid re-fetching recent data."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    logger.info(f"Loaded {len(cache_data)} cached entries")
                    return cache_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save current results to cache."""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'proceeding_titles': self.proceeding_titles,
                'metadata': {
                    'total_proceedings': len(config.SCRAPER_PROCEEDINGS),
                    'successful_fetches': len(self.proceeding_titles),
                    'error_count': len(self.errors)
                }
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Cache saved with {len(self.proceeding_titles)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _format_proceeding_for_search(self, proceeding_id: str) -> str:
        """Format proceeding ID for search using config function."""
        return config.format_proceeding_for_search(proceeding_id)
    
    def _perform_google_search(self, query: str, max_results: int = 10) -> List[str]:
        """
        Perform Google search with rate limiting and error handling.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of URLs from search results
        """
        urls = []
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                logger.debug(f"Searching Google for: '{query}' (attempt {retry_count + 1})")
                
                # Perform the search with site restriction to CPUC
                search_query = f"{query} site:cpuc.ca.gov"
                
                # Use googlesearch library with rate limiting
                if search is None:
                    logger.error("Google search library not available")
                    return []
                
                # Use googlesearch library with simplified parameters
                search_results = search(
                    search_query,
                    stop=max_results,
                    pause=self.search_delay
                )
                
                urls = list(search_results)
                logger.debug(f"Found {len(urls)} search results")
                
                # Filter for CPUC domains
                cpuc_urls = []
                for url in urls:
                    parsed = urlparse(url)
                    if any(domain in parsed.netloc for domain in self.cpuc_domains):
                        cpuc_urls.append(url)
                
                if cpuc_urls:
                    logger.debug(f"Filtered to {len(cpuc_urls)} CPUC URLs")
                    return cpuc_urls
                else:
                    logger.warning(f"No CPUC URLs found in search results for: {query}")
                    break
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Google search failed (attempt {retry_count}): {e}")
                
                if retry_count <= self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Max retries exceeded for query: {query}")
                    self.errors.append({
                        'proceeding_id': query.split()[0] if query else 'unknown',
                        'error': f"Google search failed: {str(e)}",
                        'type': 'search_error'
                    })
        
        return urls
    
    def _extract_title_from_page(self, url: str) -> Optional[str]:
        """
        Extract the page title from a CPUC URL.
        
        Args:
            url: URL to fetch and extract title from
            
        Returns:
            Extracted title or None if extraction fails
        """
        try:
            logger.debug(f"Fetching page: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try multiple methods to extract title
            title = None
            
            # Method 1: Standard HTML title tag
            title_tag = soup.find('title')
            if title_tag and title_tag.text.strip():
                title = title_tag.text.strip()
            
            # Method 2: Look for specific CPUC page elements
            if not title:
                # Look for h1 tags that might contain the proceeding title
                h1_tags = soup.find_all('h1')
                for h1 in h1_tags:
                    if h1.text and 'R.' in h1.text:
                        title = h1.text.strip()
                        break
            
            # Method 3: Look for meta description
            if not title:
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    title = meta_desc.get('content').strip()
            
            if title:
                # Clean up the title
                title = self._clean_title(title)
                logger.debug(f"Extracted title: {title}")
                return title
            else:
                logger.warning(f"No title found on page: {url}")
                return None
                
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch page {url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error extracting title from {url}: {e}")
            return None
    
    def _clean_title(self, title: str) -> str:
        """
        Clean and normalize the extracted title.
        
        Args:
            title: Raw title string
            
        Returns:
            Cleaned title string
        """
        if not title:
            return ""
        
        # Remove common CPUC website prefixes/suffixes
        title = re.sub(r'^.*?CPUC\s*-\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*-\s*CPUC.*?$', '', title, flags=re.IGNORECASE)
        title = re.sub(r'^.*?California Public Utilities Commission\s*-\s*', '', title, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        # Truncate if too long
        if len(title) > 200:
            title = title[:197] + "..."
        
        return title
    
    def _is_cached_and_recent(self, proceeding_id: str) -> bool:
        """Check if proceeding title is cached and recent enough."""
        if proceeding_id not in self.cache.get('proceeding_titles', {}):
            return False
        
        # For now, consider all cached entries as recent
        # In future, could add timestamp checking
        return True
    
    def fetch_proceeding_title(self, proceeding_id: str) -> Optional[str]:
        """
        Fetch the title for a single proceeding.
        
        Args:
            proceeding_id: Proceeding ID (e.g., 'R2207005')
            
        Returns:
            Proceeding title or None if not found
        """
        # Check cache first
        if self._is_cached_and_recent(proceeding_id):
            cached_title = self.cache['proceeding_titles'][proceeding_id]
            logger.info(f"Using cached title for {proceeding_id}: {cached_title}")
            return cached_title
        
        try:
            # Format for search
            formatted_proceeding = self._format_proceeding_for_search(proceeding_id)
            logger.info(f"Fetching title for {proceeding_id} (formatted: {formatted_proceeding})")
            
            # Perform Google search
            search_urls = self._perform_google_search(formatted_proceeding, self.max_search_results)
            
            if not search_urls:
                logger.warning(f"No search results found for {proceeding_id}")
                self.errors.append({
                    'proceeding_id': proceeding_id,
                    'error': 'No search results found',
                    'type': 'no_results'
                })
                return None
            
            # Try to extract title from the first CPUC URL
            for url in search_urls:
                title = self._extract_title_from_page(url)
                if title and formatted_proceeding.replace('.', '').replace('-', '') in title.replace('.', '').replace('-', ''):
                    logger.info(f"Successfully found title for {proceeding_id}: {title}")
                    return title
            
            # If no matching title found, use the first title with a fallback format
            if search_urls:
                first_title = self._extract_title_from_page(search_urls[0])
                if first_title:
                    logger.info(f"Using first available title for {proceeding_id}: {first_title}")
                    return first_title
            
            logger.warning(f"Could not extract title for {proceeding_id}")
            self.errors.append({
                'proceeding_id': proceeding_id,
                'error': 'Could not extract title from CPUC pages',
                'type': 'extraction_error'
            })
            return None
            
        except Exception as e:
            logger.error(f"Error fetching title for {proceeding_id}: {e}")
            self.errors.append({
                'proceeding_id': proceeding_id,
                'error': str(e),
                'type': 'general_error'
            })
            return None
    
    def fetch_all_proceeding_titles(self) -> Dict[str, str]:
        """
        Fetch titles for all proceedings in SCRAPER_PROCEEDINGS.
        
        Returns:
            Dictionary mapping proceeding IDs to titles
        """
        logger.info("="*60)
        logger.info("STARTING PROCEEDING TITLE FETCH")
        logger.info(f"Total proceedings to process: {len(config.SCRAPER_PROCEEDINGS)}")
        logger.info("="*60)
        
        total_proceedings = len(config.SCRAPER_PROCEEDINGS)
        
        for i, proceeding_id in enumerate(config.SCRAPER_PROCEEDINGS, 1):
            logger.info(f"Processing {i}/{total_proceedings}: {proceeding_id}")
            
            try:
                title = self.fetch_proceeding_title(proceeding_id)
                
                if title:
                    self.proceeding_titles[proceeding_id] = title
                    logger.info(f"✅ Success: {proceeding_id} -> {title}")
                else:
                    # Use fallback title
                    formatted_proceeding = self._format_proceeding_for_search(proceeding_id)
                    fallback_title = f"{formatted_proceeding} - CPUC Proceeding"
                    self.proceeding_titles[proceeding_id] = fallback_title
                    logger.warning(f"⚠️ Using fallback: {proceeding_id} -> {fallback_title}")
                
                # Save intermediate results periodically
                if i % 10 == 0:
                    self._save_cache()
                    logger.info(f"Intermediate save completed ({i}/{total_proceedings})")
                
                # Rate limiting between requests
                if i < total_proceedings:  # Don't sleep after the last request
                    time.sleep(self.search_delay)
                    
            except KeyboardInterrupt:
                logger.warning("Process interrupted by user")
                logger.info(f"Partial results: {len(self.proceeding_titles)} titles fetched")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing {proceeding_id}: {e}")
                continue
        
        # Final save
        self._save_cache()
        
        logger.info("="*60)
        logger.info("PROCEEDING TITLE FETCH COMPLETED")
        logger.info(f"Successfully fetched: {len(self.proceeding_titles)}")
        logger.info(f"Errors encountered: {len(self.errors)}")
        logger.info("="*60)
        
        return self.proceeding_titles
    
    def save_results(self) -> str:
        """
        Save the fetched titles to JSON file.
        
        Returns:
            Path to the saved file
        """
        try:
            # Prepare the output data
            output_data = {
                'generated_at': datetime.now().isoformat(),
                'total_proceedings': len(config.SCRAPER_PROCEEDINGS),
                'successful_fetches': len(self.proceeding_titles),
                'error_count': len(self.errors),
                'proceeding_titles': self.proceeding_titles,
                'errors': self.errors if self.errors else [],
                'metadata': {
                    'source': 'CPUC website via Google search',
                    'method': 'Automated title extraction',
                    'rate_limit_delay': self.search_delay,
                    'max_retries': self.max_retries
                }
            }
            
            # Save to JSON file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {self.output_file}")
            logger.info(f"Total entries: {len(self.proceeding_titles)}")
            
            return str(self.output_file)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def generate_report(self) -> str:
        """
        Generate a summary report of the fetching process.
        
        Returns:
            Formatted report string
        """
        total_proceedings = len(config.SCRAPER_PROCEEDINGS)
        successful_fetches = len(self.proceeding_titles)
        error_count = len(self.errors)
        success_rate = (successful_fetches / total_proceedings * 100) if total_proceedings > 0 else 0
        
        report = f"""
PROCEEDING TITLE FETCH REPORT
=============================

Summary:
--------
Total Proceedings: {total_proceedings}
Successfully Fetched: {successful_fetches}
Errors: {error_count}
Success Rate: {success_rate:.1f}%

Successful Titles:
------------------
"""
        
        for proceeding_id, title in sorted(self.proceeding_titles.items()):
            report += f"  {proceeding_id}: {title}\n"
        
        if self.errors:
            report += f"\nErrors Encountered:\n"
            report += f"-------------------\n"
            for error in self.errors:
                report += f"  {error['proceeding_id']}: {error['error']} ({error['type']})\n"
        
        report += f"\nOutput File: {self.output_file}\n"
        report += f"Cache File: {self.cache_file}\n"
        
        return report


def main():
    """Main function to run the proceeding title fetcher."""
    print("CPUC Proceeding Title Fetcher")
    print("============================")
    print()
    
    try:
        # Initialize the fetcher
        fetcher = ProceedingTitleFetcher()
        
        # Fetch all titles
        titles = fetcher.fetch_all_proceeding_titles()
        
        # Save results
        output_file = fetcher.save_results()
        
        # Generate and display report
        report = fetcher.generate_report()
        print(report)
        
        # Save report to file
        report_file = fetcher.project_root / "proceeding_title_fetch_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_file}")
        print(f"\nProcess completed successfully!")
        print(f"JSON file: {output_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())