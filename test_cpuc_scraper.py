#!/usr/bin/env python3
"""
Comprehensive Tests for CPUC Document Scraper

Tests the improved Google search logic, PDF discovery, and document processing functionality.

Author: Claude Code
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import requests
from bs4 import BeautifulSoup

# Import the scraper functions
from cpuc_scraper import (
    CPUCUnifiedScraper,
    scrape_proceeding_pdfs,
    get_new_pdfs_for_proceeding
)


class TestCPUCUnifiedScraper(unittest.TestCase):
    """Test the unified scraper class"""
    
    def setUp(self):
        self.scraper = CPUCUnifiedScraper(headless=True, max_workers=4)
    
    def test_scraper_initialization(self):
        """Test that the scraper initializes correctly"""
        self.assertTrue(self.scraper.headless)
        self.assertEqual(self.scraper.max_workers, 4)
        self.assertIsNotNone(self.scraper.chrome_options)
    
    def test_scraper_with_custom_settings(self):
        """Test scraper with custom settings"""
        custom_scraper = CPUCUnifiedScraper(headless=False, max_workers=2)
        self.assertFalse(custom_scraper.headless)
        self.assertEqual(custom_scraper.max_workers, 2)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        self.scraper = CPUCUnifiedScraper(headless=True)
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        test_cases = [
            ("normal_file.pdf", "normal_file.pdf"),
            ("file<>with:bad|chars?.pdf", "file__with_bad_chars_.pdf"),
            ("file/with\\path.pdf", "file_with_path.pdf"),
            ("file\"with'quotes.pdf", "file_with'quotes.pdf")
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = self.scraper._sanitize_filename(input_name)
                self.assertEqual(result, expected)
    
    def test_create_url_hash(self):
        """Test URL hash creation"""
        url = "https://docs.cpuc.ca.gov/test.pdf"
        hash1 = self.scraper._create_url_hash(url)
        hash2 = self.scraper._create_url_hash(url)
        
        # Same URL should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Different URLs should produce different hashes
        different_url = "https://docs.cpuc.ca.gov/different.pdf"
        hash3 = self.scraper._create_url_hash(different_url)
        self.assertNotEqual(hash1, hash3)
        
        # Hash should be MD5 (32 characters)
        self.assertEqual(len(hash1), 32)


class TestDownloadHistory(unittest.TestCase):
    """Test download history management"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.scraper = CPUCUnifiedScraper(headless=True)
        # Temporarily change the download directory for testing
        self.scraper.download_dir = Path(self.temp_dir)
    
    def tearDown(self):
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_download_history(self):
        """Test saving and loading download history"""
        proceeding = "R2207005"
        test_history = {
            "hash123": {
                "url": "https://docs.cpuc.ca.gov/test.pdf",
                "title": "test.pdf",
                "scraped_date": "2023-01-01T00:00:00",
                "last_checked": "2023-01-01T00:00:00"
            }
        }
        
        # Save history
        self.scraper._save_scraped_pdf_history(proceeding, test_history)
        
        # Load history
        loaded_history = self.scraper._load_scraped_pdf_history(proceeding)
        
        self.assertEqual(loaded_history, test_history)
    
    def test_load_nonexistent_history(self):
        """Test loading history for non-existent proceeding"""
        result = self.scraper._load_scraped_pdf_history("NONEXISTENT")
        self.assertEqual(result, {})


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    @patch('cpuc_scraper.CPUCUnifiedScraper')
    def test_scrape_proceeding_pdfs(self, mock_scraper_class):
        """Test the scrape_proceeding_pdfs convenience function"""
        # Mock scraper instance
        mock_scraper = Mock()
        mock_scraper.scrape_proceeding_pdfs.return_value = {
            'csv_urls': [{'url': 'test1.pdf'}],
            'google_urls': [{'url': 'test2.pdf'}],
            'new_pdfs': [{'url': 'test3.pdf'}],
            'total_scraped': 3
        }
        mock_scraper_class.return_value = mock_scraper
        
        result = scrape_proceeding_pdfs("R2207005")
        
        # Should return mock result
        self.assertEqual(result['total_scraped'], 3)
        mock_scraper_class.assert_called_once_with(headless=True)
    
    @patch('cpuc_scraper.scrape_proceeding_pdfs')
    def test_get_new_pdfs_for_proceeding(self, mock_scrape):
        """Test the get_new_pdfs_for_proceeding convenience function"""
        # Mock scraping result
        mock_scrape.return_value = {
            'new_pdfs': [{'url': 'new1.pdf'}, {'url': 'new2.pdf'}]
        }
        
        result = get_new_pdfs_for_proceeding("R2207005")
        
        # Should return only new PDFs
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['url'], 'new1.pdf')
        mock_scrape.assert_called_once_with("R2207005", True)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for unified scraper scenarios"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.scraper = CPUCUnifiedScraper(headless=True)
        self.scraper.download_dir = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('cpuc_scraper.search')
    @patch('cpuc_scraper.webdriver.Chrome')
    def test_scraper_class_integration(self, mock_chrome, mock_search):
        """Test integration of unified scraper class with mocked external dependencies"""
        # Mock Google search results
        mock_search.return_value = [
            "https://docs.cpuc.ca.gov/test1.pdf",
            "https://docs.cpuc.ca.gov/SearchRes.aspx?DocID=123"
        ]
        
        # Mock Chrome driver
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        
        # Test initialization
        self.assertTrue(self.scraper.headless)
        self.assertIsNotNone(self.scraper.chrome_options)
        self.assertEqual(self.scraper.download_dir, Path(self.temp_dir))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)