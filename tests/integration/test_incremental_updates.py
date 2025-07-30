#!/usr/bin/env python3
"""
Test to verify incremental update functionality in the CPUC scraper.

This test specifically verifies that scraped_pdf_history.json is updated 
live as PDFs are discovered, not just at the end.

Author: Claude Code
"""

import json
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch

from cpuc_scraper import CPUCSimplifiedScraper


def test_incremental_update_functionality():
    """
    Test that scraped_pdf_history.json is updated incrementally as PDFs are found.
    This simulates the real scraping scenario with 2000+ PDFs where failure could 
    occur at any point, and we need live updates.
    """
    print(f"\n🧪 Testing Incremental Update Functionality")
    print("=" * 60)
    
    # Setup isolated test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Test directory: {temp_dir}")
        
        # Create test proceeding structure
        proceeding = "R2207005"
        proceeding_folder = Path(temp_dir) / "cpuc_proceedings" / proceeding
        proceeding_folder.mkdir(parents=True, exist_ok=True)
        
        # Create scraper instance
        scraper = CPUCSimplifiedScraper(headless=True)
        
        # Test 1: Initialize empty JSON file
        print(f"\n📝 Test 1: Initialize PDF history file")
        scraper._initialize_pdf_history_file(proceeding_folder)
        
        history_file = proceeding_folder / f"{proceeding}_scraped_pdf_history.json"
        assert history_file.exists(), "History file should be created"
        
        with open(history_file, 'r') as f:
            initial_data = json.load(f)
        assert initial_data == {}, "Initial file should be empty"
        print(f"✅ Initialized empty history file: {history_file}")
        
        # Test 2: Add PDFs incrementally and verify live updates
        print(f"\n💾 Test 2: Incremental PDF additions")
        
        test_pdfs = [
            {
                'pdf_url': 'https://docs.cpuc.ca.gov/test1.pdf',
                'title': 'Test PDF 1',
                'document_type': 'Decision',
                'pdf_creation_date': '2024-01-01',
                'scrape_date': '2024-01-01',
                'source': 'csv',
                'parent_page_url': 'https://docs.cpuc.ca.gov/page1.html',
                'pdf_metadata': {'file_size': '1024'}
            },
            {
                'pdf_url': 'https://docs.cpuc.ca.gov/test2.pdf',
                'title': 'Test PDF 2',
                'document_type': 'Order',
                'pdf_creation_date': '2024-01-02',
                'scrape_date': '2024-01-02',
                'source': 'google search',
                'parent_page_url': 'Google Search: R.22-07-005',
                'pdf_metadata': {'file_size': '2048'}
            },
            {
                'pdf_url': 'https://docs.cpuc.ca.gov/test3.pdf',
                'title': 'Test PDF 3',
                'document_type': 'Ruling',
                'pdf_creation_date': '2024-01-03',
                'scrape_date': '2024-01-03',
                'source': 'csv',
                'parent_page_url': 'https://docs.cpuc.ca.gov/page3.html',
                'pdf_metadata': {'file_size': '4096'}
            }
        ]
        
        # Track file modifications to verify live updates
        modification_times = []
        
        for i, pdf_info in enumerate(test_pdfs, 1):
            print(f"   📄 Adding PDF {i}: {pdf_info['title']}")
            
            # Record modification time before update
            before_time = history_file.stat().st_mtime if history_file.exists() else 0
            
            # Add PDF incrementally
            scraper._save_single_pdf_to_history(proceeding_folder, pdf_info)
            
            # Record modification time after update
            after_time = history_file.stat().st_mtime
            modification_times.append(after_time)
            
            # Verify file was modified
            assert after_time > before_time, f"File should be modified after adding PDF {i}"
            
            # Verify the PDF was added to the file immediately
            with open(history_file, 'r') as f:
                current_data = json.load(f)
            
            assert len(current_data) == i, f"Should have {i} PDFs after adding PDF {i}"
            
            # Verify the specific PDF is in the file
            pdf_found = False
            for entry in current_data.values():
                if entry['url'] == pdf_info['pdf_url']:
                    pdf_found = True
                    assert entry['title'] == pdf_info['title']
                    assert entry['source'] == pdf_info['source']
                    break
            
            assert pdf_found, f"PDF {i} should be found in history file"
            print(f"   ✅ PDF {i} successfully added and verified in file")
            
            # Small delay to ensure different timestamps
            time.sleep(0.1)
        
        # Test 3: Verify all modification times are different (proving incremental updates)
        print(f"\n⏰ Test 3: Verify incremental timestamps")
        unique_times = set(modification_times)
        assert len(unique_times) == len(modification_times), "Each update should have different timestamp"
        print(f"✅ All {len(modification_times)} updates had unique timestamps")
        
        # Test 4: Verify final file structure
        print(f"\n📊 Test 4: Verify final file structure")
        with open(history_file, 'r') as f:
            final_data = json.load(f)
        
        assert len(final_data) == 3, "Should have 3 total PDFs"
        
        # Verify required fields for each entry
        required_fields = ['url', 'title', 'document_type', 'source', 'status', 'scrape_date', 'parent_url']
        for pdf_hash, entry in final_data.items():
            for field in required_fields:
                assert field in entry, f"Entry {pdf_hash} should have field {field}"
        
        print(f"✅ Final file contains {len(final_data)} entries with all required fields")
        
        # Test 5: Verify thread safety (multiple concurrent updates)
        print(f"\n🔒 Test 5: Thread safety test")
        
        def add_pdf_worker(worker_id):
            """Worker function to add PDFs concurrently"""
            pdf_info = {
                'pdf_url': f'https://docs.cpuc.ca.gov/worker{worker_id}.pdf',
                'title': f'Worker PDF {worker_id}',
                'document_type': 'Test',
                'pdf_creation_date': '2024-01-01',
                'scrape_date': '2024-01-01',
                'source': 'test',
                'parent_page_url': f'https://docs.cpuc.ca.gov/worker{worker_id}.html'
            }
            scraper._save_single_pdf_to_history(proceeding_folder, pdf_info)
        
        # Launch 5 concurrent workers
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=add_pdf_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join()
        
        # Verify all worker PDFs were added
        with open(history_file, 'r') as f:
            threaded_data = json.load(f)
        
        assert len(threaded_data) == 8, "Should have 8 PDFs (3 original + 5 workers)"
        
        worker_urls = [f'https://docs.cpuc.ca.gov/worker{i}.pdf' for i in range(5)]
        found_workers = 0
        for entry in threaded_data.values():
            if entry['url'] in worker_urls:
                found_workers += 1
        
        assert found_workers == 5, "All 5 worker PDFs should be found"
        print(f"✅ Thread safety verified: all 5 concurrent updates successful")
        
        print(f"\n🎉 INCREMENTAL UPDATE FUNCTIONALITY TEST PASSED!")
        print(f"📈 Summary:")
        print(f"   • ✅ Initialization: Empty file created")
        print(f"   • ✅ Incremental updates: 3 PDFs added with unique timestamps")
        print(f"   • ✅ File structure: All required fields present")
        print(f"   • ✅ Thread safety: 5 concurrent updates successful")
        print(f"   • ✅ Final count: {len(threaded_data)} total PDFs")
        print("=" * 60)


def test_large_scale_simulation():
    """
    Simulate a large-scale scraping scenario to verify performance 
    and reliability of incremental updates.
    """
    print(f"\n🚀 Large Scale Incremental Update Simulation")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Test directory: {temp_dir}")
        
        proceeding = "R1807006"
        proceeding_folder = Path(temp_dir) / "cpuc_proceedings" / proceeding
        proceeding_folder.mkdir(parents=True, exist_ok=True)
        
        scraper = CPUCSimplifiedScraper(headless=True)
        scraper._initialize_pdf_history_file(proceeding_folder)
        
        history_file = proceeding_folder / f"{proceeding}_scraped_pdf_history.json"
        
        # Simulate adding 100 PDFs (representing a fraction of 2000+)
        print(f"📄 Simulating 100 PDF additions...")
        
        start_time = time.time()
        
        for i in range(100):
            pdf_info = {
                'pdf_url': f'https://docs.cpuc.ca.gov/simulation{i:03d}.pdf',
                'title': f'Simulation PDF {i:03d}',
                'document_type': 'Simulation',
                'pdf_creation_date': '2024-01-01',
                'scrape_date': '2024-01-01',
                'source': 'simulation',
                'parent_page_url': f'https://docs.cpuc.ca.gov/sim{i:03d}.html',
                'pdf_metadata': {'file_size': str(1024 * (i + 1))}
            }
            
            scraper._save_single_pdf_to_history(proceeding_folder, pdf_info)
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"   📊 Progress: {i + 1}/100 PDFs added")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify final state
        with open(history_file, 'r') as f:
            simulation_data = json.load(f)
        
        print(f"✅ Large scale simulation completed!")
        print(f"   • Total PDFs: {len(simulation_data)}")
        print(f"   • Duration: {duration:.2f} seconds")
        print(f"   • Rate: {len(simulation_data) / duration:.1f} PDFs/second")
        print(f"   • File size: {history_file.stat().st_size:,} bytes")
        
        assert len(simulation_data) == 100, "Should have exactly 100 PDFs"
        
        # Verify data integrity
        for i in range(100):
            expected_url = f'https://docs.cpuc.ca.gov/simulation{i:03d}.pdf'
            url_found = any(entry['url'] == expected_url for entry in simulation_data.values())
            assert url_found, f"URL {expected_url} should be found"
        
        print(f"✅ Data integrity verified: all 100 PDFs accounted for")
        print("=" * 60)


if __name__ == '__main__':
    print("🧪 Running Incremental Update Tests")
    
    try:
        test_incremental_update_functionality()
        test_large_scale_simulation()
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Incremental update functionality is working correctly")
        print(f"✅ The scraper will now update scraped_pdf_history.json live")
        print(f"✅ No data loss risk for large scraping operations (2000+ PDFs)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)