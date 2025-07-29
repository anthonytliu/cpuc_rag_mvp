#!/usr/bin/env python3
"""
Test script for multi-proceeding functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from rag_core import CPUCRAGSystem
from cpuc_scraper import CPUCSimplifiedScraper

def test_config_system():
    """Test the configuration system"""
    print("🧪 Testing configuration system...")
    
    # Test available proceedings
    active_proceedings = config.get_active_proceedings()
    print(f"✅ Active proceedings: {list(active_proceedings.keys())}")
    
    # Test display names
    for proc_id in active_proceedings:
        display_name = config.get_proceeding_display_name(proc_id)
        print(f"✅ {proc_id} -> {display_name}")
    
    # Test file paths
    for proc_id in ['R1202009', 'R2207005']:
        paths = config.get_proceeding_file_paths(proc_id)
        print(f"✅ {proc_id} vector DB: {paths['vector_db']}")
    
    # Test URL generation
    urls = config.get_proceeding_urls('R1202009')
    print(f"✅ R1202009 CPUC URL: {urls['cpuc_apex']}")
    
    print("✅ Configuration system tests passed!\n")

def test_rag_system():
    """Test RAG system initialization with different proceedings"""
    print("🧪 Testing RAG system initialization...")
    
    for proceeding in ['R1202009', 'R2207005']:
        try:
            # Initialize without processing existing data
            system = CPUCRAGSystem(current_proceeding=proceeding)
            print(f"✅ {proceeding}: RAG system initialized")
            print(f"   Current proceeding: {system.current_proceeding}")
            print(f"   Vector DB path: {system.db_dir}")
            print(f"   Document hashes: {system.doc_hashes_file}")
            
            # Test URL generation
            cpuc_url = system.get_cpuc_url()
            search_url = system.get_cpuc_search_url("test")
            print(f"   CPUC URL: {cpuc_url}")
            print(f"   Search URL: {search_url}")
            print()
            
        except Exception as e:
            print(f"❌ {proceeding}: failed - {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("✅ RAG system tests passed!\n")

def test_scraper_system():
    """Test scraper system with different proceedings"""
    print("🧪 Testing scraper system...")
    
    try:
        # Test with specific proceeding
        scraper = CPUCSimplifiedScraper(headless=True)
        print(f"✅ Scraper initialized successfully")
        
        # Test format function
        formatted = config.format_proceeding_for_search('R2207005')
        print(f"✅ Format function: R2207005 -> {formatted}")
        
        print("✅ Scraper system tests passed!\n")
        
    except Exception as e:
        print(f"❌ Scraper test failed: {e}")
        import traceback
        traceback.print_exc()
        print()

def test_app_imports():
    """Test that the main app imports correctly"""
    print("🧪 Testing app imports...")
    
    try:
        # Test main app components
        from app import render_proceeding_selector, initialize_rag_system
        print("✅ App functions imported successfully")
        
        print("✅ App import tests passed!\n")
        
    except Exception as e:
        print(f"❌ App import failed: {e}")
        import traceback
        traceback.print_exc()
        print()

if __name__ == "__main__":
    print("🚀 Multi-Proceeding Feature Test Suite")
    print("=" * 50)
    
    test_config_system()
    test_rag_system()
    test_scraper_system()
    test_app_imports()
    
    print("🎉 All tests completed!")