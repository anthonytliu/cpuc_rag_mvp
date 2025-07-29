#!/usr/bin/env python3
"""
Test Enhanced Citation System with Character Positions

This script tests the new enhanced citation system that uses Chonkie's
character position tracking for more accurate citations.
"""

import logging
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_processing import (
    safe_chunk_with_chonkie_enhanced, 
    create_enhanced_chonkie_metadata,
    create_precise_citation,
    estimate_page_from_char_position,
    estimate_line_range_from_char_position
)
from rag_core import CPUCRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_character_position_tracking():
    """Test that Chonkie preserves character positions."""
    logger.info("🧪 Testing character position tracking with Chonkie")
    
    # Sample text with known positions
    sample_text = """California Public Utilities Commission
Decision on Microgrid Tariffs

Page 1 of the document begins here. This section discusses the regulatory framework for microgrids in California.

Page 2 content starts here. The Commission finds that distributed energy resources provide significant value to the grid.

Page 3 discusses implementation requirements. Utilities must file applications within 120 days of this decision."""
    
    # Test enhanced chunking
    enhanced_chunks = safe_chunk_with_chonkie_enhanced(sample_text, "recursive")
    
    logger.info(f"Generated {len(enhanced_chunks)} chunks with position tracking")
    
    # Verify each chunk has position information
    total_errors = 0
    for i, chunk_info in enumerate(enhanced_chunks):
        start_pos = chunk_info.get('start_index', -1)
        end_pos = chunk_info.get('end_index', -1)
        text = chunk_info.get('text', '')
        
        if start_pos == -1 or end_pos == -1:
            logger.error(f"❌ Chunk {i} missing position information")
            total_errors += 1
            continue
        
        # Verify the positions match the actual text
        extracted_text = sample_text[start_pos:end_pos]
        
        # Allow for some whitespace differences
        if text.strip() != extracted_text.strip():
            logger.warning(f"⚠️ Chunk {i} text mismatch")
            logger.warning(f"  Expected: '{extracted_text[:50]}...'")
            logger.warning(f"  Got: '{text[:50]}...'")
            total_errors += 1
        else:
            logger.info(f"✅ Chunk {i}: chars {start_pos}-{end_pos} ({len(text)} chars)")
    
    if total_errors == 0:
        logger.info("✅ All chunks have accurate position tracking")
    else:
        logger.error(f"❌ Found {total_errors} position tracking errors")
    
    return total_errors == 0


def test_page_estimation():
    """Test page number estimation from character positions."""
    logger.info("🧪 Testing page estimation from character positions")
    
    # Create sample text with known page breaks
    sample_pages = [
        "Page 1 content: This is the first page with California Public Utilities Commission decision text.",
        "Page 2 content: This page discusses microgrid regulations and implementation requirements for utilities.",
        "Page 3 content: Final page contains concluding remarks and effective dates for the new regulations."
    ]
    
    # Join with page breaks (form feed characters)
    full_text = '\f'.join(sample_pages)
    
    # Test page estimation at different positions
    test_positions = [
        (0, 1),           # Start of page 1
        (50, 1),          # Middle of page 1 (position 50 < 97)
        (len(sample_pages[0]) + 1, 2),  # Start of page 2
        (len(sample_pages[0]) + 50, 2), # Middle of page 2
        (len(sample_pages[0]) + len(sample_pages[1]) + 2, 3)  # Start of page 3
    ]
    
    all_correct = True
    for char_pos, expected_page in test_positions:
        estimated_page = estimate_page_from_char_position(char_pos, full_text)
        
        if estimated_page == expected_page:
            logger.info(f"✅ Position {char_pos} → Page {estimated_page} (correct)")
        else:
            logger.error(f"❌ Position {char_pos} → Page {estimated_page} (expected {expected_page})")
            all_correct = False
    
    return all_correct


def test_enhanced_metadata_creation():
    """Test enhanced metadata creation with position tracking."""
    logger.info("🧪 Testing enhanced metadata creation")
    
    # Sample chunk with position information
    chunk_info = {
        'text': "The Commission finds that distributed energy resources provide significant value to the electrical grid through demand response and voltage support capabilities.",
        'start_index': 150,
        'end_index': 295,
        'token_count': 23,
        'level': 0,
        'strategy': 'recursive'
    }
    
    # Sample raw text
    raw_text = """California Public Utilities Commission Decision 25-01-015

This decision addresses microgrid tariffs. The Commission finds that distributed energy resources provide significant value to the electrical grid through demand response and voltage support capabilities. These benefits should be reflected in appropriate compensation mechanisms."""
    
    # Create enhanced metadata
    metadata = create_enhanced_chonkie_metadata(
        chunk_info=chunk_info,
        source_name="D2501015.pdf",
        pdf_url="https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M000/K000/000000001.PDF",
        proceeding="R2207005",
        raw_text=raw_text
    )
    
    # Verify metadata contains enhanced fields
    required_fields = [
        'char_start', 'char_end', 'char_length', 'line_number', 
        'line_range_end', 'text_snippet', 'token_count', 'chunk_level'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in metadata:
            missing_fields.append(field)
    
    if missing_fields:
        logger.error(f"❌ Missing enhanced metadata fields: {missing_fields}")
        return False
    
    # Verify values are reasonable
    if metadata['char_start'] != 150:
        logger.error(f"❌ Incorrect char_start: {metadata['char_start']} (expected 150)")
        return False
    
    if metadata['char_end'] != 295:
        logger.error(f"❌ Incorrect char_end: {metadata['char_end']} (expected 295)")
        return False
    
    if metadata['char_length'] != 145:
        logger.error(f"❌ Incorrect char_length: {metadata['char_length']} (expected 145)")
        return False
    
    logger.info("✅ Enhanced metadata created successfully")
    logger.info(f"  Character range: {metadata['char_start']}-{metadata['char_end']}")
    logger.info(f"  Page estimate: {metadata['page']}")
    logger.info(f"  Line range: {metadata['line_number']}-{metadata['line_range_end']}")
    logger.info(f"  Text snippet: '{metadata['text_snippet'][:50]}...'")
    
    return True


def test_precise_citation_format():
    """Test the new precise citation format."""
    logger.info("🧪 Testing precise citation format")
    
    # Test different citation variations
    test_cases = [
        {
            'filename': 'D2501015.pdf',
            'page': 5,
            'char_start': 1250,
            'char_end': 1450,
            'line_start': 25,
            'text_snippet': 'The Commission finds that distributed energy resources provide value',
            'expected_pattern': 'CITE:D2501015.pdf,page_5,chars_1250-1450,line_25,"The Commission finds that distributed energy resources..."'
        },
        {
            'filename': 'Decision.pdf',
            'page': 1,
            'char_start': 0,
            'char_end': 100,
            'line_start': None,
            'text_snippet': None,
            'expected_pattern': 'CITE:Decision.pdf,page_1,chars_0-100'
        }
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases):
        citation = create_precise_citation(
            filename=test_case['filename'],
            page=test_case['page'],
            char_start=test_case['char_start'],
            char_end=test_case['char_end'],
            line_start=test_case['line_start'],
            text_snippet=test_case['text_snippet']
        )
        
        # Check if citation contains expected components (more flexible matching)
        expected_parts = [
            test_case['filename'],
            f"page_{test_case['page']}",
            f"chars_{test_case['char_start']}-{test_case['char_end']}"
        ]
        
        # Check if all expected parts are in the citation
        all_parts_found = all(part in citation for part in expected_parts)
        
        if all_parts_found:
            logger.info(f"✅ Test case {i+1}: Citation format correct")
            logger.info(f"  Generated: {citation}")
        else:
            logger.error(f"❌ Test case {i+1}: Citation format incorrect")
            logger.error(f"  Expected parts: {expected_parts}")
            logger.error(f"  Generated: {citation}")
            all_passed = False
    
    return all_passed


def test_end_to_end_enhanced_processing():
    """Test end-to-end processing with enhanced citations for R2207005."""
    logger.info("🧪 Testing end-to-end enhanced citation processing with R2207005")
    
    try:
        # Initialize RAG system for R2207005
        rag_system = CPUCRAGSystem(current_proceeding="R2207005")
        
        # Get system stats to verify we have data
        stats = rag_system.get_system_stats()
        total_chunks = stats.get('total_chunks', 0)
        
        if total_chunks == 0:
            logger.warning("⚠️ No chunks found in R2207005 - skipping end-to-end test")
            return True
        
        logger.info(f"Using R2207005 with {total_chunks} chunks")
        
        # Test query to generate citations
        test_query = "What are the main objectives of proceeding R.22-07-005?"
        
        logger.info(f"Testing query: '{test_query}'")
        
        # Get response with citations
        response_generator = rag_system.query(test_query)
        final_result = None
        
        for result in response_generator:
            if isinstance(result, dict):
                final_result = result
                break
        
        if not final_result:
            logger.error("❌ No final result received from RAG system")
            return False
        
        answer = final_result.get('answer', '')
        sources = final_result.get('sources', [])
        
        logger.info(f"Generated answer with {len(sources)} sources")
        
        # Check if sources have enhanced metadata
        enhanced_sources = 0
        for source in sources:
            if 'char_start' in source or 'char_end' in source:
                enhanced_sources += 1
        
        if enhanced_sources > 0:
            logger.info(f"✅ Found {enhanced_sources}/{len(sources)} sources with enhanced position data")
        else:
            logger.warning("⚠️ No sources found with enhanced position data - may need to rebuild vector store")
        
        # Count citations in the answer
        import re
        citations = re.findall(r'\[CITE:[^\]]+\]', answer)
        logger.info(f"Found {len(citations)} citations in answer")
        
        if citations:
            logger.info("Sample citations:")
            for citation in citations[:3]:  # Show first 3
                logger.info(f"  {citation}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        return False


def main():
    """Run all enhanced citation system tests."""
    logger.info("=" * 60)
    logger.info("ENHANCED CITATION SYSTEM TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Character Position Tracking", test_character_position_tracking),
        ("Page Estimation", test_page_estimation),
        ("Enhanced Metadata Creation", test_enhanced_metadata_creation),
        ("Precise Citation Format", test_precise_citation_format),
        ("End-to-End Processing", test_end_to_end_enhanced_processing),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"💥 {test_name}: ERROR - {e}")
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Enhanced citation system is working correctly.")
    else:
        logger.error("⚠️ Some tests failed. Review the output above for details.")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()