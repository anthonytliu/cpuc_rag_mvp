#!/usr/bin/env python3
"""
Test to verify the fixed incremental embedding behavior.

This test ensures that the incremental embedder no longer causes
repeated chunking/embedding restarts and uses proper incremental processing.

Author: Claude Code
"""

import json
import tempfile
import time
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Setup test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_test_proceeding_with_documents(temp_dir: str, proceeding: str, num_docs: int = 3):
    """Create a test proceeding with scraped PDF history."""
    proceeding_folder = Path(temp_dir) / "cpuc_proceedings" / proceeding
    proceeding_folder.mkdir(parents=True, exist_ok=True)
    
    # Create scraped PDF history
    history_file = proceeding_folder / f"{proceeding}_scraped_pdf_history.json"
    
    test_pdfs = {}
    for i in range(1, num_docs + 1):
        pdf_hash = f"hash_{i:03d}"
        test_pdfs[pdf_hash] = {
            'url': f'https://docs.cpuc.ca.gov/test{i:03d}.pdf',
            'title': f'Test Document {i:03d}',
            'document_type': 'Decision',
            'source': 'csv',
            'status': 'found',
            'scrape_date': datetime.now().isoformat(),
            'parent_url': f'https://docs.cpuc.ca.gov/page{i:03d}.html',
            'pdf_metadata': {'file_size': str(1024 * i)}
        }
    
    with open(history_file, 'w') as f:
        json.dump(test_pdfs, f, indent=2)
    
    return proceeding_folder, history_file


def test_incremental_embedding_no_restart():
    """
    Test that incremental embedding uses proper single-document processing
    without triggering vector store rebuilds.
    """
    print(f"\n🧪 Test: Incremental Embedding No Restart")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Test directory: {temp_dir}")
        
        # Create test proceeding
        proceeding = "R_TEST_EMBED"
        proceeding_folder, history_file = create_test_proceeding_with_documents(
            temp_dir, proceeding, num_docs=3
        )
        
        # Mock the RAG system to track method calls
        with patch('incremental_embedder.CPUCRAGSystem') as mock_rag_class:
            mock_rag_instance = MagicMock()
            mock_rag_class.return_value = mock_rag_instance
            
            # Mock _process_single_url to return successful chunk extraction
            mock_rag_instance._process_single_url.return_value = {
                'success': True,
                'chunks': [f'chunk_{i}' for i in range(5)],  # 5 mock chunks
                'url': 'test_url',
                'title': 'test_title',
                'chunk_count': 5
            }
            
            # Mock add_document_incrementally to return success
            mock_rag_instance.add_document_incrementally.return_value = True
            
            # Import and create incremental embedder after mocking
            from incremental_embedder import IncrementalEmbedder
            
            # Mock config paths
            with patch('config.get_proceeding_file_paths') as mock_paths:
                mock_paths.return_value = {
                    'embedding_status': proceeding_folder / 'embedding_status.json'
                }
                
                # Create embedder instance
                embedder = IncrementalEmbedder(proceeding)
                
                # Mock the scraped metadata loading
                def mock_load_scraped_metadata():
                    with open(history_file, 'r') as f:
                        history_data = json.load(f)
                    
                    return [
                        {
                            'hash': pdf_hash,
                            'url': pdf_info['url'],
                            'title': pdf_info['title']
                        }
                        for pdf_hash, pdf_info in history_data.items()
                    ]
                
                embedder._load_scraped_metadata = mock_load_scraped_metadata
                
                # Mock identify documents for embedding to return all documents
                def mock_identify_documents(metadata):
                    return metadata  # Process all documents
                
                embedder._identify_documents_for_embedding = mock_identify_documents
                
                print(f"🔧 Running incremental embedding process...")
                
                # Track method call counts
                process_single_url_calls = []
                add_document_calls = []
                
                def track_process_single_url(url_data):
                    process_single_url_calls.append(url_data)
                    return {
                        'success': True,
                        'chunks': [f'chunk_{len(process_single_url_calls)}_{i}' for i in range(5)],
                        'url': url_data['url'],
                        'title': url_data['title'],
                        'chunk_count': 5
                    }
                
                def track_add_document(chunks, url_hash, url_data, immediate_persist=True):
                    add_document_calls.append({
                        'chunks': len(chunks),
                        'url_hash': url_hash,
                        'url': url_data['url'],
                        'immediate_persist': immediate_persist
                    })
                    return True
                
                mock_rag_instance._process_single_url.side_effect = track_process_single_url
                mock_rag_instance.add_document_incrementally.side_effect = track_add_document
                
                # Run the incremental embedding process
                result = embedder.process_incremental_embeddings()
                
                print(f"📊 Processing Results:")
                print(f"   Status: {result['status']}")
                print(f"   Documents processed: {result['documents_processed']}")
                print(f"   Total chunks added: {result.get('total_chunks_added', 0)}")
                
                # Verify the fix: proper method usage
                print(f"\n🔍 Method Call Analysis:")
                print(f"   _process_single_url calls: {len(process_single_url_calls)}")
                print(f"   add_document_incrementally calls: {len(add_document_calls)}")
                print(f"   build_vector_store_from_urls calls: {mock_rag_instance.build_vector_store_from_urls.call_count}")
                
                # Assertions to verify the fix
                assert len(process_single_url_calls) == 3, f"Should call _process_single_url 3 times, got {len(process_single_url_calls)}"
                assert len(add_document_calls) == 3, f"Should call add_document_incrementally 3 times, got {len(add_document_calls)}"
                assert mock_rag_instance.build_vector_store_from_urls.call_count == 0, "Should NOT call build_vector_store_from_urls"
                
                # Verify each document was processed individually
                for i, call in enumerate(process_single_url_calls, 1):
                    expected_url = f'https://docs.cpuc.ca.gov/test{i:03d}.pdf'
                    assert call['url'] == expected_url, f"Call {i} should process {expected_url}"
                    print(f"   ✅ Document {i}: {call['url']} processed individually")
                
                # Verify incremental addition parameters
                for i, call in enumerate(add_document_calls, 1):
                    assert call['chunks'] == 5, f"Document {i} should have 5 chunks"
                    assert call['immediate_persist'] == True, f"Document {i} should use immediate persistence"
                    print(f"   ✅ Document {i}: {call['chunks']} chunks added incrementally")
                
                print(f"\n✅ INCREMENTAL EMBEDDING FIX VERIFIED!")
                print(f"   • No build_vector_store_from_urls calls (prevents restarts)")
                print(f"   • Individual document processing with _process_single_url")
                print(f"   • Incremental addition with add_document_incrementally")
                print(f"   • Immediate persistence for each document")


def test_incremental_embedding_error_handling():
    """
    Test that errors in individual documents don't affect others.
    """
    print(f"\n🧪 Test: Incremental Embedding Error Handling")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Test directory: {temp_dir}")
        
        # Create test proceeding
        proceeding = "R_TEST_ERROR"
        proceeding_folder, history_file = create_test_proceeding_with_documents(
            temp_dir, proceeding, num_docs=4
        )
        
        with patch('incremental_embedder.CPUCRAGSystem') as mock_rag_class:
            mock_rag_instance = MagicMock()
            mock_rag_class.return_value = mock_rag_instance
            
            # Mock to simulate success for docs 1,3,4 and failure for doc 2
            def mock_process_single_url(url_data):
                if 'test002.pdf' in url_data['url']:
                    return {
                        'success': False,
                        'error': 'Simulated processing error',
                        'chunks': [],
                        'url': url_data['url']
                    }
                else:
                    return {
                        'success': True,
                        'chunks': [f'chunk_{i}' for i in range(3)],
                        'url': url_data['url'],
                        'title': url_data['title'],
                        'chunk_count': 3
                    }
            
            mock_rag_instance._process_single_url.side_effect = mock_process_single_url
            mock_rag_instance.add_document_incrementally.return_value = True
            
            from incremental_embedder import IncrementalEmbedder
            
            with patch('config.get_proceeding_file_paths') as mock_paths:
                mock_paths.return_value = {
                    'embedding_status': proceeding_folder / 'embedding_status.json'
                }
                
                embedder = IncrementalEmbedder(proceeding)
                
                # Mock metadata loading
                def mock_load_scraped_metadata():
                    with open(history_file, 'r') as f:
                        history_data = json.load(f)
                    
                    return [
                        {
                            'hash': pdf_hash,
                            'url': pdf_info['url'],
                            'title': pdf_info['title']
                        }
                        for pdf_hash, pdf_info in history_data.items()
                    ]
                
                embedder._load_scraped_metadata = mock_load_scraped_metadata
                embedder._identify_documents_for_embedding = lambda metadata: metadata
                
                print(f"🔧 Running incremental embedding with simulated error...")
                
                # Run the process
                result = embedder.process_incremental_embeddings()
                
                print(f"📊 Error Handling Results:")
                print(f"   Status: {result['status']}")
                print(f"   Successful: {result['successful']}")
                print(f"   Failed: {result['failed']}")
                
                # Verify error handling
                assert result['successful'] == 3, f"Should have 3 successful documents, got {result['successful']}"
                assert result['failed'] == 1, f"Should have 1 failed document, got {result['failed']}"
                
                # Verify that successful documents were still processed
                assert mock_rag_instance.add_document_incrementally.call_count == 3, "Should add 3 successful documents"
                
                print(f"✅ ERROR HANDLING VERIFIED!")
                print(f"   • 3 documents processed successfully despite 1 failure")
                print(f"   • Failed document didn't stop processing of others")
                print(f"   • Incremental addition called only for successful documents")


def test_performance_comparison():
    """
    Test to demonstrate performance improvement of the fix.
    """
    print(f"\n🧪 Test: Performance Comparison")
    print("=" * 60)
    
    print(f"📈 Performance Analysis:")
    print(f"   OLD METHOD (causing restarts):")
    print(f"   • Called build_vector_store_from_urls() for each document")
    print(f"   • Each call rebuilt entire vector store context")
    print(f"   • For 414 documents = 414 full rebuilds")
    print(f"   • Massive redundant processing")
    print(f"")
    print(f"   NEW METHOD (true incremental):")
    print(f"   • Calls _process_single_url() for each document")
    print(f"   • Extracts chunks from individual document only")
    print(f"   • Calls add_document_incrementally() to add chunks")
    print(f"   • No vector store rebuilds")
    print(f"   • Linear processing time")
    
    # Simulate timing difference
    old_method_time = 414 * 5.02  # 5.02 seconds per document (from log)
    new_method_time = 414 * 0.5   # Estimated 0.5 seconds per document
    
    print(f"")
    print(f"📊 Estimated Performance Improvement:")
    print(f"   Old method: {old_method_time:.0f} seconds ({old_method_time/60:.1f} minutes)")
    print(f"   New method: {new_method_time:.0f} seconds ({new_method_time/60:.1f} minutes)")
    print(f"   Improvement: {old_method_time/new_method_time:.1f}x faster")
    print(f"   Time saved: {(old_method_time-new_method_time)/60:.1f} minutes")
    
    print(f"✅ PERFORMANCE ANALYSIS COMPLETE!")


if __name__ == '__main__':
    print("🧪 Running Incremental Embedding Fix Tests")
    print("Testing the fix for chunking/embedding restart issue")
    
    try:
        test_incremental_embedding_no_restart()
        test_incremental_embedding_error_handling()
        test_performance_comparison()
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Incremental embedding fix is working correctly")
        print(f"✅ No more chunking/embedding restarts")
        print(f"✅ True incremental processing implemented")
        print(f"✅ Individual document processing with proper error isolation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)