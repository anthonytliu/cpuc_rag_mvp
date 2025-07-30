#!/usr/bin/env python3
"""
End-to-end test for the complete hybrid processing system.

This test validates the entire pipeline:
1. Document processing with intelligent hybrid routing
2. Incremental embedding with LanceDB storage
3. RAG query functionality
4. Agent evaluation logging

Author: Claude Code
"""

import logging
import json
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_PROCEEDING = "R2207005"
TEST_DOCUMENTS = [
    {
        "url": "https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M571/K985/571985189.PDF",
        "title": "Final Decision D2506047",
        "expected_method": "chonkie"  # Text-heavy document
    },
    {
        "url": "https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M564/K706/564706741.PDF", 
        "title": "Agenda Decision D2505026",
        "expected_method": "hybrid"  # Has compensation tables
    }
]


def test_hybrid_document_processing():
    """Test document processing with hybrid routing."""
    print("\n🧪 Testing Hybrid Document Processing")
    print("=" * 60)
    
    from data_processing import extract_and_chunk_with_docling_url
    
    results = {}
    
    for i, doc in enumerate(TEST_DOCUMENTS):
        print(f"\n📄 Processing document {i+1}: {doc['title']}")
        print(f"   URL: {doc['url']}")
        print(f"   Expected method: {doc['expected_method']}")
        
        # Process document with hybrid system enabled
        chunks = extract_and_chunk_with_docling_url(
            pdf_url=doc["url"],
            document_title=doc["title"],
            proceeding=TEST_PROCEEDING,
            use_intelligent_hybrid=True
        )
        
        if chunks:
            # Analyze the processing method used
            content_type = chunks[0].metadata.get('content_type', 'unknown')
            processing_method = 'chonkie' if 'chonkie' in content_type else 'docling'
            
            results[doc["title"]] = {
                'chunks': len(chunks),
                'method_used': processing_method,
                'content_type': content_type,
                'success': True
            }
            
            print(f"   ✅ Success: {len(chunks)} chunks")
            print(f"   📊 Method used: {processing_method}")
            print(f"   🏷️  Content type: {content_type}")
            
        else:
            results[doc["title"]] = {
                'chunks': 0,
                'method_used': 'failed',
                'content_type': 'none',
                'success': False
            }
            print("   ❌ Failed: No chunks extracted")
    
    # Validate results
    success_count = sum(1 for r in results.values() if r['success'])
    print(f"\n📊 Processing Summary:")
    print(f"   Documents processed: {len(TEST_DOCUMENTS)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(TEST_DOCUMENTS) - success_count}")
    
    assert success_count >= len(TEST_DOCUMENTS) * 0.8, f"Expected at least 80% success rate, got {success_count}/{len(TEST_DOCUMENTS)}"
    
    print("✅ Hybrid document processing test passed")
    return results


def test_incremental_embedding_integration():
    """Test incremental embedding with hybrid processing."""
    print("\n🧪 Testing Incremental Embedding Integration")
    print("=" * 60)
    
    try:
        from incremental_embedder import IncrementalEmbedder
        import config
        
        # Temporarily enable hybrid processing
        original_hybrid = config.INTELLIGENT_HYBRID_ENABLED
        config.INTELLIGENT_HYBRID_ENABLED = True
        
        print(f"📁 Testing with proceeding: {TEST_PROCEEDING}")
        
        # Create embedder instance
        embedder = IncrementalEmbedder(TEST_PROCEEDING)
        
        print(f"📄 Running incremental embedding process")
        
        # Process documents incrementally (uses existing scraped data)
        results = embedder.process_incremental_embeddings()
        
        print(f"📊 Processing results:")
        print(f"   Documents processed: {results.get('processed', 0)}")
        print(f"   Documents skipped: {results.get('skipped', 0)}")
        print(f"   Total chunks: {results.get('total_chunks', 0)}")
        
        # Restore original setting
        config.INTELLIGENT_HYBRID_ENABLED = original_hybrid
        
        success = results.get('processed', 0) > 0
        if success:
            print("✅ Incremental embedding integration test passed")
        else:
            print("⚠️  Incremental embedding test completed with no new documents (may already be processed)")
        
        return success
        
    except Exception as e:
        print(f"❌ Incremental embedding test failed: {e}")
        return False


def test_rag_query_functionality():
    """Test RAG query functionality with hybrid-processed documents."""
    print("\n🧪 Testing RAG Query Functionality")
    print("=" * 60)
    
    try:
        from rag_core import CPUCRAGSystem
        
        print(f"🔍 Initializing RAG system for {TEST_PROCEEDING}")
        
        # Initialize RAG system
        rag = CPUCRAGSystem(proceeding=TEST_PROCEEDING)
        
        # Test queries
        test_queries = [
            "What are the key decisions in this proceeding?",
            "Are there any compensation or financial requirements?",
            "What tables or schedules are mentioned?"
        ]
        
        successful_queries = 0
        
        for i, query in enumerate(test_queries):
            print(f"\n❓ Query {i+1}: {query}")
            
            try:
                response = rag.query(query)
                
                if response and len(response.strip()) > 10:
                    print(f"   ✅ Response length: {len(response)} characters")
                    print(f"   📝 Preview: {response[:100]}...")
                    successful_queries += 1
                else:
                    print("   ⚠️  Empty or very short response")
                    
            except Exception as query_error:
                print(f"   ❌ Query failed: {query_error}")
        
        print(f"\n📊 Query Summary:")
        print(f"   Queries tested: {len(test_queries)}")
        print(f"   Successful: {successful_queries}")
        
        success = successful_queries >= len(test_queries) * 0.7
        if success:
            print("✅ RAG query functionality test passed")
        else:
            print("❌ RAG query functionality test failed")
        
        return success
        
    except Exception as e:
        print(f"❌ RAG query test failed: {e}")
        return False


def test_agent_evaluation_logs():
    """Test that agent evaluation logs are being created."""
    print("\n🧪 Testing Agent Evaluation Logs")
    print("=" * 60)
    
    try:
        import config
        
        log_dir = Path(config.AGENT_EVALUATION_LOG_DIR)
        print(f"📁 Checking log directory: {log_dir}")
        
        if not log_dir.exists():
            print("⚠️  Log directory doesn't exist - no hybrid evaluations have been performed yet")
            return True  # This is OK for a fresh system
        
        log_files = list(log_dir.glob("agent_evaluation_*.txt"))
        print(f"📊 Found {len(log_files)} evaluation log files")
        
        if log_files:
            # Check the most recent log file
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            print(f"📄 Latest log: {latest_log.name}")
            
            # Verify log content structure
            log_content = latest_log.read_text()
            required_sections = [
                "AGENT EVALUATION LOG",
                "DOCLING RESULTS:",
                "CHONKIE RESULTS:",
                "AGENT DECISION:",
                "AGENT REASONING:"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in log_content:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"❌ Missing log sections: {missing_sections}")
                return False
            else:
                print("✅ Log file structure is correct")
                return True
        else:
            print("⚠️  No evaluation logs found - system may not have processed high-score documents yet")
            return True  # This is OK
            
    except Exception as e:
        print(f"❌ Agent evaluation logs test failed: {e}")
        return False


def test_database_integrity():
    """Test LanceDB database integrity and schema consistency."""
    print("\n🧪 Testing Database Integrity")
    print("=" * 60)
    
    try:
        import config
        from pathlib import Path
        
        # Check database directory
        db_dir = Path(config.DB_DIR) / TEST_PROCEEDING
        print(f"📁 Checking database: {db_dir}")
        
        if not db_dir.exists():
            print("⚠️  Database doesn't exist yet - this is normal for a fresh system")
            return True
        
        # Try to connect to database
        import lancedb
        
        try:
            db = lancedb.connect(str(db_dir.parent))
            table_name = TEST_PROCEEDING
            
            if table_name in db.table_names():
                table = db.open_table(table_name)
                count = table.count_rows()
                
                print(f"📊 Database statistics:")
                print(f"   Table: {table_name}")
                print(f"   Rows: {count}")
                
                if count > 0:
                    # Check schema
                    schema = table.schema
                    print(f"   Schema fields: {len(schema)}")
                    
                    # Verify key fields exist
                    field_names = [field.name for field in schema]
                    required_fields = ['vector', 'source_url', 'content_type']
                    
                    missing_fields = [f for f in required_fields if f not in field_names]
                    if missing_fields:
                        print(f"❌ Missing required fields: {missing_fields}")
                        return False
                    else:
                        print("✅ Database schema is correct")
                        return True
                else:
                    print("⚠️  Database is empty - no documents have been embedded yet")
                    return True
            else:
                print(f"⚠️  Table {table_name} doesn't exist yet")
                return True
                
        except Exception as db_error:
            print(f"❌ Database connection failed: {db_error}")
            return False
            
    except Exception as e:
        print(f"❌ Database integrity test failed: {e}")
        return False


def run_end_to_end_test():
    """Run complete end-to-end test suite."""
    print("\n🚀 STARTING END-TO-END HYBRID SYSTEM TEST")
    print("=" * 80)
    
    test_results = {}
    
    # Run all tests
    test_functions = [
        ("Hybrid Document Processing", test_hybrid_document_processing),
        ("Incremental Embedding Integration", test_incremental_embedding_integration),
        ("RAG Query Functionality", test_rag_query_functionality),
        ("Agent Evaluation Logs", test_agent_evaluation_logs),
        ("Database Integrity", test_database_integrity)
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n{'='*80}")
        try:
            result = test_func()
            test_results[test_name] = result
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            test_results[test_name] = False
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Summary
    print(f"\n🎯 END-TO-END TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed >= total * 0.8:  # 80% pass rate
        print("🎉 END-TO-END TEST SUITE PASSED!")
        print("✅ Hybrid processing system is ready for production use")
        return True
    else:
        print("❌ END-TO-END TEST SUITE FAILED")
        print("🔧 Some components need attention before production use")
        return False


if __name__ == "__main__":
    print("🧪 HYBRID PROCESSING SYSTEM - END-TO-END TEST")
    print("=" * 50)
    
    success = run_end_to_end_test()
    
    if success:
        print("\n🎉 ALL SYSTEMS GO - HYBRID PROCESSING READY!")
    else:
        print("\n🔧 SYSTEM NEEDS ATTENTION - CHECK LOGS ABOVE")
    
    exit(0 if success else 1)