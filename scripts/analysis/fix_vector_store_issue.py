#!/usr/bin/env python3
"""
Vector Store Issue Diagnostic and Fix Script

This script helps diagnose and fix the vector store population issue where:
1. Documents are processed successfully (781 docs)
2. Only 4 chunks make it to the LanceDB vector store
3. App shows "4 chunks available" instead of thousands

Usage:
    python fix_vector_store_issue.py R2207005 --diagnose
    python fix_vector_store_issue.py R2207005 --rebuild
    python fix_vector_store_issue.py R2207005 --test-batch
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config
from rag_core import CPUCRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStoreIssueFixer:
    """Diagnose and fix vector store population issues."""
    
    def __init__(self, proceeding: str):
        self.proceeding = proceeding
        self.project_root = Path(__file__).parent
        self.proceeding_dir = self.project_root / "cpuc_proceedings" / proceeding
        self.embedding_status_path = self.proceeding_dir / "embeddings" / "embedding_status.json"
        self.lance_db_path = self.project_root / "local_lance_db" / proceeding
        
    def diagnose_issue(self) -> Dict:
        """Comprehensive diagnosis of the vector store issue."""
        logger.info(f"🔍 Diagnosing vector store issue for {self.proceeding}")
        
        diagnosis = {
            'proceeding': self.proceeding,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'issues_found': [],
            'recommendations': []
        }
        
        # Check 1: Embedding status vs actual vector store
        logger.info("📊 Checking embedding status vs vector store content...")
        
        # Load embedding status
        if self.embedding_status_path.exists():
            with open(self.embedding_status_path, 'r') as f:
                embedding_status = json.load(f)
            
            # Support both old and new format
            embedded_docs = embedding_status.get('embedded_documents', {})
            processed_docs = embedding_status.get('processed_documents', {})
            
            if embedded_docs:
                # New format: all docs in embedded_documents are considered embedded
                processed_count = len(embedded_docs)
                embedded_count = len(embedded_docs)
            else:
                # Old format: check status field
                processed_count = len(processed_docs)
                embedded_count = sum(1 for doc in processed_docs.values() 
                                   if doc.get('status') == 'embedded')
            
            diagnosis['embedding_status'] = {
                'total_processed': processed_count,
                'marked_as_embedded': embedded_count,
                'file_exists': True
            }
            
            logger.info(f"  Embedding status: {embedded_count}/{processed_count} marked as embedded")
        else:
            diagnosis['embedding_status'] = {'file_exists': False}
            diagnosis['issues_found'].append("No embedding_status.json file found")
        
        # Check 2: Actual LanceDB content
        logger.info("🗄️ Checking actual LanceDB content...")
        
        try:
            import lancedb
            if self.lance_db_path.exists():
                db = lancedb.connect(str(self.lance_db_path))
                table_name = f"{self.proceeding}_documents"
                
                if table_name in db.table_names():
                    table = db.open_table(table_name)
                    actual_chunks = len(table.to_pandas())
                    
                    diagnosis['vector_store'] = {
                        'table_exists': True,
                        'actual_chunks': actual_chunks,
                        'table_schema': list(table.schema.names)
                    }
                    
                    logger.info(f"  LanceDB content: {actual_chunks} chunks in table")
                    
                    # Sample some data to check for issues
                    if actual_chunks > 0:
                        sample_data = table.to_pandas().head(3)
                        diagnosis['vector_store']['sample_metadata_keys'] = list(sample_data.columns)
                        
                        # Check for metadata issues
                        for col in sample_data.columns:
                            if col != 'vector':
                                null_count = sample_data[col].isnull().sum()
                                if null_count > 0:
                                    diagnosis['issues_found'].append(f"Column '{col}' has {null_count} null values")
                else:
                    diagnosis['vector_store'] = {'table_exists': False}
                    diagnosis['issues_found'].append(f"LanceDB table '{table_name}' does not exist")
            else:
                diagnosis['vector_store'] = {'db_exists': False}
                diagnosis['issues_found'].append("LanceDB directory does not exist")
                
        except Exception as e:
            diagnosis['vector_store'] = {'error': str(e)}
            diagnosis['issues_found'].append(f"Failed to check LanceDB: {e}")
        
        # Check 3: Test RAG system initialization
        logger.info("🔧 Testing RAG system initialization...")
        
        try:
            rag_system = CPUCRAGSystem(current_proceeding=self.proceeding)
            stats = rag_system.get_system_stats()
            
            diagnosis['rag_system'] = {
                'initialization_success': True,
                'stats': stats
            }
            
            logger.info(f"  RAG system stats: {stats}")
            
            # Test adding a small batch
            if hasattr(rag_system, 'doc_hashes'):
                diagnosis['rag_system']['doc_hashes_count'] = len(rag_system.doc_hashes)
                logger.info(f"  Document hashes tracked: {len(rag_system.doc_hashes)}")
            
        except Exception as e:
            diagnosis['rag_system'] = {
                'initialization_success': False,
                'error': str(e)
            }
            diagnosis['issues_found'].append(f"RAG system initialization failed: {e}")
        
        # Generate recommendations
        self._generate_recommendations(diagnosis)
        
        return diagnosis
    
    def _generate_recommendations(self, diagnosis: Dict):
        """Generate recommendations based on diagnosis."""
        recommendations = []
        
        # Check for major discrepancies
        embedding_count = diagnosis.get('embedding_status', {}).get('marked_as_embedded', 0)
        vector_count = diagnosis.get('vector_store', {}).get('actual_chunks', 0)
        
        if embedding_count > 0 and vector_count == 0:
            recommendations.append("CRITICAL: All documents marked as embedded but no chunks in vector store")
            recommendations.append("Action: Run --rebuild to reprocess documents")
        elif embedding_count > vector_count * 10:  # Significant discrepancy
            recommendations.append(f"MAJOR: Large discrepancy between embedded ({embedding_count}) and stored ({vector_count}) documents")
            recommendations.append("Action: Run --test-batch to identify specific failures")
            recommendations.append("Action: Consider running --rebuild if test-batch shows systematic issues")
        elif vector_count < 10:
            recommendations.append("WARNING: Very few chunks in vector store")
            recommendations.append("Action: Check if document processing completed successfully")
        
        # Check for specific issues
        if not diagnosis.get('vector_store', {}).get('table_exists', False):
            recommendations.append("Action: Initialize vector store by running data processor")
        
        if not diagnosis.get('rag_system', {}).get('initialization_success', False):
            recommendations.append("Action: Fix RAG system configuration issues")
        
        diagnosis['recommendations'] = recommendations
    
    def test_batch_processing(self, sample_size: int = 5) -> Dict:
        """Test processing a small batch of documents to identify issues."""
        logger.info(f"🧪 Testing batch processing with {sample_size} documents...")
        
        if not self.embedding_status_path.exists():
            return {'error': 'No embedding_status.json file found'}
        
        with open(self.embedding_status_path, 'r') as f:
            embedding_status = json.load(f)
        
        processed_docs = embedding_status.get('embedded_documents', embedding_status.get('processed_documents', {}))
        if not processed_docs:
            return {'error': 'No processed documents found in embedding status'}
        
        # Get a sample of documents
        sample_docs = list(processed_docs.items())[:sample_size]
        
        try:
            rag_system = CPUCRAGSystem(current_proceeding=self.proceeding)
            results = {'successful': [], 'failed': [], 'details': []}
            
            for doc_hash, doc_info in sample_docs:
                logger.info(f"Testing document: {doc_info.get('title', 'Unknown')}")
                
                try:
                    # Create test document data
                    url_data = {
                        'url': doc_info['url'],
                        'title': doc_info.get('title', 'Test Document')
                    }
                    
                    # Try to process the document
                    processing_result = rag_system._process_single_url(url_data)
                    
                    if processing_result['success']:
                        chunks = processing_result.get('chunks', [])
                        
                        # Try to add to vector store
                        success = rag_system.add_document_incrementally(
                            chunks=chunks,
                            url_hash=doc_hash,
                            url_data=url_data,
                            immediate_persist=True
                        )
                        
                        result_info = {
                            'doc_hash': doc_hash,
                            'title': doc_info.get('title', 'Unknown'),
                            'chunks_extracted': len(chunks),
                            'vector_store_success': success
                        }
                        
                        if success:
                            results['successful'].append(result_info)
                            logger.info(f"  ✅ Success: {len(chunks)} chunks")
                        else:
                            results['failed'].append(result_info)
                            logger.info(f"  ❌ Failed: Vector store addition failed")
                        
                        results['details'].append(result_info)
                    else:
                        result_info = {
                            'doc_hash': doc_hash,
                            'title': doc_info.get('title', 'Unknown'),
                            'processing_error': processing_result.get('error', 'Unknown error')
                        }
                        results['failed'].append(result_info)
                        results['details'].append(result_info)
                        logger.info(f"  ❌ Processing failed: {processing_result.get('error', 'Unknown')}")
                
                except Exception as e:
                    result_info = {
                        'doc_hash': doc_hash,
                        'title': doc_info.get('title', 'Unknown'),
                        'exception': str(e)
                    }
                    results['failed'].append(result_info)
                    results['details'].append(result_info)
                    logger.error(f"  💥 Exception: {e}")
            
            # Summary
            total_tested = len(sample_docs)
            success_count = len(results['successful'])
            failure_count = len(results['failed'])
            
            results['summary'] = {
                'total_tested': total_tested,
                'successful': success_count,
                'failed': failure_count,
                'success_rate': f"{success_count/total_tested:.1%}" if total_tested > 0 else "0%"
            }
            
            logger.info(f"🏁 Test complete: {success_count}/{total_tested} succeeded ({results['summary']['success_rate']})")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch testing failed: {e}")
            return {'error': str(e)}
    
    def rebuild_vector_store(self, batch_size: int = 10, max_docs: int = None) -> Dict:
        """Rebuild the vector store from embedding status."""
        logger.info(f"🔨 Rebuilding vector store for {self.proceeding}")
        
        if not self.embedding_status_path.exists():
            return {'error': 'No embedding_status.json file found'}
        
        with open(self.embedding_status_path, 'r') as f:
            embedding_status = json.load(f)
        
        processed_docs = embedding_status.get('embedded_documents', embedding_status.get('processed_documents', {}))
        if not processed_docs:
            return {'error': 'No processed documents found in embedding status'}
        
        # Clear existing vector store
        logger.info("🗑️ Clearing existing vector store...")
        try:
            if self.lance_db_path.exists():
                import shutil
                shutil.rmtree(self.lance_db_path)
                logger.info("  Existing vector store removed")
        except Exception as e:
            logger.warning(f"  Could not remove existing vector store: {e}")
        
        # Initialize fresh RAG system
        try:
            rag_system = CPUCRAGSystem(current_proceeding=self.proceeding)
            results = {'successful': 0, 'failed': 0, 'errors': []}
            
            # Process documents in batches
            doc_items = list(processed_docs.items())
            if max_docs:
                doc_items = doc_items[:max_docs]
                logger.info(f"  Processing first {max_docs} documents only")
            
            total_docs = len(doc_items)
            
            for i in range(0, total_docs, batch_size):
                batch = doc_items[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_docs + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                for doc_hash, doc_info in batch:
                    try:
                        url_data = {
                            'url': doc_info['url'],
                            'title': doc_info.get('title', 'Rebuilt Document')
                        }
                        
                        # Process and add to vector store
                        processing_result = rag_system._process_single_url(url_data)
                        
                        if processing_result['success']:
                            chunks = processing_result.get('chunks', [])
                            success = rag_system.add_document_incrementally(
                                chunks=chunks,
                                url_hash=doc_hash,
                                url_data=url_data,
                                immediate_persist=True
                            )
                            
                            if success:
                                results['successful'] += 1
                                logger.debug(f"  ✅ {doc_info.get('title', 'Unknown')}")
                            else:
                                results['failed'] += 1
                                results['errors'].append(f"Vector store failed: {doc_info.get('title', 'Unknown')}")
                        else:
                            results['failed'] += 1
                            results['errors'].append(f"Processing failed: {doc_info.get('title', 'Unknown')} - {processing_result.get('error', 'Unknown')}")
                    
                    except Exception as e:
                        results['failed'] += 1
                        results['errors'].append(f"Exception: {doc_info.get('title', 'Unknown')} - {str(e)}")
                        logger.error(f"  💥 {doc_info.get('title', 'Unknown')}: {e}")
            
            # Final verification
            stats = rag_system.get_system_stats()
            results['final_stats'] = stats
            results['success_rate'] = f"{results['successful']/(results['successful'] + results['failed']):.1%}" if (results['successful'] + results['failed']) > 0 else "0%"
            
            logger.info(f"🏁 Rebuild complete: {results['successful']} successful, {results['failed']} failed ({results['success_rate']})")
            logger.info(f"📊 Final vector store: {stats.get('total_chunks', 0)} chunks")
            
            return results
            
        except Exception as e:
            logger.error(f"Rebuild failed: {e}")
            return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Diagnose and fix vector store issues")
    parser.add_argument("proceeding", help="Proceeding number (e.g., R2207005)")
    parser.add_argument("--diagnose", action="store_true", help="Run comprehensive diagnosis")
    parser.add_argument("--test-batch", action="store_true", help="Test processing a small batch")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector store")
    parser.add_argument("--sample-size", type=int, default=5, help="Sample size for test batch")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for rebuild")
    parser.add_argument("--max-docs", type=int, help="Maximum documents to process in rebuild")
    
    args = parser.parse_args()
    
    if not any([args.diagnose, args.test_batch, args.rebuild]):
        parser.error("Must specify one of: --diagnose, --test-batch, --rebuild")
    
    fixer = VectorStoreIssueFixer(args.proceeding)
    
    if args.diagnose:
        logger.info("=" * 60)
        logger.info("VECTOR STORE ISSUE DIAGNOSIS")
        logger.info("=" * 60)
        
        diagnosis = fixer.diagnose_issue()
        
        print("\n" + "=" * 60)
        print("DIAGNOSIS RESULTS")
        print("=" * 60)
        print(f"Proceeding: {diagnosis['proceeding']}")
        print(f"Timestamp: {diagnosis['timestamp']}")
        
        if diagnosis.get('embedding_status', {}).get('file_exists'):
            embedding_data = diagnosis['embedding_status']
            print(f"\nEmbedding Status:")
            print(f"  Total processed: {embedding_data.get('total_processed', 0)}")
            print(f"  Marked as embedded: {embedding_data.get('marked_as_embedded', 0)}")
        
        if diagnosis.get('vector_store', {}).get('table_exists'):
            vector_data = diagnosis['vector_store']
            print(f"\nVector Store:")
            print(f"  Actual chunks: {vector_data.get('actual_chunks', 0)}")
            print(f"  Table schema: {vector_data.get('table_schema', [])}")
        
        if diagnosis.get('issues_found'):
            print(f"\nIssues Found:")
            for issue in diagnosis['issues_found']:
                print(f"  ❌ {issue}")
        
        if diagnosis.get('recommendations'):
            print(f"\nRecommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  💡 {rec}")
        
        print("=" * 60)
    
    elif args.test_batch:
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING TEST")
        logger.info("=" * 60)
        
        results = fixer.test_batch_processing(args.sample_size)
        
        if 'error' in results:
            print(f"❌ Test failed: {results['error']}")
        else:
            summary = results['summary']
            print(f"\n🏁 Test Results: {summary['successful']}/{summary['total_tested']} succeeded ({summary['success_rate']})")
            
            if results['failed']:
                print(f"\nFailed Documents:")
                for failure in results['failed']:
                    print(f"  ❌ {failure.get('title', 'Unknown')}: {failure.get('processing_error', failure.get('exception', 'Unknown error'))}")
    
    elif args.rebuild:
        logger.info("=" * 60)
        logger.info("VECTOR STORE REBUILD")
        logger.info("=" * 60)
        
        results = fixer.rebuild_vector_store(args.batch_size, args.max_docs)
        
        if 'error' in results:
            print(f"❌ Rebuild failed: {results['error']}")
        else:
            print(f"\n🏁 Rebuild Results:")
            print(f"  Successful: {results['successful']}")
            print(f"  Failed: {results['failed']}")
            print(f"  Success rate: {results['success_rate']}")
            
            if results.get('final_stats'):
                stats = results['final_stats']
                print(f"  Final chunks: {stats.get('total_chunks', 0)}")
            
            if results.get('errors'):
                print(f"\nErrors encountered:")
                for error in results['errors'][:10]:  # Show first 10 errors
                    print(f"  ❌ {error}")
                if len(results['errors']) > 10:
                    print(f"  ... and {len(results['errors']) - 10} more errors")


if __name__ == "__main__":
    main()