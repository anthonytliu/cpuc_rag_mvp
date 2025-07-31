#!/usr/bin/env python3
"""
R1311007 Batch Processor
Processes a small batch of failed R1311007 documents to test fixes.
"""

import json
import logging
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

# Add src to path and suppress warnings
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)

from data_processing.incremental_embedder import create_incremental_embedder
from data_processing.embedding_only_system import EmbeddingOnlySystem
from core import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class R1311007BatchProcessor:
    """Process a small batch of R1311007 failed documents."""
    
    def __init__(self, batch_size=5):
        self.proceeding = "R1311007"
        self.batch_size = batch_size
        self.proceeding_paths = config.get_proceeding_file_paths(self.proceeding)
        self.embedding_status_file = self.proceeding_paths['embeddings_dir'] / 'embedding_status.json'
        self.failed_documents = self._load_failed_documents()
        
        logger.info(f"🔧 R1311007 Batch Processor initialized")
        logger.info(f"   Total failed documents: {len(self.failed_documents)}")
        logger.info(f"   Batch size: {batch_size}")
    
    def _load_failed_documents(self):
        """Load failed documents from embedding status."""
        try:
            with open(self.embedding_status_file) as f:
                status = json.load(f)
            return status.get('failed_documents', {})
        except Exception as e:
            logger.error(f"Failed to load embedding status: {e}")
            return {}
    
    def get_sample_batch(self, category=None):
        """Get a sample batch of failed documents."""
        if category:
            # Filter by error category
            filtered_docs = {}
            for doc_id, details in self.failed_documents.items():
                error = details.get('error', '')
                
                if category == 'schema' and ('document_date' in error.lower() or 'not found in target schema' in error.lower()):
                    filtered_docs[doc_id] = details
                elif category == 'recursion' and 'recursion level' in error.lower():
                    filtered_docs[doc_id] = details
                elif category == 'timeout' and 'timeout' in error.lower():
                    filtered_docs[doc_id] = details
                    
                if len(filtered_docs) >= self.batch_size:
                    break
            
            return filtered_docs
        else:
            # Get first N documents
            return dict(list(self.failed_documents.items())[:self.batch_size])
    
    def process_batch(self, batch_docs, category_name="Mixed"):
        """Process a batch of failed documents."""
        logger.info(f"🔄 Processing {category_name} batch: {len(batch_docs)} documents")
        
        start_time = time.time()
        results = {
            'successful': [],
            'still_failed': [],
            'processing_time': 0
        }
        
        # Create incremental embedder with timeout enabled
        embedder = create_incremental_embedder(self.proceeding, enable_timeout=True)
        
        for doc_id, doc_details in batch_docs.items():
            url = doc_details['url']
            original_error = doc_details.get('error', 'Unknown error')
            
            logger.info(f"🔄 Processing: {url}")
            logger.info(f"   Original error: {original_error[:100]}...")
            
            try:
                # Extract document title from URL
                title = url.split('/')[-1].replace('.PDF', '').replace('.pdf', '')
                
                # Check if already processed (might have been fixed)
                system = EmbeddingOnlySystem(self.proceeding)
                if system.is_document_processed(url):
                    logger.info(f"✅ Already processed: {url}")
                    results['successful'].append({
                        'url': url,
                        'method': 'already_processed',
                        'chunks_added': 0
                    })
                    continue
                
                # Process the document with the standard embedder
                processing_result = embedder.process_single_document(url)
                
                if processing_result and processing_result.get('success', False):
                    results['successful'].append({
                        'url': url,
                        'method': 'batch_processing',
                        'chunks_added': processing_result.get('chunks_added', 0),
                        'original_error': original_error
                    })
                    logger.info(f"✅ Successfully processed: {url}")
                else:
                    error_msg = processing_result.get('error', 'Unknown processing error') if processing_result else 'No result returned'
                    results['still_failed'].append({
                        'url': url,
                        'original_error': original_error,
                        'new_error': error_msg
                    })
                    logger.warning(f"❌ Still failed: {url} - {error_msg}")
                
            except Exception as e:
                logger.error(f"❌ Exception processing {url}: {e}")
                results['still_failed'].append({
                    'url': url,
                    'original_error': original_error,
                    'new_error': str(e)
                })
        
        results['processing_time'] = time.time() - start_time
        
        logger.info(f"📊 {category_name} batch results:")
        logger.info(f"   ✅ Successful: {len(results['successful'])}")
        logger.info(f"   ❌ Still failed: {len(results['still_failed'])}")
        logger.info(f"   ⏱️ Processing time: {results['processing_time']:.2f}s")
        
        return results
    
    def run_category_tests(self):
        """Run tests on different failure categories."""
        logger.info("🧪 Running category-specific tests...")
        
        all_results = {}
        
        # Test each category
        categories = [
            ('schema', 'Schema Compatibility'),
            ('timeout', 'Processing Timeout'),
            ('recursion', 'ArrowSchema Recursion')
        ]
        
        for category_key, category_name in categories:
            logger.info(f"\n🔄 Testing {category_name} fixes...")
            
            batch = self.get_sample_batch(category_key)
            if not batch:
                logger.warning(f"⚠️ No documents found for {category_name}")
                continue
            
            results = self.process_batch(batch, category_name)
            all_results[category_name] = results
        
        return all_results
    
    def generate_summary(self, all_results):
        """Generate summary of batch processing results."""
        logger.info("\n📊 BATCH PROCESSING SUMMARY")
        logger.info("=" * 50)
        
        total_successful = 0
        total_failed = 0
        total_time = 0
        
        for category, results in all_results.items():
            successful = len(results['successful'])
            failed = len(results['still_failed'])
            processing_time = results['processing_time']
            
            total_successful += successful
            total_failed += failed
            total_time += processing_time
            
            success_rate = (successful / (successful + failed) * 100) if (successful + failed) > 0 else 0
            
            logger.info(f"\n{category}:")
            logger.info(f"   ✅ Successful: {successful}")
            logger.info(f"   ❌ Still failed: {failed}")
            logger.info(f"   📊 Success rate: {success_rate:.1f}%")
            logger.info(f"   ⏱️ Time: {processing_time:.2f}s")
        
        overall_success_rate = (total_successful / (total_successful + total_failed) * 100) if (total_successful + total_failed) > 0 else 0
        
        logger.info(f"\nOVERALL RESULTS:")
        logger.info(f"   ✅ Total successful: {total_successful}")
        logger.info(f"   ❌ Total failed: {total_failed}")
        logger.info(f"   📊 Overall success rate: {overall_success_rate:.1f}%")
        logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
        
        return {
            'total_successful': total_successful,
            'total_failed': total_failed,
            'success_rate': overall_success_rate,
            'total_time': total_time
        }


def main():
    """Run R1311007 batch processing test."""
    print("🔧 R1311007 Batch Processor")
    print("=" * 40)
    
    # Process small batches from each category
    processor = R1311007BatchProcessor(batch_size=3) # Small batches for testing
    
    # Run category-specific tests
    all_results = processor.run_category_tests()
    
    # Generate summary
    summary = processor.generate_summary(all_results)
    
    if summary['success_rate'] > 50:
        print(f"\n🎉 Batch test SUCCESS! {summary['success_rate']:.1f}% success rate")
        print("The fixes are working. Ready for full processing.")
        return True
    else:
        print(f"\n⚠️ Batch test showed issues. {summary['success_rate']:.1f}% success rate") 
        print("More debugging needed before full processing.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)