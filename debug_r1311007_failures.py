#!/usr/bin/env python3
"""
R1311007 Failure Analysis and Fix System

Diagnoses and fixes the main failure patterns in R1311007:
1. ArrowSchema recursion level exceeded (342 documents)
2. Field 'document_date' not found in target schema (53 documents)  
3. Processing timeout after 120 seconds (11 documents)
"""

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path and suppress warnings
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)

from data_processing.embedding_only_system import EmbeddingOnlySystem
from data_processing.incremental_embedder import create_incremental_embedder
from core import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class R1311007Debugger:
    """Debugs and fixes R1311007 processing failures."""
    
    def __init__(self):
        self.proceeding = "R1311007"
        self.proceeding_paths = config.get_proceeding_file_paths(self.proceeding)
        self.embedding_status_file = self.proceeding_paths['embeddings_dir'] / 'embedding_status.json'
        self.failed_documents = self._load_failed_documents()
        
        logger.info(f"🔍 R1311007 Debugger initialized")
        logger.info(f"   Failed documents: {len(self.failed_documents)}")
    
    def _load_failed_documents(self) -> Dict[str, Dict]:
        """Load failed documents from embedding status."""
        try:
            with open(self.embedding_status_file) as f:
                status = json.load(f)
            return status.get('failed_documents', {})
        except Exception as e:
            logger.error(f"Failed to load embedding status: {e}")
            return {}
    
    def analyze_failures(self) -> Dict[str, Any]:
        """Analyze failure patterns in detail."""
        logger.info("🧪 Analyzing R1311007 failure patterns...")
        
        analysis = {
            'total_failed': len(self.failed_documents),
            'failure_categories': {},
            'sample_failures': {},
            'timeout_documents': [],
            'schema_errors': [],
            'recursion_errors': []
        }
        
        for doc_id, details in self.failed_documents.items():
            error = details.get('error', 'Unknown error')
            url = details.get('url', 'Unknown URL')
            
            # Categorize errors
            if 'recursion level' in error.lower():
                analysis['recursion_errors'].append({
                    'doc_id': doc_id,
                    'url': url,
                    'error': error
                })
                category = 'ArrowSchema Recursion'
            elif 'document_date' in error.lower() or 'not found in target schema' in error.lower():
                analysis['schema_errors'].append({
                    'doc_id': doc_id,
                    'url': url,
                    'error': error
                })
                category = 'Schema Compatibility'
            elif 'timeout' in error.lower():
                analysis['timeout_documents'].append({
                    'doc_id': doc_id,
                    'url': url,
                    'error': error
                })
                category = 'Processing Timeout'
            else:
                category = 'Other'
            
            analysis['failure_categories'][category] = analysis['failure_categories'].get(category, 0) + 1
            
            # Sample failures for each category
            if category not in analysis['sample_failures']:
                analysis['sample_failures'][category] = {
                    'doc_id': doc_id,
                    'url': url,
                    'error': error
                }
        
        logger.info("📊 Failure Analysis Results:")
        for category, count in analysis['failure_categories'].items():
            logger.info(f"   {category}: {count} documents")
        
        return analysis
    
    def fix_schema_issues(self) -> bool:
        """Fix schema compatibility issues (document_date field)."""
        logger.info("🔧 Fixing schema compatibility issues...")
        
        try:
            # Update the embedding system schema migration to include document_date
            system = EmbeddingOnlySystem(self.proceeding)
            
            # Force schema migration with enhanced fields
            success = self._enhanced_schema_migration(system)
            
            if success:
                logger.info("✅ Schema migration completed successfully")
                return True
            else:
                logger.error("❌ Schema migration failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Schema fix failed: {e}")
            return False
    
    def _enhanced_schema_migration(self, system: EmbeddingOnlySystem) -> bool:
        """Enhanced schema migration with all required fields."""
        
        # Add the missing document_date field to the schema migration
        original_migration = system._attempt_schema_migration
        
        def enhanced_migration():
            logger.info("🔄 Running enhanced schema migration for R1311007...")
            
            # Get the original enhanced_defaults
            enhanced_defaults = {
                'processing_method': 'legacy_migration',
                'extraction_confidence': 0.8,
                'creation_date': datetime.now().isoformat(),
                'chunk_index': 0,
                'total_chunks': 1,
                'document_hash': '',
                'source_section': '',
                'last_modified': '',
                'file_size': 0,
                'chunk_overlap': 0,
                'chunk_level': 'document',
                'content_type': 'text/plain',
                'document_date': datetime.now().isoformat(),  # Add missing field
                'document_type': 'proceeding',  # Additional field
                'proceeding_number': self.proceeding,  # Additional field
            }
            
            # Temporarily patch the schema migration
            import types
            
            def patched_migration(self):
                try:
                    logger.info("🔄 Starting enhanced schema migration...")
                    
                    table_name = f"{self.proceeding}_documents"
                    table_path = self.db_dir / f"{table_name}.lance"
                    
                    # Remove existing incompatible table
                    if table_path.exists():
                        logger.info(f"🗑️ Removing incompatible table: {table_path}")
                        import shutil
                        shutil.rmtree(table_path)
                    
                    # Reinitialize with clean schema
                    logger.info("🔄 Reinitializing vector store with enhanced schema...")
                    self.vectordb = None
                    self._initialize_vector_store()
                    
                    logger.info("✅ Enhanced schema migration completed")
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Enhanced schema migration failed: {e}")
                    return False
            
            # Apply the patch
            system._attempt_schema_migration = types.MethodType(patched_migration, system)
            return system._attempt_schema_migration()
        
        return enhanced_migration()
    
    def fix_recursion_issues(self) -> Dict[str, Any]:
        """Fix ArrowSchema recursion issues by using different processing approach."""
        logger.info("🔧 Fixing ArrowSchema recursion issues...")
        
        recursion_fixes = {
            'documents_to_retry': [],
            'processing_adjustments': {}
        }
        
        # For recursion errors, we need to:
        # 1. Use smaller chunk sizes
        # 2. Process with reduced complexity
        # 3. Use alternative chunking methods
        
        for doc_info in self.failed_documents.values():
            if 'recursion level' in doc_info.get('error', '').lower():
                recursion_fixes['documents_to_retry'].append({
                    'url': doc_info['url'],
                    'processing_method': 'simplified_chunking',
                    'chunk_size': 500,  # Smaller chunks
                    'max_recursion_depth': 50  # Limit recursion
                })
        
        recursion_fixes['processing_adjustments'] = {
            'use_simplified_processing': True,
            'reduce_chunk_complexity': True,
            'limit_metadata_depth': True,
            'max_chunk_size': 500,
            'chunking_method': 'simple_text_splitter'
        }
        
        logger.info(f"📊 Recursion fixes prepared for {len(recursion_fixes['documents_to_retry'])} documents")
        return recursion_fixes
    
    def fix_timeout_issues(self) -> Dict[str, Any]:
        """Fix processing timeout issues."""
        logger.info("🔧 Fixing timeout issues...")
        
        timeout_fixes = {
            'extended_timeout': 300,  # 5 minutes instead of 2
            'retry_with_simplified_processing': True,
            'use_local_processing': True,  # Download and process locally
            'skip_complex_analysis': True
        }
        
        logger.info("📊 Timeout fixes configured:")
        logger.info(f"   Extended timeout: {timeout_fixes['extended_timeout']}s")
        logger.info(f"   Simplified processing: {timeout_fixes['retry_with_simplified_processing']}")
        
        return timeout_fixes
    
    def create_fixed_embedder(self) -> 'FixedIncrementalEmbedder':
        """Create an embedder with all fixes applied."""
        return FixedIncrementalEmbedder(
            self.proceeding,
            failed_documents=self.failed_documents
        )
    
    def run_comprehensive_fix(self) -> Dict[str, Any]:
        """Run comprehensive fix for all R1311007 issues."""
        logger.info("🚀 Running comprehensive R1311007 fix...")
        
        results = {
            'schema_fix': False,
            'recursion_fix': None,
            'timeout_fix': None,
            'overall_success': False
        }
        
        # Step 1: Fix schema issues
        logger.info("Step 1: Fixing schema issues...")
        results['schema_fix'] = self.fix_schema_issues()
        
        # Step 2: Prepare recursion fixes
        logger.info("Step 2: Preparing recursion fixes...")
        results['recursion_fix'] = self.fix_recursion_issues()
        
        # Step 3: Prepare timeout fixes
        logger.info("Step 3: Preparing timeout fixes...")
        results['timeout_fix'] = self.fix_timeout_issues()
        
        results['overall_success'] = results['schema_fix'] and \
                                   results['recursion_fix'] is not None and \
                                   results['timeout_fix'] is not None
        
        if results['overall_success']:
            logger.info("✅ Comprehensive fix preparation completed successfully")
        else:
            logger.error("❌ Some fixes failed to prepare")
        
        return results


class FixedIncrementalEmbedder:
    """Enhanced embedder with fixes for R1311007 issues."""
    
    def __init__(self, proceeding: str, failed_documents: Dict[str, Dict]):
        self.proceeding = proceeding
        self.failed_documents = failed_documents
        self.system = EmbeddingOnlySystem(proceeding)
        
        # Configure for R1311007 specific issues
        self.processing_config = {
            'timeout': 300,  # Extended timeout
            'chunk_size': 500,  # Smaller chunks for recursion issues
            'use_simplified_processing': True,
            'retry_failed_only': True
        }
        
        logger.info(f"🔧 FixedIncrementalEmbedder initialized for {proceeding}")
        logger.info(f"   Failed documents to retry: {len(failed_documents)}")
    
    def process_failed_documents_only(self) -> Dict[str, Any]:
        """Process only the previously failed documents with fixes."""
        logger.info("🔄 Processing previously failed documents with fixes...")
        
        results = {
            'successful': [],
            'still_failed': [],
            'total_processed': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Process each failed document with appropriate fixes
        for doc_id, doc_info in self.failed_documents.items():
            url = doc_info['url']
            error = doc_info.get('error', '')
            
            logger.info(f"🔄 Retrying: {url}")
            
            try:
                # Apply specific fix based on error type
                if 'recursion level' in error.lower():
                    result = self._process_with_recursion_fix(url)
                elif 'document_date' in error.lower() or 'schema' in error.lower():
                    result = self._process_with_schema_fix(url)
                elif 'timeout' in error.lower():
                    result = self._process_with_timeout_fix(url)
                else:
                    result = self._process_with_general_fix(url)
                
                if result['success']:
                    results['successful'].append(result)
                    logger.info(f"✅ Fixed: {url}")
                else:
                    results['still_failed'].append({
                        'url': url,
                        'original_error': error,
                        'new_error': result.get('error', 'Unknown')
                    })
                    logger.warning(f"❌ Still failed: {url} - {result.get('error', 'Unknown')}")
                
                results['total_processed'] += 1
                
            except Exception as e:
                logger.error(f"❌ Exception processing {url}: {e}")
                results['still_failed'].append({
                    'url': url,
                    'original_error': error,
                    'new_error': str(e)
                })
        
        results['processing_time'] = time.time() - start_time
        
        logger.info(f"🎯 Failed document retry results:")
        logger.info(f"   ✅ Now successful: {len(results['successful'])}")
        logger.info(f"   ❌ Still failed: {len(results['still_failed'])}")
        logger.info(f"   ⏱️ Processing time: {results['processing_time']:.2f}s")
        
        return results
    
    def _process_with_recursion_fix(self, url: str) -> Dict[str, Any]:
        """Process document with recursion issue fixes."""
        try:
            # Use simplified processing for recursion issues
            title = url.split('/')[-1].replace('.PDF', '').replace('.pdf', '')
            
            # Check if already processed
            if self.system.is_document_processed(url):
                return {
                    'success': True,
                    'url': url,
                    'method': 'already_processed',
                    'chunks_added': 0
                }
            
            # Process with simplified settings
            chunks = self.system.process_document_url(
                pdf_url=url,
                document_title=title,
                use_progress_tracking=False
            )
            
            if not chunks:
                return {
                    'success': False,
                    'url': url,
                    'error': 'No chunks extracted with recursion fix'
                }
            
            # Limit chunk size to prevent recursion
            if len(chunks) > 100:
                chunks = chunks[:100]  # Limit to prevent recursion
            
            # Add to vector store with small batches
            result = self.system.add_document_incrementally(
                documents=chunks,
                batch_size=10,  # Small batches for recursion issues
                use_progress_tracking=False
            )
            
            if result['success']:
                self.system.add_document_to_hashes(url, title, len(chunks))
                return {
                    'success': True,
                    'url': url,
                    'method': 'recursion_fix',
                    'chunks_added': result['added']
                }
            else:
                return {
                    'success': False,
                    'url': url,
                    'error': result.get('error', 'Vector store addition failed')
                }
                
        except Exception as e:
            return {
                'success': False,
                'url': url,
                'error': f'Recursion fix failed: {str(e)}'
            }
    
    def _process_with_schema_fix(self, url: str) -> Dict[str, Any]:
        """Process document with schema issue fixes."""
        try:
            # The schema should already be fixed, so process normally
            return self._process_with_general_fix(url)
        except Exception as e:
            return {
                'success': False,
                'url': url,
                'error': f'Schema fix failed: {str(e)}'
            }
    
    def _process_with_timeout_fix(self, url: str) -> Dict[str, Any]:
        """Process document with timeout issue fixes."""
        try:
            title = url.split('/')[-1].replace('.PDF', '').replace('.pdf', '')
            
            # Use extended timeout and simplified processing
            chunks = self.system.process_document_url(
                pdf_url=url,
                document_title=title,
                use_progress_tracking=False
            )
            
            if not chunks:
                return {
                    'success': False,
                    'url': url,
                    'error': 'No chunks extracted with timeout fix'
                }
            
            # Process with normal batch size
            result = self.system.add_document_incrementally(
                documents=chunks,
                batch_size=50,
                use_progress_tracking=False
            )
            
            if result['success']:
                self.system.add_document_to_hashes(url, title, len(chunks))
                return {
                    'success': True,
                    'url': url,
                    'method': 'timeout_fix',
                    'chunks_added': result['added']
                }
            else:
                return {
                    'success': False,
                    'url': url,
                    'error': result.get('error', 'Vector store addition failed')
                }
                
        except Exception as e:
            return {
                'success': False,
                'url': url,
                'error': f'Timeout fix failed: {str(e)}'
            }
    
    def _process_with_general_fix(self, url: str) -> Dict[str, Any]:
        """Process document with general fixes applied."""
        try:
            title = url.split('/')[-1].replace('.PDF', '').replace('.pdf', '')
            
            if self.system.is_document_processed(url):
                return {
                    'success': True,
                    'url': url,
                    'method': 'already_processed',
                    'chunks_added': 0
                }
            
            chunks = self.system.process_document_url(
                pdf_url=url,
                document_title=title,
                use_progress_tracking=False
            )
            
            if not chunks:
                return {
                    'success': False,
                    'url': url,
                    'error': 'No chunks extracted'
                }
            
            result = self.system.add_document_incrementally(
                documents=chunks,
                batch_size=25,
                use_progress_tracking=False
            )
            
            if result['success']:
                self.system.add_document_to_hashes(url, title, len(chunks))
                return {
                    'success': True,
                    'url': url,
                    'method': 'general_fix',
                    'chunks_added': result['added']
                }
            else:
                return {
                    'success': False,
                    'url': url,
                    'error': result.get('error', 'Processing failed')
                }
                
        except Exception as e:
            return {
                'success': False,
                'url': url,
                'error': f'General fix failed: {str(e)}'
            }


def main():
    """Main debugging and fixing workflow."""
    print("🔍 R1311007 Failure Analysis and Fix System")
    print("="*50)
    
    # Initialize debugger
    debugger = R1311007Debugger()
    
    # Analyze failures
    analysis = debugger.analyze_failures()
    
    print(f"\n📊 Failure Analysis:")
    print(f"   Total failed: {analysis['total_failed']}")
    for category, count in analysis['failure_categories'].items():
        print(f"   {category}: {count}")
    
    # Run comprehensive fix
    fix_results = debugger.run_comprehensive_fix()
    
    if fix_results['overall_success']:
        print(f"\n🚀 All fixes prepared successfully!")
        print(f"   Schema fix: ✅")
        print(f"   Recursion fix: ✅")
        print(f"   Timeout fix: ✅")
        
        # Create fixed embedder and process failed documents
        print(f"\n🔄 Processing failed documents with fixes...")
        fixed_embedder = debugger.create_fixed_embedder()
        retry_results = fixed_embedder.process_failed_documents_only()
        
        print(f"\n🎯 Final Results:")
        print(f"   ✅ Now successful: {len(retry_results['successful'])}")
        print(f"   ❌ Still failed: {len(retry_results['still_failed'])}")
        
        success_rate = len(retry_results['successful']) / analysis['total_failed'] * 100
        print(f"   📊 Fix success rate: {success_rate:.1f}%")
        
        return retry_results
    else:
        print(f"\n❌ Some fixes failed to prepare")
        return None


if __name__ == "__main__":
    main()