#!/usr/bin/env python3
"""
Fix Chunk-Limited Documents

This script identifies and reprocesses documents that were processed with the old 100-chunk limit.
These documents likely contain more content that was truncated due to the ArrowSchema recursion workaround.

Usage:
    python fix_chunk_limited_documents.py [proceeding] [--max-reprocess N] [--dry-run]

Examples:
    python fix_chunk_limited_documents.py R2110002
    python fix_chunk_limited_documents.py R2110002 --max-reprocess 5 --dry-run
    python fix_chunk_limited_documents.py R2110002 --max-reprocess 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data_processing.embedding_only_system import EmbeddingOnlySystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Fix documents that were processed with 100-chunk limit')
    parser.add_argument('proceeding', help='Proceeding number (e.g., R2110002)')
    parser.add_argument('--max-reprocess', type=int, default=None,
                       help='Maximum number of documents to reprocess (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Just identify documents, do not reprocess them')
    parser.add_argument('--chunk-limit', type=int, default=100,
                       help='Chunk limit to check for (default: 100)')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Automatically answer yes to confirmation prompt')
    
    args = parser.parse_args()
    
    print(f"🔍 Chunk-Limited Document Fix Tool")
    print(f"Proceeding: {args.proceeding}")
    print(f"Chunk limit to check: {args.chunk_limit}")
    print(f"Max reprocess: {args.max_reprocess or 'All'}")
    print(f"Mode: {'Dry run (identify only)' if args.dry_run else 'Full reprocessing'}")
    print("=" * 60)
    
    try:
        # Initialize embedding system
        logger.info(f"🔄 Initializing embedding system for {args.proceeding}...")
        system = EmbeddingOnlySystem(args.proceeding)
        
        # Check system health
        health = system.health_check()
        if not health.get('healthy', False):
            logger.error(f"❌ Embedding system is not healthy: {health}")
            return 1
        
        logger.info(f"✅ System healthy - Current vector count: {health.get('vector_count', 0)}")
        
        # Find documents with chunk limit
        logger.info(f"🔍 Searching for documents with {args.chunk_limit}-chunk limit...")
        limited_docs = system.find_chunk_limited_documents(chunk_limit=args.chunk_limit)
        
        if not limited_docs:
            print(f"🎉 No documents found with {args.chunk_limit}-chunk limit!")
            print("All documents appear to have been processed without artificial limits.")
            return 0
        
        print(f"\n📋 Found {len(limited_docs)} documents with {args.chunk_limit}-chunk limit:")
        print("-" * 80)
        
        for i, doc_info in enumerate(limited_docs[:10]):  # Show first 10
            print(f"{i+1:2d}. {doc_info['title']}")
            print(f"    URL: {doc_info['url']}")
            print(f"    Chunks: {doc_info['chunk_count']}")
            print(f"    Last processed: {doc_info['last_processed']}")
            print()
        
        if len(limited_docs) > 10:
            print(f"    ... and {len(limited_docs) - 10} more documents")
        
        if args.dry_run:
            print(f"\n🔍 DRY RUN MODE - No reprocessing performed")
            print(f"To reprocess these documents, run without --dry-run flag")
            return 0
        
        # Confirm reprocessing
        if args.max_reprocess:
            to_process = min(len(limited_docs), args.max_reprocess)
        else:
            to_process = len(limited_docs)
        
        print(f"\n⚠️  About to reprocess {to_process} documents")
        print("This will:")
        print("• Re-extract content from PDFs (may take time)")
        print("• Generate new embeddings for all chunks")
        print("• Update the vector store with complete data")
        print("• Replace old chunk-limited entries")
        
        if not args.yes:
            response = input(f"\nProceed with reprocessing {to_process} documents? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("❌ Reprocessing cancelled by user")
                return 0
        else:
            print(f"\n✅ Auto-confirming reprocessing of {to_process} documents (--yes flag provided)")
        
        # Start reprocessing
        start_time = time.time()
        logger.info(f"🚀 Starting reprocessing of {to_process} documents...")
        
        result = system.reprocess_chunk_limited_documents(
            chunk_limit=args.chunk_limit,
            max_reprocess=args.max_reprocess,
            enable_ocr_fallback=True
        )
        
        processing_time = time.time() - start_time
        
        print(f"\n🎯 REPROCESSING COMPLETE")
        print("=" * 40)
        print(f"✅ Successfully reprocessed: {result.get('reprocessed', 0)}")
        print(f"❌ Failed: {result.get('failed', 0)}")
        print(f"📊 Total new chunks: {result.get('total_new_chunks', 0)}")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        
        # Final health check
        final_health = system.health_check()
        print(f"📈 Final vector count: {final_health.get('vector_count', 0)}")
        
        if result.get('success'):
            print(f"\n🎉 SUCCESS: Documents with {args.chunk_limit}-chunk limit have been reprocessed!")
            print("Your vector store now contains complete document data without artificial limits.")
            return 0
        else:
            print(f"\n⚠️ PARTIAL SUCCESS: Some documents could not be reprocessed")
            print("Check the logs above for specific error details.")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Failed to fix chunk-limited documents: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())