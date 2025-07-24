#!/usr/bin/env python3
"""
Standalone Data Processing System for CPUC Documents

This script processes scraped PDF data from the cpuc_proceedings structure,
creating embeddings and chunks for use in the RAG system. It operates
independently from the main application and maintains all scraped data
as read-only.

Features:
- Processes existing scraped PDF metadata from cpuc_proceedings structure
- Creates embeddings folder within each proceeding directory
- Populates centralized vector database in root directory
- Provides progress tracking and error recovery
- Maintains backward compatibility with existing data

Usage:
    python standalone_data_processor.py [proceeding_number]
    python standalone_data_processor.py --all
    python standalone_data_processor.py --list-proceedings
    
Examples:
    python standalone_data_processor.py R2207005
    python standalone_data_processor.py --all
    python standalone_data_processor.py --list-proceedings

Author: Claude Code
"""

import sys
import argparse
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

# Import config for DEBUG setting
import config

# Setup logging for standalone operation - level depends on DEBUG flag
log_level = logging.DEBUG if config.DEBUG else logging.INFO
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if config.DEBUG else '%(message)s'

logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('standalone_data_processor.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone CPUC Data Processor - Convert scraped PDFs to embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        # Process all proceedings from config
  %(prog)s R2207005               # Process specific proceeding
  %(prog)s --all                  # Process all proceedings found in cpuc_proceedings/
  %(prog)s --list-proceedings     # List available proceedings to process
  %(prog)s --status R2207005      # Show processing status for proceeding
        """
    )
    
    parser.add_argument(
        'proceeding',
        nargs='?',
        help='CPUC proceeding number (e.g., R2207005). If not specified, processes all from config.'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all proceedings found in cpuc_proceedings directory'
    )
    
    parser.add_argument(
        '--list-proceedings',
        action='store_true',
        help='List available proceedings and their processing status'
    )
    
    parser.add_argument(
        '--status',
        metavar='PROCEEDING',
        help='Show detailed processing status for a specific proceeding'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild of all embeddings, even if they already exist'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of documents to process in each batch (default: 10)'
    )
    
    return parser.parse_args()

def get_config_proceedings() -> List[str]:
    """Get list of proceedings from config."""
    try:
        import config
        proceedings = getattr(config, 'SCRAPER_PROCEEDINGS', ['R2207005'])
        logger.info(f"Loaded {len(proceedings)} proceedings from config: {proceedings}")
        return proceedings
    except ImportError:
        logger.warning("Could not import config, using fallback default")
        return ['R2207005']

def discover_available_proceedings() -> List[str]:
    """Discover all available proceedings in cpuc_proceedings directory."""
    cpuc_proceedings_dir = Path('cpuc_proceedings')
    if not cpuc_proceedings_dir.exists():
        logger.warning("cpuc_proceedings directory not found")
        return []
    
    proceedings = []
    for proceeding_dir in cpuc_proceedings_dir.iterdir():
        if proceeding_dir.is_dir() and proceeding_dir.name.startswith('R'):
            # Check if it has scraped PDF history
            history_file = proceeding_dir / f"{proceeding_dir.name}_scraped_pdf_history.json"
            if history_file.exists():
                proceedings.append(proceeding_dir.name)
    
    proceedings.sort()
    logger.info(f"Discovered {len(proceedings)} proceedings with scraped data: {proceedings}")
    return proceedings

def get_proceeding_status(proceeding: str) -> Dict:
    """Get processing status for a proceeding."""
    try:
        import config
        paths = config.get_proceeding_file_paths(proceeding)
        
        status = {
            'proceeding': proceeding,
            'scraped_data_exists': False,
            'embeddings_dir_exists': False,
            'vector_db_exists': False,
            'total_scraped_pdfs': 0,
            'total_embedded_docs': 0,
            'processing_status': 'unknown',
            'last_processed': None,
            'errors': []
        }
        
        # Check scraped data
        if paths['scraped_pdf_history'].exists():
            status['scraped_data_exists'] = True
            try:
                with open(paths['scraped_pdf_history'], 'r') as f:
                    scraped_data = json.load(f)
                status['total_scraped_pdfs'] = len(scraped_data)
            except Exception as e:
                status['errors'].append(f"Could not read scraped data: {e}")
        
        # Check embeddings directory
        status['embeddings_dir_exists'] = paths['embeddings_dir'].exists()
        
        # Check vector database
        status['vector_db_exists'] = paths['vector_db'].exists()
        
        # Check processing status
        if paths['embedding_status'].exists():
            try:
                with open(paths['embedding_status'], 'r') as f:
                    embedding_status = json.load(f)
                status['total_embedded_docs'] = embedding_status.get('total_embedded', 0)
                status['last_processed'] = embedding_status.get('last_updated')
                
                if status['total_embedded_docs'] > 0:
                    if status['total_embedded_docs'] >= status['total_scraped_pdfs']:
                        status['processing_status'] = 'completed'
                    else:
                        status['processing_status'] = 'partial'
                else:
                    status['processing_status'] = 'not_started'
            except Exception as e:
                status['errors'].append(f"Could not read embedding status: {e}")
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get status for {proceeding}: {e}")
        return {
            'proceeding': proceeding,
            'processing_status': 'error',
            'errors': [str(e)]
        }

def list_proceedings_status() -> None:
    """List all proceedings and their processing status."""
    logger.info("Scanning for proceedings...")
    
    # Get proceedings from both config and directory discovery
    config_proceedings = get_config_proceedings()
    discovered_proceedings = discover_available_proceedings()
    
    all_proceedings = list(set(config_proceedings + discovered_proceedings))
    all_proceedings.sort()
    
    print(f"\n📊 CPUC Data Processing Status")
    print("=" * 60)
    
    if config_proceedings:
        print(f"\n⚙️  Configured Proceedings:")
        for proceeding in config_proceedings:
            print(f"  • {proceeding}")
    
    print(f"\n📁 Available Proceedings ({len(all_proceedings)} total):")
    
    for proceeding in all_proceedings:
        status = get_proceeding_status(proceeding)
        
        # Status indicators
        scraped_icon = "✅" if status['scraped_data_exists'] else "❌"
        embeddings_icon = "✅" if status['embeddings_dir_exists'] else "❌"
        vector_icon = "✅" if status['vector_db_exists'] else "❌"
        
        processing_status = status['processing_status']
        if processing_status == 'completed':
            status_icon = "🟢"
        elif processing_status == 'partial':
            status_icon = "🟡"
        elif processing_status == 'not_started':
            status_icon = "🔴"
        else:
            status_icon = "⚪"
        
        print(f"\n  {proceeding}:")
        print(f"    Scraped Data: {scraped_icon}  Embeddings: {embeddings_icon}  Vector DB: {vector_icon}  Status: {status_icon}")
        print(f"    PDFs: {status['total_scraped_pdfs']}  Embedded: {status['total_embedded_docs']}  Processing: {processing_status}")
        
        if status['last_processed']:
            try:
                last_time = datetime.fromisoformat(status['last_processed'])
                print(f"    Last Processed: {last_time.strftime('%Y-%m-%d %H:%M')}")
            except:
                print(f"    Last Processed: {status['last_processed']}")
        
        if status['errors']:
            print(f"    ⚠️  Errors: {'; '.join(status['errors'][:2])}")

def setup_proceeding_directories(proceeding: str) -> bool:
    """Setup the embeddings directory structure for a proceeding."""
    try:
        import config
        paths = config.get_proceeding_file_paths(proceeding)
        
        # Create embeddings directory
        paths['embeddings_dir'].mkdir(parents=True, exist_ok=True)
        logger.info(f"Created embeddings directory: {paths['embeddings_dir']}")
        
        # Create vector DB directory
        paths['vector_db'].mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured vector DB directory: {paths['vector_db']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup directories for {proceeding}: {e}")
        return False

def load_scraped_pdf_data(proceeding: str) -> Optional[Dict]:
    """Load scraped PDF data for a proceeding."""
    try:
        import config
        paths = config.get_proceeding_file_paths(proceeding)
        
        if not paths['scraped_pdf_history'].exists():
            logger.warning(f"No scraped PDF history found for {proceeding}")
            return None
        
        with open(paths['scraped_pdf_history'], 'r') as f:
            scraped_data = json.load(f)
        
        logger.info(f"Loaded {len(scraped_data)} scraped PDFs for {proceeding}")
        return scraped_data
        
    except Exception as e:
        logger.error(f"Failed to load scraped data for {proceeding}: {e}")
        return None

def process_proceeding_documents(proceeding: str, batch_size: int = 10, force_rebuild: bool = False) -> Dict:
    """
    Process all documents for a proceeding into embeddings.
    
    Args:
        proceeding: Proceeding number (e.g., 'R2207005')
        batch_size: Number of documents to process in each batch
        force_rebuild: Whether to rebuild existing embeddings
        
    Returns:
        Dictionary with processing results
    """
    # Clean output for non-debug mode
    if not config.DEBUG:
        print(f"\n🚀 Starting data processing for proceeding: {proceeding}")
        print("="*60)
    else:
        logger.debug(f"🚀 Starting data processing for proceeding: {proceeding}")
    
    try:
        # Setup directories
        if not setup_proceeding_directories(proceeding):
            return {'status': 'error', 'error': 'Failed to setup directories'}
        
        # Load scraped PDF data
        scraped_data = load_scraped_pdf_data(proceeding)
        if not scraped_data:
            return {'status': 'no_data', 'error': 'No scraped PDF data found'}
        
        if not config.DEBUG:
            print(f"📄 Found {len(scraped_data)} scraped PDFs for {proceeding}")
        
        # Initialize incremental embedder
        from incremental_embedder import IncrementalEmbedder
        
        def progress_callback(message: str, progress: int):
            if config.VERBOSE_LOGGING:
                logger.debug(f"[{progress}%] {message}")
        
        embedder = IncrementalEmbedder(proceeding, progress_callback=progress_callback)
        
        # Process embeddings
        results = embedder.process_incremental_embeddings()
        
        # Log results with clean formatting
        if results['status'] == 'completed':
            if not config.DEBUG:
                print(f"✅ Processing completed for {proceeding}")
                print(f"   Documents processed: {results['documents_processed']}")
                print(f"   Successful: {results['successful']}")
                print(f"   Failed: {results['failed']}")
            else:
                logger.info(f"✅ Processing completed for {proceeding}")
                logger.info(f"   Documents processed: {results['documents_processed']}")
                logger.info(f"   Successful: {results['successful']}")
                logger.info(f"   Failed: {results['failed']}")
        else:
            if not config.DEBUG:
                print(f"⚠️ Processing status for {proceeding}: {results['status']}")
                if 'error' in results:
                    print(f"   Error: {results['error']}")
            else:
                logger.warning(f"⚠️ Processing status for {proceeding}: {results['status']}")
                if 'error' in results:
                    logger.error(f"   Error: {results['error']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process {proceeding}: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'documents_processed': 0
        }

def process_multiple_proceedings(proceedings: List[str], batch_size: int = 10, force_rebuild: bool = False) -> Dict:
    """Process multiple proceedings."""
    if not config.DEBUG:
        print(f"\n🚀 Starting data processing for {len(proceedings)} proceedings")
        print(f"📋 Proceedings: {', '.join(proceedings)}")
        print("="*80)
    else:
        logger.info(f"🚀 Starting data processing for {len(proceedings)} proceedings: {proceedings}")
    
    results = {
        'total_proceedings': len(proceedings),
        'successful': 0,
        'failed': 0,
        'proceedings_results': {},
        'summary': {}
    }
    
    for i, proceeding in enumerate(proceedings, 1):
        if not config.DEBUG:
            print(f"\n📊 Processing proceeding {i}/{len(proceedings)}: {proceeding}")
        else:
            logger.info(f"📊 Processing proceeding {i}/{len(proceedings)}: {proceeding}")
        
        try:
            result = process_proceeding_documents(proceeding, batch_size, force_rebuild)
            results['proceedings_results'][proceeding] = result
            
            if result.get('status') == 'completed':
                results['successful'] += 1
                if config.VERBOSE_LOGGING:
                    logger.info(f"✅ Completed {proceeding}: {result.get('documents_processed', 0)} documents")
            else:
                results['failed'] += 1
                if not config.DEBUG:
                    print(f"❌ Failed {proceeding}: {result.get('error', 'Unknown error')}")
                else:
                    logger.error(f"❌ Failed {proceeding}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            results['failed'] += 1
            results['proceedings_results'][proceeding] = {'status': 'error', 'error': str(e)}
            if not config.DEBUG:
                print(f"❌ Exception in {proceeding}: {e}")
            else:
                logger.error(f"❌ Exception in {proceeding}: {e}")
    
    # Calculate summary
    total_docs = sum(r.get('documents_processed', 0) for r in results['proceedings_results'].values())
    results['summary'] = {
        'total_documents_processed': total_docs,
        'success_rate': f"{(results['successful'] / results['total_proceedings']) * 100:.1f}%"
    }
    
    # Print final summary with clear formatting
    if not config.DEBUG:
        print(f"\n🎯 FINAL RESULTS")
        print("="*80)
        print(f"✅ Successful proceedings: {results['successful']}/{results['total_proceedings']}")
        print(f"📄 Total documents processed: {total_docs}")
        print(f"📊 Success rate: {results['summary']['success_rate']}")
        print("="*80)
    else:
        logger.info(f"🎯 Overall Results: {results['successful']}/{results['total_proceedings']} successful, {total_docs} total documents")
    
    return results

def show_detailed_status(proceeding: str) -> None:
    """Show detailed processing status for a proceeding."""
    status = get_proceeding_status(proceeding)
    
    print(f"\n📊 Detailed Status for {proceeding}")
    print("=" * 50)
    
    print(f"Scraped Data: {'✅ Found' if status['scraped_data_exists'] else '❌ Missing'}")
    if status['scraped_data_exists']:
        print(f"  Total PDFs: {status['total_scraped_pdfs']}")
    
    print(f"Embeddings Directory: {'✅ Exists' if status['embeddings_dir_exists'] else '❌ Missing'}")
    print(f"Vector Database: {'✅ Exists' if status['vector_db_exists'] else '❌ Missing'}")
    
    print(f"Processing Status: {status['processing_status']}")
    if status['total_embedded_docs'] > 0:
        progress = (status['total_embedded_docs'] / max(status['total_scraped_pdfs'], 1)) * 100
        print(f"  Progress: {status['total_embedded_docs']}/{status['total_scraped_pdfs']} ({progress:.1f}%)")
    
    if status['last_processed']:
        try:
            last_time = datetime.fromisoformat(status['last_processed'])
            print(f"  Last Processed: {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(f"  Last Processed: {status['last_processed']}")
    
    if status['errors']:
        print(f"\n⚠️  Errors:")
        for error in status['errors']:
            print(f"  • {error}")

def main():
    """Main entry point for standalone data processor."""
    args = parse_arguments()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Handle list proceedings request
    if args.list_proceedings:
        list_proceedings_status()
        return 0
    
    # Handle status request
    if args.status:
        show_detailed_status(args.status)
        return 0
    
    # Determine proceedings to process
    if args.proceeding:
        # Single proceeding specified
        proceedings = [args.proceeding]
        
        # Validate proceeding format
        if not args.proceeding.startswith('R') or len(args.proceeding) != 8:
            logger.error(f"Invalid proceeding format: {args.proceeding}")
            logger.error("Expected format: R2207005 (R + 7 digits)")
            return 1
    elif args.all:
        # Process all discovered proceedings
        proceedings = discover_available_proceedings()
        if not proceedings:
            logger.error("No proceedings found in cpuc_proceedings directory")
            return 1
    else:
        # No proceeding specified - use all from config
        proceedings = get_config_proceedings()
        logger.info(f"No proceeding specified, using all from config: {proceedings}")
    
    # Run the data processor
    print(f"\n🔍 CPUC Standalone Data Processor")
    print(f"=" * 50)
    if len(proceedings) == 1:
        print(f"Proceeding: {proceedings[0]}")
    else:
        print(f"Proceedings: {len(proceedings)} total")
        for i, p in enumerate(proceedings, 1):
            print(f"  {i}. {p}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Force Rebuild: {args.force_rebuild}")
    print(f"=" * 50)
    
    if len(proceedings) == 1:
        # Single proceeding
        results = process_proceeding_documents(
            proceedings[0], 
            batch_size=args.batch_size,
            force_rebuild=args.force_rebuild
        )
        
        if results.get('status') == 'completed':
            print(f"\n✅ Data processing completed successfully!")
            print(f"📊 Summary: {results.get('documents_processed', 0)} documents processed")
            return 0
        else:
            print(f"\n❌ Data processing failed: {results.get('error', 'Unknown error')}")
            return 1
    else:
        # Multiple proceedings
        results = process_multiple_proceedings(
            proceedings, 
            batch_size=args.batch_size,
            force_rebuild=args.force_rebuild
        )
        
        if results['successful'] > 0:
            print(f"\n✅ Data processing completed!")
            print(f"📊 Summary: {results['successful']}/{results['total_proceedings']} successful")
            print(f"📄 Total Documents: {results['summary']['total_documents_processed']}")
            print(f"📈 Success Rate: {results['summary']['success_rate']}")
            return 0 if results['failed'] == 0 else 1
        else:
            print(f"\n❌ All proceedings failed!")
            return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)