#!/usr/bin/env python3
"""
Standalone CPUC Document Scraper

This script runs the CPUC document scraper independently from the main application.
Use this to manually discover and process documents without starting the full RAG system.

Usage:
    python standalone_scraper.py [proceeding_number]
    
Examples:
    python standalone_scraper.py R2207005
    python standalone_scraper.py  # Uses default proceeding from config
    
Author: Claude Code
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging for standalone operation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('standalone_scraper.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone CPUC Document Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Scrape all proceedings from config
  %(prog)s R2207005           # Scrape specific proceeding
  %(prog)s --list-proceedings # List available proceedings
  %(prog)s --help             # Show this help message
        """
    )
    
    parser.add_argument(
        'proceeding',
        nargs='?',
        help='CPUC proceeding number (e.g., R2207005). If not specified, scrapes all proceedings from config.'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run browser in headless mode (default: True)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--list-proceedings',
        action='store_true',
        help='List available proceedings and exit'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Scrape all proceedings from config (same as not specifying a proceeding)'
    )
    
    return parser.parse_args()

def get_scraper_proceedings() -> list:
    """Get list of proceedings to scrape from config."""
    try:
        import config
        proceedings = getattr(config, 'SCRAPER_PROCEEDINGS', ['R2207005'])
        logger.info(f"Loaded {len(proceedings)} proceedings from config: {proceedings}")
        return proceedings
    except ImportError:
        logger.warning("Could not import config, using fallback default")
        return ['R2207005']

def get_default_proceeding() -> str:
    """Get default proceeding from config."""
    try:
        import config
        return getattr(config, 'DEFAULT_PROCEEDING', 'R2207005')
    except ImportError:
        logger.warning("Could not import config, using fallback default")
        return 'R2207005'

def list_available_proceedings() -> None:
    """List proceedings that have been previously scraped."""
    logger.info("Scanning for available proceedings...")
    
    # Show config proceedings first
    config_proceedings = get_scraper_proceedings()
    print(f"\n⚙️  Configured Proceedings (from config.py):")
    for proceeding in config_proceedings:
        print(f"  • {proceeding}")
    
    # Check cpuc_proceedings directory for existing proceedings
    cpuc_proceedings_dir = Path('cpuc_proceedings')
    if cpuc_proceedings_dir.exists():
        proceeding_dirs = [d for d in cpuc_proceedings_dir.iterdir() if d.is_dir()]
        if proceeding_dirs:
            print("\n📁 Available Proceedings (in cpuc_proceedings/):")
            for proceeding_dir in sorted(proceeding_dirs):
                proceeding_name = proceeding_dir.name
                
                # Check for CSV files
                documents_dir = proceeding_dir / "documents"
                csv_files = list(documents_dir.glob('*_documents.csv')) if documents_dir.exists() else []
                
                # Check for PDF history
                history_files = list(proceeding_dir.glob('*_scraped_pdf_history.json'))
                
                status_parts = []
                if csv_files:
                    status_parts.append("CSV")
                if history_files:
                    status_parts.append("PDF History")
                
                status = f" ({', '.join(status_parts)})" if status_parts else " (empty)"
                print(f"  • {proceeding_name}{status}")
        else:
            print("\n📁 No proceedings found in cpuc_proceedings directory")
    else:
        print("\n📁 cpuc_proceedings directory not found")
    
    # Check for legacy files in old locations
    csvs_dir = Path('cpuc_csvs')
    if csvs_dir.exists():
        csv_files = list(csvs_dir.glob('*_resultCSV.csv'))
        if csv_files:
            print("\n📋 Legacy Proceedings (in cpuc_csvs/ - old format):")
            for csv_file in sorted(csv_files):
                proceeding = csv_file.stem.replace('_resultCSV', '')
                print(f"  • {proceeding}")
    
    # Check local_chroma_db for vector stores
    db_dir = Path('local_chroma_db')
    if db_dir.exists():
        vector_dirs = [d for d in db_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
        if vector_dirs:
            print("\n🗄️ Proceedings with Vector Stores:")
            for vector_dir in sorted(vector_dirs):
                print(f"  • {vector_dir.name}")
        else:
            print("\n🗄️ No vector stores found")

def run_standalone_scraper(proceeding: str, headless: bool = True) -> Dict[str, Any]:
    """
    Run the scraper for a specific proceeding.
    
    Args:
        proceeding: CPUC proceeding number
        headless: Whether to run browser in headless mode
        
    Returns:
        Dictionary with scraping results
    """
    logger.info(f"🚀 Starting standalone scraper for proceeding: {proceeding}")
    logger.info(f"📁 Working directory: {Path.cwd()}")
    
    try:
        # Import the scraper
        from cpuc_scraper import CPUCSimplifiedScraper
        
        # Initialize scraper with background processing enabled
        logger.info("🔧 Initializing CPUC scraper for background processing...")
        scraper = CPUCSimplifiedScraper(headless=True)  # Force headless for background operation
        
        # Run scraping
        logger.info(f"🔍 Starting document discovery for {proceeding}...")
        results = scraper.scrape_proceeding(proceeding)
        
        # Log results
        if results.get('success', True):
            logger.info("✅ Scraping completed successfully!")
            logger.info(f"📊 Results Summary:")
            logger.info(f"   • CSV PDFs: {results.get('csv_pdfs', 0)}")
            logger.info(f"   • Google PDFs: {results.get('google_pdfs', 0)}")
            logger.info(f"   • Total PDFs: {results.get('total_pdfs', 0)}")
            
            # Check for created files
            proceeding_dir = Path('cpuc_proceedings') / proceeding
            if proceeding_dir.exists():
                csv_file = proceeding_dir / "documents" / f"{proceeding}_documents.csv"
                history_file = proceeding_dir / f"{proceeding}_scraped_pdf_history.json"
                
                logger.info(f"📁 Output Files:")
                if csv_file.exists():
                    size = csv_file.stat().st_size
                    logger.info(f"   • CSV: {csv_file} ({size:,} bytes)")
                if history_file.exists():
                    size = history_file.stat().st_size
                    logger.info(f"   • History: {history_file} ({size:,} bytes)")
            
        else:
            logger.error(f"❌ Scraping failed: {results.get('error', 'Unknown error')}")
            
        return results
        
    except ImportError as e:
        error_msg = f"Failed to import scraper module: {e}"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}
        
    except Exception as e:
        error_msg = f"Scraping failed with error: {e}"
        logger.error(error_msg, exc_info=True)
        return {'success': False, 'error': error_msg}
        
    finally:
        # Clean up scraper
        if 'scraper' in locals():
            try:
                scraper._cleanup_driver()
                logger.info("🧹 Scraper cleanup completed")
            except Exception as e:
                logger.warning(f"Scraper cleanup warning: {e}")

def run_multiple_proceedings(proceedings: list, headless: bool = True) -> Dict[str, Any]:
    """
    Run the scraper for multiple proceedings.
    
    Args:
        proceedings: List of CPUC proceeding numbers
        headless: Whether to run browser in headless mode
        
    Returns:
        Dictionary with overall scraping results
    """
    logger.info(f"🚀 Starting standalone scraper for {len(proceedings)} proceedings: {proceedings}")
    
    results = {
        'total_proceedings': len(proceedings),
        'successful': 0,
        'failed': 0,
        'proceedings_results': {},
        'summary': {}
    }
    
    for i, proceeding in enumerate(proceedings, 1):
        logger.info(f"📊 Processing proceeding {i}/{len(proceedings)}: {proceeding}")
        
        try:
            result = run_standalone_scraper(proceeding, headless)
            results['proceedings_results'][proceeding] = result
            
            if result.get('success', True):
                results['successful'] += 1
                logger.info(f"✅ Completed {proceeding}: {result.get('total_pdfs', 0)} PDFs")
            else:
                results['failed'] += 1
                logger.error(f"❌ Failed {proceeding}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            results['failed'] += 1
            results['proceedings_results'][proceeding] = {'success': False, 'error': str(e)}
            logger.error(f"❌ Exception in {proceeding}: {e}")
    
    # Calculate summary
    total_pdfs = sum(r.get('total_pdfs', 0) for r in results['proceedings_results'].values())
    results['summary'] = {
        'total_pdfs_discovered': total_pdfs,
        'success_rate': f"{(results['successful'] / results['total_proceedings']) * 100:.1f}%"
    }
    
    logger.info(f"🎯 Overall Results: {results['successful']}/{results['total_proceedings']} successful, {total_pdfs} total PDFs")
    return results

def main():
    """Main entry point for standalone scraper."""
    args = parse_arguments()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Handle list proceedings request
    if args.list_proceedings:
        list_available_proceedings()
        return 0
    
    # Determine proceedings to scrape
    if args.proceeding:
        # Single proceeding specified
        proceedings = [args.proceeding]
        
        # Validate proceeding format
        if not args.proceeding.startswith('R') or len(args.proceeding) != 8:
            logger.error(f"Invalid proceeding format: {args.proceeding}")
            logger.error("Expected format: R2207005 (R + 7 digits)")
            return 1
    else:
        # No proceeding specified - use all from config
        proceedings = get_scraper_proceedings()
        logger.info(f"No proceeding specified, using all from config: {proceedings}")
    
    # Run the scraper
    print(f"\n🔍 CPUC Standalone Scraper")
    print(f"=" * 50)
    if len(proceedings) == 1:
        print(f"Proceeding: {proceedings[0]}")
    else:
        print(f"Proceedings: {len(proceedings)} total")
        for i, p in enumerate(proceedings, 1):
            print(f"  {i}. {p}")
    print(f"Headless Mode: {args.headless}")
    print(f"=" * 50)
    
    if len(proceedings) == 1:
        # Single proceeding
        results = run_standalone_scraper(proceedings[0], args.headless)
        
        if results.get('success', True):
            print(f"\n✅ Scraping completed successfully!")
            print(f"📊 Summary: {results.get('total_pdfs', 0)} PDFs discovered")
            return 0
        else:
            print(f"\n❌ Scraping failed: {results.get('error', 'Unknown error')}")
            return 1
    else:
        # Multiple proceedings
        results = run_multiple_proceedings(proceedings, args.headless)
        
        if results['successful'] > 0:
            print(f"\n✅ Scraping completed!")
            print(f"📊 Summary: {results['successful']}/{results['total_proceedings']} successful")
            print(f"🗂️  Total PDFs: {results['summary']['total_pdfs_discovered']}")
            print(f"📈 Success Rate: {results['summary']['success_rate']}")
            return 0 if results['failed'] == 0 else 1
        else:
            print(f"\n❌ All proceedings failed!")
            return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)