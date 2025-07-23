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
  %(prog)s R2207005           # Scrape specific proceeding
  %(prog)s --list-proceedings # List available proceedings
  %(prog)s --help             # Show this help message
        """
    )
    
    parser.add_argument(
        'proceeding',
        nargs='?',
        help='CPUC proceeding number (e.g., R2207005)'
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
    
    return parser.parse_args()

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
    
    # Check cpuc_csvs directory for existing proceedings
    csvs_dir = Path('cpuc_csvs')
    if csvs_dir.exists():
        csv_files = list(csvs_dir.glob('*_resultCSV.csv'))
        if csv_files:
            print("\n📋 Available Proceedings (with CSV data):")
            for csv_file in sorted(csv_files):
                proceeding = csv_file.stem.replace('_resultCSV', '')
                print(f"  • {proceeding}")
        else:
            print("\n📋 No proceedings found with CSV data")
    
    # Check for scraped PDF history files
    history_files = list(csvs_dir.glob('*_scraped_pdf_history.json')) if csvs_dir.exists() else []
    if history_files:
        print("\n📄 Proceedings with PDF History:")
        for history_file in sorted(history_files):
            proceeding = history_file.stem.replace('_scraped_pdf_history', '')
            print(f"  • {proceeding}")
    else:
        print("\n📄 No proceedings found with PDF history")
    
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
        
        # Initialize scraper
        logger.info("🔧 Initializing CPUC scraper...")
        scraper = CPUCSimplifiedScraper(headless=headless)
        
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
            proceeding_dir = Path(f'cpuc_csvs')
            if proceeding_dir.exists():
                csv_file = proceeding_dir / f"{proceeding}_resultCSV.csv"
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
    
    # Determine proceeding to scrape
    proceeding = args.proceeding
    if not proceeding:
        proceeding = get_default_proceeding()
        logger.info(f"No proceeding specified, using default: {proceeding}")
    
    # Validate proceeding format
    if not proceeding.startswith('R') or len(proceeding) != 8:
        logger.error(f"Invalid proceeding format: {proceeding}")
        logger.error("Expected format: R2207005 (R + 7 digits)")
        return 1
    
    # Run the scraper
    print(f"\n🔍 CPUC Standalone Scraper")
    print(f"=" * 50)
    print(f"Proceeding: {proceeding}")
    print(f"Headless Mode: {args.headless}")
    print(f"=" * 50)
    
    results = run_standalone_scraper(proceeding, args.headless)
    
    if results.get('success', True):
        print(f"\n✅ Scraping completed successfully!")
        print(f"📊 Summary: {results.get('total_pdfs', 0)} PDFs discovered")
        return 0
    else:
        print(f"\n❌ Scraping failed: {results.get('error', 'Unknown error')}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)