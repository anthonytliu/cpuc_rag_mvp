#!/usr/bin/env python3
"""
Standalone CPUC Document Scraper

This script runs the CPUC document scraper independently from the main application.
Use this to manually discover and process documents without starting the full RAG system.

Usage:
    python standalone_scraper.py [proceeding_number]
    
Examples:
    python standalone_scraper.py                # Scrape ALL proceedings from config.SCRAPER_PROCEEDINGS
    python standalone_scraper.py R2207005       # Scrape specific proceeding only
    python standalone_scraper.py --list-proceedings  # List available proceedings
    
Default Behavior:
    When called with no arguments, scrapes all proceedings listed in config.SCRAPER_PROCEEDINGS
    starting from the first entry in the list.
    
CSV Age Optimization:
    For each proceeding, checks if a valid CSV file exists and is less than a week old.
    - If CSV < 7 days old: Skips CSV download, only runs Google search for new PDFs
    - If CSV >= 7 days old or missing: Downloads fresh CSV and runs full scraping process
    
Author: Claude Code
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta
import os

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

# Import CSV cleaning function from cpuc_scraper to avoid duplication
def _get_csv_cleaner():
    """Import and return the CSV cleaning function from cpuc_scraper"""
    try:
        # Add src to path and import cpuc_scraper
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        from scrapers.cpuc_scraper import CPUCSimplifiedScraper
        # Create temporary instance to access the cleaning method
        temp_scraper = CPUCSimplifiedScraper()
        return temp_scraper._clean_csv_document_types
    except ImportError as e:
        logger.error(f"Failed to import CSV cleaner from cpuc_scraper: {e}")
        return None

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
  %(prog)s --clean-all-csvs   # Clean all existing CSV files
  %(prog)s --help             # Show this help message

Optimization:
  - Skips CSV scraping if recent file exists with >=98%% coverage (configurable)
  - Skips Google search if coverage is >=100%% (comprehensive coverage achieved)
  - Runs Google search only if coverage meets threshold but <100%% (to find potential new docs)
  
CSV Cleaning:
  - Removes invalid Document Type links from CSV files
  - Ensures accurate coverage calculations
  - Creates backups before modifying files
        """
    )
    
    parser.add_argument(
        'proceeding',
        nargs='?',
        help='CPUC proceeding number (e.g., R2207005). If not specified, scrapes ALL proceedings from config.SCRAPER_PROCEEDINGS starting from the first entry.'
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
    
    parser.add_argument(
        '--clean-all-csvs',
        action='store_true',
        help='Clean all existing documents.csv files in cpuc_proceedings subfolders and exit'
    )
    
    return parser.parse_args()

def get_scraper_proceedings() -> list:
    """Get list of proceedings to scrape from config."""
    try:
        # Add src to path and import config
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        from core import config
        proceedings = getattr(config, 'SCRAPER_PROCEEDINGS', ['R2207005'])
        logger.info(f"Loaded {len(proceedings)} proceedings from config: {proceedings}")
        return proceedings
    except ImportError:
        logger.warning("Could not import config, using fallback default")
        return ['R2207005']

def get_coverage_threshold() -> float:
    """Get coverage threshold from config."""
    try:
        import config
        threshold = getattr(config, 'SCRAPER_COVERAGE_THRESHOLD', 100.0)
        logger.debug(f"Using coverage threshold: {threshold}%")
        return threshold
    except ImportError:
        logger.warning("Could not import config, using fallback threshold: 100%")
        return 100.0

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
    
    # Check local_lance_db for vector stores
    db_dir = Path('local_lance_db')
    if db_dir.exists():
        vector_dirs = [d for d in db_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
        if vector_dirs:
            print("\n🗄️ Proceedings with Vector Stores:")
            for vector_dir in sorted(vector_dirs):
                print(f"  • {vector_dir.name}")
        else:
            print("\n🗄️ No vector stores found")

def check_coverage_percentage(proceeding: str, csv_file: Path) -> Dict[str, Any]:
    """
    Check the coverage percentage between CSV entries and scraped history entries.
    
    Args:
        proceeding: CPUC proceeding number
        csv_file: Path to the CSV file
        
    Returns:
        Dictionary with coverage information:
        - csv_count: number of entries in CSV
        - history_count: number of entries in scraped history
        - coverage_percent: percentage coverage
        - has_complete_coverage: whether coverage is >= 100%
    """
    coverage_result = {
        'csv_count': 0,
        'history_count': 0,
        'coverage_percent': 0.0,
        'has_complete_coverage': False
    }
    
    try:
        # Count CSV entries - clean CSV first for accurate coverage calculation
        if csv_file.exists():
            import pandas as pd
            try:
                df = pd.read_csv(csv_file)
                # Clean CSV to remove invalid document type links before counting
                csv_cleaner = _get_csv_cleaner()
                if csv_cleaner:
                    cleaned_df = csv_cleaner(df)
                else:
                    logger.warning("CSV cleaner not available, using uncleaned data")
                    cleaned_df = df
                coverage_result['csv_count'] = len(cleaned_df)
                logger.debug(f"Coverage calculation: Using cleaned CSV count ({len(cleaned_df)} valid rows from {len(df)} original rows)")
            except Exception as e:
                logger.warning(f"Failed to read CSV file for coverage check: {e}")
                return coverage_result
        
        # Count scraped history entries
        proceeding_dir = csv_file.parent.parent  # Go up from documents/ to proceeding/
        history_file = proceeding_dir / f"{proceeding}_scraped_pdf_history.json"
        
        if history_file.exists():
            import json
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                coverage_result['history_count'] = len(history_data)
            except Exception as e:
                logger.warning(f"Failed to read history file for coverage check: {e}")
                return coverage_result
        
        # Calculate coverage percentage
        if coverage_result['csv_count'] > 0:
            coverage_result['coverage_percent'] = (coverage_result['history_count'] / coverage_result['csv_count']) * 100
            coverage_result['has_complete_coverage'] = coverage_result['coverage_percent'] >= 100.0
        else:
            coverage_result['coverage_percent'] = 0.0
            coverage_result['has_complete_coverage'] = False
        
        logger.debug(f"Coverage check: {coverage_result['history_count']}/{coverage_result['csv_count']} = {coverage_result['coverage_percent']:.1f}%")
        
    except Exception as e:
        logger.error(f"Coverage percentage check failed: {e}")
    
    return coverage_result

def check_existing_csv_file(proceeding: str) -> Dict[str, Any]:
    """
    Check if a valid, recent CSV file exists for the proceeding with complete coverage.
    
    Args:
        proceeding: CPUC proceeding number
        
    Returns:
        Dictionary with check results:
        - exists: bool - whether CSV file exists
        - is_recent: bool - whether file is less than a week old
        - age_days: int - age of file in days
        - path: Path - path to CSV file
        - csv_count: int - number of entries in CSV
        - history_count: int - number of entries in scraped history
        - coverage_percent: float - percentage coverage (history/csv * 100)
        - has_complete_coverage: bool - whether coverage is >= 100%
        - should_skip_csv: bool - whether to skip CSV scraping
    """
    proceeding_dir = Path('cpuc_proceedings') / proceeding
    documents_dir = proceeding_dir / 'documents'
    csv_file = documents_dir / f"{proceeding}_documents.csv"
    
    result = {
        'exists': False,
        'is_recent': False,
        'age_days': None,
        'path': csv_file,
        'file_size': 0,
        'csv_count': 0,
        'history_count': 0,
        'coverage_percent': 0.0,
        'has_complete_coverage': False,
        'should_skip_csv': False
    }
    
    if not csv_file.exists():
        logger.info(f"📋 No existing CSV file found: {csv_file}")
        return result
    
    # Check file size (empty or very small files are considered invalid)
    file_size = csv_file.stat().st_size
    result['file_size'] = file_size
    
    if file_size < 50:  # Less than 50 bytes is likely empty/invalid
        logger.info(f"📋 CSV file exists but is too small ({file_size} bytes): {csv_file}")
        return result
    
    result['exists'] = True
    
    # Check file age
    file_mtime = csv_file.stat().st_mtime
    file_date = datetime.fromtimestamp(file_mtime)
    current_date = datetime.now()
    age_delta = current_date - file_date
    age_days = age_delta.days
    
    result['age_days'] = age_days
    result['is_recent'] = age_days < 7  # Less than a week old
    
    # If CSV is recent, check coverage percentage
    if result['is_recent']:
        coverage_check = check_coverage_percentage(proceeding, csv_file)
        result.update(coverage_check)
        
        # Get coverage threshold from config
        coverage_threshold = get_coverage_threshold()
        
        # Only skip CSV if recent AND coverage meets threshold
        result['should_skip_csv'] = result['is_recent'] and result['coverage_percent'] >= coverage_threshold
        
        logger.info(f"📋 Found existing CSV file: {csv_file}")
        logger.info(f"   Size: {file_size:,} bytes")
        logger.info(f"   Age: {age_days} days old")
        logger.info(f"   CSV entries: {result['csv_count']}")
        logger.info(f"   History entries: {result['history_count']}")
        logger.info(f"   Coverage: {result['coverage_percent']:.1f}%")
        coverage_status = 'Complete' if result['coverage_percent'] >= coverage_threshold else 'Incomplete'
        logger.info(f"   Status: Recent with {coverage_status} coverage (threshold: {coverage_threshold}%)")
        
        if result['coverage_percent'] < coverage_threshold:
            logger.info(f"   ⚠️  Coverage is below {coverage_threshold}% - will run full scraping to ensure completeness")
    else:
        result['should_skip_csv'] = False
        logger.info(f"📋 Found existing CSV file: {csv_file}")
        logger.info(f"   Size: {file_size:,} bytes")
        logger.info(f"   Age: {age_days} days old")
        logger.info(f"   Status: Stale (skipping coverage check)")
    
    return result

def run_google_search_only(scraper, proceeding: str, csv_check: Dict) -> Dict[str, Any]:
    """
    Run only Google search for a proceeding using existing CSV data.
    
    Args:
        scraper: Initialized CPUCSimplifiedScraper instance
        proceeding: CPUC proceeding number
        csv_check: Results from CSV file check
        
    Returns:
        Dictionary with Google search results
    """
    try:
        # Load existing CSV data as base
        existing_pdfs = []
        csv_path = csv_check['path']
        
        if csv_path.exists():
            import pandas as pd
            try:
                df = pd.read_csv(csv_path)
                existing_pdfs = []
                for _, row in df.iterrows():
                    pdf_info = {
                        'pdf_url': row.get('PDF URL', ''),
                        'title': row.get('Title', ''),
                        'source': 'csv'  # Mark as CSV source
                    }
                    existing_pdfs.append(pdf_info)
                logger.info(f"📋 Loaded {len(existing_pdfs)} existing PDFs from CSV")
            except Exception as e:
                logger.warning(f"Failed to load existing CSV data: {e}")
        
        # Initialize proceeding folder structure
        proceeding_folder = Path('cpuc_proceedings') / proceeding
        proceeding_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize PDF history file
        scraper._initialize_pdf_history_file(proceeding_folder)
        
        # Create a simple progress tracker
        class SimpleProgress:
            def update(self, step, current, total):
                pass
        
        progress = SimpleProgress()
        
        # Run Google search for additional PDFs
        logger.info(f"🔍 Searching Google for additional PDFs...")
        additional_pdfs = scraper._google_search_for_pdfs(proceeding, existing_pdfs, progress)
        
        # Update PDF history with any new Google results
        if additional_pdfs:
            scraper._update_pdf_history_incremental(proceeding_folder, additional_pdfs)
            logger.info(f"🔍 Found {len(additional_pdfs)} additional PDFs from Google search")
        else:
            logger.info("🔍 No additional PDFs found from Google search")
        
        # Return results in expected format
        results = {
            'success': True,
            'csv_pdfs': len(existing_pdfs),
            'google_pdfs': len(additional_pdfs), 
            'total_pdfs': len(existing_pdfs) + len(additional_pdfs),
            'proceeding': proceeding,
            'mode': 'google_search_only'
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Google search failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'csv_pdfs': 0,
            'google_pdfs': 0,
            'total_pdfs': 0,
            'mode': 'google_search_only'
        }

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
    
    # Check if we have a recent CSV file
    csv_check = check_existing_csv_file(proceeding)
    
    try:
        # Import the scraper
        from cpuc_scraper import CPUCSimplifiedScraper
        
        # Initialize scraper with background processing enabled
        logger.info("🔧 Initializing CPUC scraper for background processing...")
        scraper = CPUCSimplifiedScraper(headless=True)  # Force headless for background operation
        
        # Get coverage threshold from config
        coverage_threshold = get_coverage_threshold()
        
        if csv_check['should_skip_csv']:
            # CSV is recent with coverage meeting threshold - skip both CSV scraping AND Google search
            logger.info(f"⏭️  Skipping CSV scraping (file is {csv_check['age_days']} days old with {csv_check['coverage_percent']:.1f}% coverage)")
            
            if csv_check['coverage_percent'] >= 100:
                # >=100% coverage means we have all or more documents than the official CSV
                # Skip both CSV scraping and Google search since we already have comprehensive coverage
                logger.info(f"✅ Coverage is {csv_check['coverage_percent']:.1f}% (>=100%) - skipping Google search as well")
                logger.info(f"📚 Using existing data: {csv_check['history_count']} documents already discovered")
                
                # Return success with existing data count
                results = {
                    'success': True,
                    'total_pdfs': csv_check['history_count'],
                    'csv_skipped': True,
                    'google_search_skipped': True,
                    'csv_age_days': csv_check['age_days'],
                    'coverage_percent': csv_check['coverage_percent'],
                    'message': f"Comprehensive coverage ({csv_check['coverage_percent']:.1f}%) - no scraping needed"
                }
            else:
                # Coverage meets threshold but is below 100% - run Google search for potential additional documents
                logger.info(f"🔍 Coverage is {csv_check['coverage_percent']:.1f}% (meets {coverage_threshold}% threshold) - running Google search for potential additional documents...")
                
                # Run Google search only by calling the scraper with a flag
                results = run_google_search_only(scraper, proceeding, csv_check)
                results['csv_skipped'] = True
                results['google_search_skipped'] = False
                results['csv_age_days'] = csv_check['age_days']
                results['coverage_percent'] = csv_check['coverage_percent']
            
        else:
            # Run full scraping process
            if csv_check['exists'] and csv_check['is_recent'] and csv_check['coverage_percent'] < coverage_threshold:
                logger.info(f"🔄 CSV file is recent but coverage is {csv_check['coverage_percent']:.1f}% (below {coverage_threshold}% threshold) - running full scraping")
            elif csv_check['exists']:
                logger.info(f"🔄 CSV file is {csv_check['age_days']} days old - running full scraping")
            else:
                logger.info(f"🆕 No CSV file found - running full scraping")
            
            logger.info(f"🔍 Starting full document discovery for {proceeding}...")
            results = scraper.scrape_proceeding(proceeding)
            results['csv_skipped'] = False
        
        # Log results
        if results.get('success', True):
            if results.get('google_search_skipped', False):
                # Both CSV and Google search were skipped due to >100% coverage
                logger.info("✅ Scraping skipped - comprehensive coverage already achieved!")
                logger.info(f"📊 Results Summary (No Scraping Needed):")
                logger.info(f"   • Coverage: {results.get('coverage_percent', 0):.1f}% (exceeds 100%)")
                logger.info(f"   • Total PDFs: {results.get('total_pdfs', 0)} (existing)")
                logger.info(f"   • CSV file age: {results.get('csv_age_days', 0)} days")
                logger.info(f"   • Status: {results.get('message', 'Complete')}")
            elif results.get('mode') == 'google_search_only':
                logger.info("✅ Google search completed successfully!")
                logger.info(f"📊 Results Summary (Google Search Only):")
                logger.info(f"   • Existing CSV PDFs: {results.get('csv_pdfs', 0)}")
                logger.info(f"   • New Google PDFs: {results.get('google_pdfs', 0)}")
                logger.info(f"   • Total PDFs: {results.get('total_pdfs', 0)}")
                if results.get('csv_age_days') is not None:
                    logger.info(f"   • CSV file age: {results.get('csv_age_days')} days")
            else:
                logger.info("✅ Full scraping completed successfully!")
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
                    status = "existing" if results.get('csv_skipped', False) else "updated"
                    logger.info(f"   • CSV: {csv_file} ({size:,} bytes) [{status}]")
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
                mode = result.get('mode', 'full_scraping')
                mode_indicator = " (Google only)" if mode == 'google_search_only' else ""
                logger.info(f"✅ Completed {proceeding}: {result.get('total_pdfs', 0)} PDFs{mode_indicator}")
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

def clean_all_existing_csvs() -> Dict[str, Any]:
    """
    Clean all existing documents.csv files in cpuc_proceedings subfolders.
    This removes invalid Document Type links from all CSV files to ensure 
    accurate coverage calculations and processing.
    
    Returns:
        Dictionary with cleaning results and statistics
    """
    logger.info("🧹 Starting cleanup of all existing documents.csv files...")
    
    results = {
        'total_csvs_found': 0,
        'csvs_cleaned': 0,
        'csvs_failed': 0,
        'total_rows_removed': 0,
        'total_rows_kept': 0,
        'csv_details': {}
    }
    
    # Get CSV cleaner function
    csv_cleaner = _get_csv_cleaner()
    if not csv_cleaner:
        logger.error("❌ Failed to get CSV cleaner function from cpuc_scraper.py")
        return {'error': 'CSV cleaner not available'}
    
    # Scan cpuc_proceedings directory
    cpuc_proceedings_dir = Path('cpuc_proceedings')
    if not cpuc_proceedings_dir.exists():
        logger.warning("📁 cpuc_proceedings directory not found")
        return results
    
    # Find all documents.csv files
    csv_files = []
    for proceeding_dir in cpuc_proceedings_dir.iterdir():
        if proceeding_dir.is_dir():
            documents_dir = proceeding_dir / 'documents'
            if documents_dir.exists():
                # Look for CSV files matching pattern: {proceeding}_documents.csv
                proceeding_name = proceeding_dir.name
                csv_file = documents_dir / f"{proceeding_name}_documents.csv"
                if csv_file.exists():
                    csv_files.append((proceeding_name, csv_file))
    
    results['total_csvs_found'] = len(csv_files)
    logger.info(f"📊 Found {len(csv_files)} documents.csv files to clean")
    
    if len(csv_files) == 0:
        logger.info("✅ No CSV files found to clean")
        return results
    
    # Clean each CSV file
    import pandas as pd
    
    for proceeding_name, csv_file in csv_files:
        logger.info(f"🧹 Cleaning CSV for proceeding: {proceeding_name}")
        logger.info(f"   File: {csv_file}")
        
        try:
            # Read the CSV file
            df_original = pd.read_csv(csv_file)
            original_count = len(df_original)
            logger.info(f"   Original rows: {original_count}")
            
            # Clean the CSV
            df_cleaned = csv_cleaner(df_original)
            cleaned_count = len(df_cleaned)
            removed_count = original_count - cleaned_count
            
            logger.info(f"   Cleaned rows: {cleaned_count}")
            logger.info(f"   Removed rows: {removed_count}")
            
            # Update statistics
            results['total_rows_removed'] += removed_count
            results['total_rows_kept'] += cleaned_count
            
            # Save cleaned CSV back to file if there were changes
            if removed_count > 0:
                # Create backup first
                backup_file = csv_file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                df_original.to_csv(backup_file, index=False)
                logger.info(f"   💾 Created backup: {backup_file}")
                
                # Save cleaned data
                df_cleaned.to_csv(csv_file, index=False)
                logger.info(f"   ✅ Cleaned CSV saved with {removed_count} rows removed")
            else:
                logger.info(f"   ✅ CSV was already clean, no changes needed")
            
            # Record details
            results['csv_details'][proceeding_name] = {
                'file': str(csv_file),
                'original_rows': original_count,
                'cleaned_rows': cleaned_count,
                'removed_rows': removed_count,
                'backup_created': removed_count > 0,
                'status': 'success'
            }
            
            results['csvs_cleaned'] += 1
            
        except Exception as e:
            logger.error(f"   ❌ Failed to clean CSV for {proceeding_name}: {e}")
            results['csv_details'][proceeding_name] = {
                'file': str(csv_file),
                'status': 'failed',
                'error': str(e)
            }
            results['csvs_failed'] += 1
    
    # Log summary
    logger.info(f"🎯 CSV Cleaning Summary:")
    logger.info(f"   📊 Total CSVs found: {results['total_csvs_found']}")
    logger.info(f"   ✅ CSVs cleaned successfully: {results['csvs_cleaned']}")
    logger.info(f"   ❌ CSVs failed: {results['csvs_failed']}")
    logger.info(f"   🗑️  Total rows removed: {results['total_rows_removed']}")
    logger.info(f"   📝 Total rows kept: {results['total_rows_kept']}")
    
    if results['csvs_failed'] > 0:
        logger.warning(f"⚠️  {results['csvs_failed']} CSV files failed cleaning - check logs for details")
    
    logger.info("✅ CSV cleaning completed!")
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
    
    # Handle clean all CSVs request
    if args.clean_all_csvs:
        print(f"\n🧹 CPUC CSV Cleaner")
        print(f"=" * 50)
        print(f"Cleaning all documents.csv files in cpuc_proceedings/")
        print(f"=" * 50)
        
        results = clean_all_existing_csvs()
        
        if 'error' in results:
            print(f"\n❌ CSV cleaning failed: {results['error']}")
            return 1
        
        print(f"\n✅ CSV cleaning completed!")
        print(f"📊 Summary:")
        print(f"   • CSVs found: {results['total_csvs_found']}")
        print(f"   • CSVs cleaned: {results['csvs_cleaned']}")
        print(f"   • CSVs failed: {results['csvs_failed']}")
        print(f"   • Rows removed: {results['total_rows_removed']}")
        print(f"   • Rows kept: {results['total_rows_kept']}")
        
        return 0 if results['csvs_failed'] == 0 else 1
    
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
        logger.info(f"No proceeding specified - scraping ALL proceedings from config.SCRAPER_PROCEEDINGS")
        logger.info(f"Total proceedings to scrape: {len(proceedings)}")
        logger.debug(f"Proceedings list: {proceedings}")
    
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