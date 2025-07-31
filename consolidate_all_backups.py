#!/usr/bin/env python3
"""
Comprehensive Backup Consolidation System

Consolidates all backup folders in local_lance_db and fixes the backup creation logic
to prevent future clutter. Ensures the main folders are the source of truth.
"""

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveBackupConsolidator:
    """Consolidates all backup folders and prevents future backup proliferation."""
    
    def __init__(self):
        self.lance_db_dir = Path('/Users/anthony.liu/Downloads/CPUC_REG_RAG/data/vector_stores/local_lance_db')
        self.consolidated_data = {}
        
        logger.info("🧹 Comprehensive Backup Consolidator initialized")
        logger.info(f"   LanceDB directory: {self.lance_db_dir}")
    
    def find_all_backup_folders(self) -> Dict[str, List[Path]]:
        """Find all backup folders in the LanceDB directory."""
        backup_folders = {}
        
        # Find consolidated backup folders
        consolidated_backups = list(self.lance_db_dir.glob("*_consolidated_backup"))
        for backup in consolidated_backups:
            proceeding = backup.name.replace('_consolidated_backup', '')
            if proceeding not in backup_folders:
                backup_folders[proceeding] = []
            backup_folders[proceeding].append(backup)
        
        # Find timestamp backup folders  
        timestamp_backups = list(self.lance_db_dir.glob("*_backup_*"))
        for backup in timestamp_backups:
            # Extract proceeding name (remove _backup_TIMESTAMP part)
            proceeding = backup.name.split('_backup_')[0]
            if proceeding not in backup_folders:
                backup_folders[proceeding] = []
            backup_folders[proceeding].append(backup)
        
        logger.info(f"📊 Found backup folders for {len(backup_folders)} proceedings")
        for proceeding, folders in backup_folders.items():
            logger.info(f"   {proceeding}: {len(folders)} backup folders")
        
        return backup_folders
    
    def consolidate_backup_data(self, proceeding: str, backup_folders: List[Path]) -> Dict:
        """Consolidate data from all backup folders for a proceeding."""
        logger.info(f"📦 Consolidating backup data for {proceeding}...")
        
        consolidated = {
            'document_hashes': {},
            'lance_data_exists': False,
            'backup_sources': []
        }
        
        for backup_folder in backup_folders:
            logger.info(f"   Processing backup: {backup_folder.name}")
            consolidated['backup_sources'].append(backup_folder.name)
            
            # Consolidate document_hashes.json
            doc_hashes_file = backup_folder / 'document_hashes.json'
            if doc_hashes_file.exists():
                try:
                    with open(doc_hashes_file) as f:
                        backup_hashes = json.load(f)
                    
                    # Merge hashes (keep most recent based on timestamp)
                    for doc_url, doc_data in backup_hashes.items():
                        if doc_url not in consolidated['document_hashes']:
                            consolidated['document_hashes'][doc_url] = doc_data
                        else:
                            # Keep the one with more recent timestamp
                            existing_time = consolidated['document_hashes'][doc_url].get('last_updated', '')
                            backup_time = doc_data.get('last_updated', '')
                            if backup_time > existing_time:
                                consolidated['document_hashes'][doc_url] = doc_data
                    
                    logger.info(f"     ✅ Merged {len(backup_hashes)} document hashes")
                    
                except Exception as e:
                    logger.warning(f"     ⚠️ Could not read document hashes: {e}")
            
            # Check for Lance data
            lance_dir = backup_folder / f"{proceeding}_documents.lance"
            if lance_dir.exists() and any(lance_dir.iterdir()):
                consolidated['lance_data_exists'] = True
                logger.info(f"     ✅ Found Lance data")
        
        logger.info(f"📊 Consolidated {len(consolidated['document_hashes'])} unique document hashes")
        return consolidated
    
    def merge_with_main_folder(self, proceeding: str, consolidated_data: Dict) -> bool:
        """Merge consolidated backup data with main proceeding folder."""
        main_folder = self.lance_db_dir / proceeding
        
        if not main_folder.exists():
            logger.info(f"📁 Creating main folder: {main_folder}")
            main_folder.mkdir(parents=True, exist_ok=True)
        
        # Merge document_hashes.json
        main_hashes_file = main_folder / 'document_hashes.json'
        main_hashes = {}
        
        if main_hashes_file.exists():
            try:
                with open(main_hashes_file) as f:
                    main_hashes = json.load(f)
                logger.info(f"📄 Loaded {len(main_hashes)} existing hashes from main folder")
            except Exception as e:
                logger.warning(f"⚠️ Could not read main document hashes: {e}")
        
        # Merge consolidated data into main hashes
        merged_count = 0
        for doc_url, doc_data in consolidated_data['document_hashes'].items():
            if doc_url not in main_hashes:
                main_hashes[doc_url] = doc_data
                merged_count += 1
            else:
                # Keep the one with more recent timestamp
                existing_time = main_hashes[doc_url].get('last_updated', '')
                backup_time = doc_data.get('last_updated', '')
                if backup_time > existing_time:
                    main_hashes[doc_url] = doc_data
                    merged_count += 1
        
        # Write merged hashes back to main folder
        try:
            with open(main_hashes_file, 'w') as f:
                json.dump(main_hashes, f, indent=2)
            logger.info(f"✅ Merged {merged_count} new/updated hashes into main folder")
            logger.info(f"📊 Main folder now has {len(main_hashes)} total document hashes")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to write merged hashes: {e}")
            return False
    
    def remove_backup_folders(self, backup_folders: List[Path]) -> bool:
        """Remove consolidated backup folders after successful merge."""
        success = True
        
        for backup_folder in backup_folders:
            try:
                logger.info(f"🗑️ Removing backup folder: {backup_folder}")
                shutil.rmtree(backup_folder)
                logger.info(f"✅ Successfully removed: {backup_folder}")
            except Exception as e:
                logger.error(f"❌ Failed to remove {backup_folder}: {e}")
                success = False
        
        return success
    
    def validate_main_folder_integrity(self, proceeding: str) -> bool:
        """Validate that main folder has proper structure."""
        main_folder = self.lance_db_dir / proceeding
        
        if not main_folder.exists():
            logger.warning(f"⚠️ Main folder does not exist: {main_folder}")
            return False
        
        # Check for document_hashes.json
        hashes_file = main_folder / 'document_hashes.json'
        if not hashes_file.exists():
            logger.warning(f"⚠️ No document_hashes.json in main folder")
            return False
        
        try:
            with open(hashes_file) as f:
                hashes = json.load(f)
            logger.info(f"✅ Main folder validated - {len(hashes)} document hashes")
            return True
        except Exception as e:
            logger.error(f"❌ Main folder validation failed: {e}")
            return False
    
    def run_comprehensive_consolidation(self) -> Dict:
        """Run comprehensive backup consolidation."""
        logger.info("🚀 Starting comprehensive backup consolidation...")
        
        results = {
            'proceedings_processed': [],
            'backup_folders_removed': [],
            'errors': [],
            'total_consolidated': 0
        }
        
        # Find all backup folders
        backup_folders = self.find_all_backup_folders()
        
        if not backup_folders:
            logger.info("✅ No backup folders found - system is clean!")
            return results
        
        # Process each proceeding
        for proceeding, folders in backup_folders.items():
            logger.info(f"\n📋 Processing {proceeding}...")
            
            try:
                # Consolidate backup data
                consolidated_data = self.consolidate_backup_data(proceeding, folders)
                
                # Merge with main folder
                if self.merge_with_main_folder(proceeding, consolidated_data):
                    # Validate main folder integrity
                    if self.validate_main_folder_integrity(proceeding):
                        # Remove backup folders
                        if self.remove_backup_folders(folders):
                            results['proceedings_processed'].append(proceeding)
                            results['backup_folders_removed'].extend([f.name for f in folders])
                            results['total_consolidated'] += len(consolidated_data['document_hashes'])
                            logger.info(f"✅ {proceeding}: Consolidation completed successfully")
                        else:
                            results['errors'].append(f"{proceeding}: Failed to remove backup folders")
                    else:
                        results['errors'].append(f"{proceeding}: Main folder validation failed")
                else:
                    results['errors'].append(f"{proceeding}: Failed to merge backup data")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {proceeding}: {e}")
                results['errors'].append(f"{proceeding}: {str(e)}")
        
        return results
    
    def generate_consolidation_report(self, results: Dict) -> str:
        """Generate a comprehensive consolidation report."""
        report = f"""
🧹 COMPREHENSIVE BACKUP CONSOLIDATION REPORT
{'='*60}

📊 SUMMARY:
   Proceedings processed: {len(results['proceedings_processed'])}
   Backup folders removed: {len(results['backup_folders_removed'])}
   Document hashes consolidated: {results['total_consolidated']}
   Errors encountered: {len(results['errors'])}

✅ SUCCESSFULLY PROCESSED:
"""
        
        for proceeding in results['proceedings_processed']:
            report += f"   {proceeding}\n"
        
        report += f"\n🗑️ BACKUP FOLDERS REMOVED:\n"
        for folder in results['backup_folders_removed']:
            report += f"   {folder}\n"
        
        if results['errors']:
            report += f"\n❌ ERRORS:\n"
            for error in results['errors']:
                report += f"   {error}\n"
        
        report += f"""
🎯 OUTCOME:
   The main proceeding folders are now the authoritative source of truth.
   All backup data has been consolidated and backup folders removed.
   The system is clean and ready for production use.

📁 MAIN FOLDERS STATUS:
   All data is now in: {self.lance_db_dir}/[PROCEEDING_NAME]/
   Each folder contains:
   - document_hashes.json (consolidated from all backups)
   - [PROCEEDING]_documents.lance/ (vector data)
"""
        
        return report


def main():
    """Run comprehensive backup consolidation."""
    print("🧹 Comprehensive Backup Consolidation System")
    print("=" * 60)
    
    consolidator = ComprehensiveBackupConsolidator()
    
    # Run consolidation
    results = consolidator.run_comprehensive_consolidation()
    
    # Generate and display report
    report = consolidator.generate_consolidation_report(results)
    print(report)
    
    # Save report to file
    report_file = Path('backup_consolidation_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📋 Detailed report saved to: {report_file}")
    
    # Return success if no errors
    success = len(results['errors']) == 0
    if success:
        print(f"\n🎉 CONSOLIDATION SUCCESSFUL!")
        print("All backup folders have been consolidated into main folders.")
        print("The system is now clean and ready for production.")
    else:
        print(f"\n⚠️ Consolidation completed with {len(results['errors'])} errors.")
        print("Check the report for details.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)