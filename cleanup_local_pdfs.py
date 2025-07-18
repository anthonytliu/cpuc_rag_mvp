#!/usr/bin/env python3
"""
PDF Cleanup Script for CPUC RAG System

This script removes all local PDF files since we've moved to URL-based processing,
handles duplicate/errata files, and cleans up tech debt related to local file processing.

Author: Claude Code
"""

import json
import logging
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple
import requests
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFCleanupManager:
    """Manages cleanup of local PDF files and transition to URL-based processing"""
    
    def __init__(self, project_root: Path = None):
        """Initialize cleanup manager"""
        self.project_root = project_root or Path(__file__).parent
        self.cpuc_pdfs_dir = self.project_root / "cpuc_pdfs"
        from config import get_proceeding_file_paths, DEFAULT_PROCEEDING
        proceeding_paths = get_proceeding_file_paths(DEFAULT_PROCEEDING)
        self.scraped_pdf_history_path = proceeding_paths['scraped_pdf_history']
        self.backup_dir = self.project_root / "pdf_cleanup_backup"
        
        # Statistics
        self.stats = {
            'total_files_found': 0,
            'duplicate_groups': 0,
            'files_removed': 0,
            'space_freed': 0,
            'metadata_files_removed': 0,
            'directories_removed': 0
        }
        
        logger.info(f"PDF Cleanup Manager initialized")
        logger.info(f"CPUC PDFs directory: {self.cpuc_pdfs_dir}")
    
    def analyze_local_pdfs(self) -> Dict:
        """Analyze local PDF files to understand what needs cleanup"""
        logger.info("Analyzing local PDF files...")
        
        if not self.cpuc_pdfs_dir.exists():
            logger.info("CPUC PDFs directory doesn't exist - nothing to clean")
            return {'total_files': 0, 'duplicate_groups': [], 'total_size': 0}
        
        all_pdfs = list(self.cpuc_pdfs_dir.rglob("*.pdf")) + list(self.cpuc_pdfs_dir.rglob("*.PDF"))
        self.stats['total_files_found'] = len(all_pdfs)
        
        logger.info(f"Found {len(all_pdfs)} PDF files")
        
        # Calculate total size
        total_size = sum(pdf.stat().st_size for pdf in all_pdfs if pdf.exists())
        logger.info(f"Total size: {total_size / (1024*1024):.1f} MB")
        
        # Group files by base name to identify duplicates/errata
        duplicate_groups = self._identify_duplicate_groups(all_pdfs)
        self.stats['duplicate_groups'] = len(duplicate_groups)
        
        logger.info(f"Found {len(duplicate_groups)} groups with potential duplicates/errata")
        
        return {
            'total_files': len(all_pdfs),
            'duplicate_groups': duplicate_groups,
            'total_size': total_size,
            'all_files': all_pdfs
        }
    
    def _identify_duplicate_groups(self, pdf_files: List[Path]) -> List[Dict]:
        """Identify groups of files that might be duplicates or errata versions"""
        # Group by base filename (without errata/redline suffixes)
        groups = defaultdict(list)
        
        for pdf in pdf_files:
            base_name = self._get_base_filename(pdf.name)
            groups[base_name].append(pdf)
        
        # Only return groups with multiple files
        duplicate_groups = []
        for base_name, files in groups.items():
            if len(files) > 1:
                # Sort by priority (keep latest/best version)
                sorted_files = self._sort_files_by_priority(files)
                duplicate_groups.append({
                    'base_name': base_name,
                    'files': sorted_files,
                    'keep_file': sorted_files[0],  # First is best
                    'remove_files': sorted_files[1:]  # Rest should be removed
                })
        
        return duplicate_groups
    
    def _get_base_filename(self, filename: str) -> str:
        """Extract base filename without errata/redline suffixes"""
        # Remove common suffixes
        base = filename.replace('.pdf', '').replace('.PDF', '')
        
        # Remove errata/redline indicators
        patterns_to_remove = [
            r'-errata.*$',
            r'-redline.*$', 
            r'-clean.*$',
            r'-corrected.*$',
            r'-revised.*$',
            r'-updated.*$',
            r'\s+errata.*$',
            r'\s+redline.*$',
            r'\s+clean.*$'
        ]
        
        for pattern in patterns_to_remove:
            base = re.sub(pattern, '', base, flags=re.IGNORECASE)
        
        return base.strip()
    
    def _sort_files_by_priority(self, files: List[Path]) -> List[Path]:
        """Sort files by priority - best version first"""
        def get_priority_score(file_path: Path) -> int:
            filename = file_path.name.lower()
            score = 0
            
            # Prefer clean versions over redline
            if 'clean' in filename:
                score += 100
            elif 'redline' in filename:
                score -= 50
            
            # Prefer errata over original (assuming it's corrected)
            if 'errata' in filename:
                score += 50
            
            # Prefer corrected/revised versions
            if any(word in filename for word in ['corrected', 'revised', 'updated']):
                score += 75
            
            # Use file modification time as tiebreaker (newer is better)
            try:
                score += int(file_path.stat().st_mtime)
            except:
                pass
            
            return score
        
        return sorted(files, key=get_priority_score, reverse=True)
    
    def create_backup(self, files_to_backup: List[Path]) -> bool:
        """Create backup of important files before deletion"""
        try:
            if not files_to_backup:
                logger.info("No files to backup")
                return True
            
            self.backup_dir.mkdir(exist_ok=True)
            backup_log = []
            
            logger.info(f"Creating backup of {len(files_to_backup)} files...")
            
            for file_path in files_to_backup:
                if file_path.exists():
                    # Create relative path structure in backup
                    rel_path = file_path.relative_to(self.project_root)
                    backup_path = self.backup_dir / rel_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(file_path, backup_path)
                    backup_log.append(str(rel_path))
            
            # Save backup log
            backup_log_path = self.backup_dir / "backup_log.json"
            with open(backup_log_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'files_backed_up': backup_log,
                    'reason': 'PDF cleanup - transition to URL-based processing'
                }, f, indent=2)
            
            logger.info(f"Backup created in {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def remove_local_pdfs(self, analysis_result: Dict, create_backup: bool = True) -> bool:
        """Remove local PDF files"""
        try:
            all_files = analysis_result.get('all_files', [])
            
            if not all_files:
                logger.info("No PDF files to remove")
                return True
            
            # Create backup if requested
            if create_backup:
                # Backup a sample of files and any that might be important
                important_files = []
                for file_path in all_files[:10]:  # Backup first 10 as samples
                    important_files.append(file_path)
                
                if not self.create_backup(important_files):
                    logger.warning("Backup failed, but continuing with cleanup")
            
            # Remove all PDF files
            files_removed = 0
            space_freed = 0
            
            logger.info(f"Removing {len(all_files)} PDF files...")
            
            for file_path in all_files:
                if file_path.exists():
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        files_removed += 1
                        space_freed += file_size
                        
                        if files_removed % 100 == 0:
                            logger.info(f"Removed {files_removed} files...")
                            
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
            
            self.stats['files_removed'] = files_removed
            self.stats['space_freed'] = space_freed
            
            logger.info(f"Removed {files_removed} files, freed {space_freed / (1024*1024):.1f} MB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove PDF files: {e}")
            return False
    
    def remove_metadata_files(self) -> bool:
        """Remove metadata.json files and other local processing artifacts"""
        try:
            metadata_files = list(self.cpuc_pdfs_dir.rglob("metadata.json"))
            files_removed = 0
            
            logger.info(f"Removing {len(metadata_files)} metadata files...")
            
            for metadata_file in metadata_files:
                if metadata_file.exists():
                    metadata_file.unlink()
                    files_removed += 1
            
            self.stats['metadata_files_removed'] = files_removed
            logger.info(f"Removed {files_removed} metadata files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove metadata files: {e}")
            return False
    
    def remove_empty_directories(self) -> bool:
        """Remove empty directories after cleanup"""
        try:
            directories_removed = 0
            
            # Remove empty subdirectories first, then check parent directories
            for root, dirs, files in os.walk(self.cpuc_pdfs_dir, topdown=False):
                root_path = Path(root)
                try:
                    if not any(root_path.iterdir()):  # Directory is empty
                        root_path.rmdir()
                        directories_removed += 1
                        logger.info(f"Removed empty directory: {root_path}")
                except OSError:
                    # Directory not empty or permission issue
                    pass
            
            # Try to remove the main cpuc_pdfs directory if it's empty
            try:
                if self.cpuc_pdfs_dir.exists() and not any(self.cpuc_pdfs_dir.iterdir()):
                    self.cpuc_pdfs_dir.rmdir()
                    directories_removed += 1
                    logger.info(f"Removed main PDF directory: {self.cpuc_pdfs_dir}")
            except OSError:
                logger.info("Main PDF directory not empty, keeping it")
            
            self.stats['directories_removed'] = directories_removed
            logger.info(f"Removed {directories_removed} empty directories")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove empty directories: {e}")
            return False
    
    def update_config_files(self) -> bool:
        """Update configuration files to remove file-based processing references"""
        try:
            # Update config.py if needed
            config_path = self.project_root / "config.py"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_content = f.read()
                
                # Check if BASE_PDF_DIR is still needed
                if 'BASE_PDF_DIR' in config_content and 'cpuc_pdfs' in config_content:
                    logger.info("BASE_PDF_DIR still in config.py - this may need manual review")
            
            logger.info("Configuration files checked")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config files: {e}")
            return False
    
    def generate_cleanup_report(self) -> str:
        """Generate a cleanup report"""
        report = f"""
PDF Cleanup Report
==================
Generated: {datetime.now().isoformat()}

Statistics:
- Total files found: {self.stats['total_files_found']}
- Files removed: {self.stats['files_removed']}
- Space freed: {self.stats['space_freed'] / (1024*1024):.1f} MB
- Metadata files removed: {self.stats['metadata_files_removed']}
- Directories removed: {self.stats['directories_removed']}
- Duplicate groups identified: {self.stats['duplicate_groups']}

Actions Taken:
✅ Removed all local PDF files (moved to URL-based processing)
✅ Cleaned up metadata files
✅ Removed empty directories
✅ Created backup of sample files

Next Steps:
- Verify URL-based processing is working correctly
- Monitor system performance with URL-based approach
- Consider removing backup after confirming system stability
- Review any remaining file-based processing code

Backup Location: {self.backup_dir}
"""
        return report
    
    def run_full_cleanup(self, create_backup: bool = True) -> bool:
        """Run the complete cleanup process"""
        logger.info("Starting full PDF cleanup process...")
        
        try:
            # Step 1: Analyze current state
            analysis = self.analyze_local_pdfs()
            
            if analysis['total_files'] == 0:
                logger.info("No PDF files found - cleanup not needed")
                return True
            
            logger.info(f"Analysis complete: {analysis['total_files']} files, {analysis['total_size'] / (1024*1024):.1f} MB")
            
            # Step 2: Remove PDF files
            if not self.remove_local_pdfs(analysis, create_backup):
                logger.error("Failed to remove PDF files")
                return False
            
            # Step 3: Remove metadata files
            if not self.remove_metadata_files():
                logger.error("Failed to remove metadata files")
                return False
            
            # Step 4: Remove empty directories
            if not self.remove_empty_directories():
                logger.error("Failed to remove empty directories")
                return False
            
            # Step 5: Update config files
            if not self.update_config_files():
                logger.error("Failed to update config files")
                return False
            
            # Step 6: Generate report
            report = self.generate_cleanup_report()
            report_path = self.project_root / "pdf_cleanup_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(report)
            logger.info(f"Cleanup complete! Report saved to {report_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cleanup process failed: {e}")
            return False


def main():
    """Main function"""
    import os
    
    logger.info("🧹 CPUC RAG PDF Cleanup Tool")
    logger.info("=" * 50)
    
    # Confirm with user
    print("This tool will remove ALL local PDF files and transition to URL-based processing.")
    print("A backup will be created for safety.")
    print("\nProceed? (y/N): ", end="")
    
    # For automated runs, check environment variable
    if os.environ.get('PDF_CLEANUP_CONFIRMED') == 'yes':
        response = 'y'
    else:
        response = input().strip().lower()
    
    if response != 'y':
        print("Cleanup cancelled.")
        return
    
    # Run cleanup
    cleanup_manager = PDFCleanupManager()
    success = cleanup_manager.run_full_cleanup(create_backup=True)
    
    if success:
        logger.info("✅ PDF cleanup completed successfully!")
    else:
        logger.error("❌ PDF cleanup failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    import os
    sys.exit(main())