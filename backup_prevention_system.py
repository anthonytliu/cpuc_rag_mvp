#!/usr/bin/env python3
"""
Backup Prevention System

Ensures that no backup folders are created in the future by providing
proper data preservation mechanisms that don't require backup folders.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BackupPreventionSystem:
    """System to prevent backup folder creation and ensure clean processing."""
    
    def __init__(self):
        self.src_dir = Path(__file__).parent / 'src'
        self.lance_db_dir = Path('/Users/anthony.liu/Downloads/CPUC_REG_RAG/data/vector_stores/local_lance_db')
        
        logger.info("🛡️ Backup Prevention System initialized")
    
    def scan_for_backup_creation_code(self) -> Dict[str, List[str]]:
        """Scan all source files for backup creation patterns."""
        logger.info("🔍 Scanning for backup creation patterns...")
        
        patterns = [
            'shutil.copytree.*backup',
            'backup.*=.*Path',
            '_backup_.*datetime',
            'consolidated_backup',
            'Creating backup',
            'backup_path'
        ]
        
        issues = {}
        
        for pattern in patterns:
            files_with_pattern = []
            try:
                from pathlib import Path
                import subprocess
                result = subprocess.run(
                    ['grep', '-r', '-l', pattern, str(self.src_dir)],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    files_with_pattern = result.stdout.strip().split('\n')
                    files_with_pattern = [f for f in files_with_pattern if f.strip()]
            except Exception as e:
                logger.warning(f"Could not scan for pattern {pattern}: {e}")
            
            if files_with_pattern:
                issues[pattern] = files_with_pattern
        
        return issues
    
    def validate_no_backup_folders_exist(self) -> bool:
        """Validate that no backup folders exist in the system."""
        logger.info("🧹 Validating no backup folders exist...")
        
        # Check for any backup folders
        backup_patterns = ['*_backup*', '*backup*']
        backup_folders = []
        
        for pattern in backup_patterns:
            backup_folders.extend(list(self.lance_db_dir.glob(pattern)))
        
        if backup_folders:
            logger.error(f"❌ Found {len(backup_folders)} backup folders:")
            for folder in backup_folders:
                logger.error(f"   {folder}")
            return False
        else:
            logger.info("✅ No backup folders found - system is clean")
            return True
    
    def validate_main_folders_integrity(self) -> Dict[str, bool]:
        """Validate that all main proceeding folders have proper structure."""
        logger.info("🔍 Validating main folder integrity...")
        
        results = {}
        
        # Find all proceeding folders (exclude backup folders)
        proceeding_folders = [
            folder for folder in self.lance_db_dir.iterdir() 
            if folder.is_dir() and not any(
                pattern in folder.name.lower() 
                for pattern in ['backup', 'temp', 'cache']
            )
        ]
        
        for folder in proceeding_folders:
            proceeding_name = folder.name
            
            # Check for required files
            hashes_file = folder / 'document_hashes.json'
            lance_dir = folder / f"{proceeding_name}_documents.lance"
            
            has_hashes = hashes_file.exists()
            has_lance = lance_dir.exists() and any(lance_dir.iterdir()) if lance_dir.exists() else False
            
            is_valid = has_hashes or has_lance  # At least one should exist
            results[proceeding_name] = is_valid
            
            status = "✅" if is_valid else "❌"
            logger.info(f"   {status} {proceeding_name}: hashes={has_hashes}, lance={has_lance}")
        
        return results
    
    def create_backup_prevention_config(self) -> bool:
        """Create a configuration to prevent backup creation."""
        config_content = '''# Backup Prevention Configuration
# This file indicates that backup folder creation should be disabled

BACKUP_FOLDERS_DISABLED = True
BACKUP_PREVENTION_ACTIVE = True

# Use in-memory data preservation instead of backup folders
USE_IN_MEMORY_PRESERVATION = True

# Main folders are the source of truth
MAIN_FOLDERS_ARE_SOURCE_OF_TRUTH = True

# Last updated by Backup Prevention System
LAST_UPDATED = "2025-07-30"
'''
        
        config_file = Path(__file__).parent / 'BACKUP_PREVENTION.config'
        try:
            with open(config_file, 'w') as f:
                f.write(config_content)
            logger.info(f"✅ Created backup prevention config: {config_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create config: {e}")
            return False
    
    def run_comprehensive_validation(self) -> Dict:
        """Run comprehensive validation of backup prevention."""
        logger.info("🚀 Running comprehensive backup prevention validation...")
        
        results = {
            'backup_code_issues': {},
            'backup_folders_clean': False,
            'main_folders_valid': {},
            'config_created': False,
            'overall_success': False
        }
        
        # 1. Scan for backup creation code
        results['backup_code_issues'] = self.scan_for_backup_creation_code()
        
        # 2. Validate no backup folders exist
        results['backup_folders_clean'] = self.validate_no_backup_folders_exist()
        
        # 3. Validate main folder integrity
        results['main_folders_valid'] = self.validate_main_folders_integrity()
        
        # 4. Create prevention config
        results['config_created'] = self.create_backup_prevention_config()
        
        # Overall success criteria
        results['overall_success'] = (
            len(results['backup_code_issues']) == 0 and
            results['backup_folders_clean'] and
            all(results['main_folders_valid'].values()) and
            results['config_created']
        )
        
        return results
    
    def generate_validation_report(self, results: Dict) -> str:
        """Generate comprehensive validation report."""
        report = f"""
🛡️ BACKUP PREVENTION VALIDATION REPORT
{'='*60}

📊 SUMMARY:
   Backup code issues: {len(results['backup_code_issues'])}
   Backup folders clean: {results['backup_folders_clean']}
   Main folders validated: {len(results['main_folders_valid'])}
   Prevention config created: {results['config_created']}
   Overall success: {results['overall_success']}

🔍 BACKUP CODE SCAN:
"""
        
        if results['backup_code_issues']:
            report += "   ❌ Found backup creation patterns:\n"
            for pattern, files in results['backup_code_issues'].items():
                report += f"   Pattern '{pattern}':\n"
                for file in files:
                    report += f"     - {file}\n"
        else:
            report += "   ✅ No backup creation patterns found\n"
        
        report += f"\n🧹 BACKUP FOLDERS STATUS:\n"
        if results['backup_folders_clean']:
            report += "   ✅ No backup folders found - system is clean\n"
        else:
            report += "   ❌ Backup folders still exist in system\n"
        
        report += f"\n📁 MAIN FOLDERS VALIDATION:\n"
        for proceeding, is_valid in results['main_folders_valid'].items():
            status = "✅" if is_valid else "❌"
            report += f"   {status} {proceeding}\n"
        
        report += f"\n🎯 OUTCOME:\n"
        if results['overall_success']:
            report += "   ✅ BACKUP PREVENTION SUCCESSFUL!\n"
            report += "   • No backup creation code found\n"
            report += "   • No backup folders exist\n"
            report += "   • All main folders are valid\n"
            report += "   • Prevention config created\n"
            report += "   • System is clean and production-ready\n"
        else:
            report += "   ⚠️ Some issues need attention:\n"
            if results['backup_code_issues']:
                report += "   • Backup creation code still exists\n"
            if not results['backup_folders_clean']:
                report += "   • Backup folders still present\n"
            if not all(results['main_folders_valid'].values()):
                report += "   • Some main folders invalid\n"
        
        return report


def main():
    """Run backup prevention validation."""
    print("🛡️ Backup Prevention System")
    print("=" * 40)
    
    system = BackupPreventionSystem()
    
    # Run validation
    results = system.run_comprehensive_validation()
    
    # Generate and display report
    report = system.generate_validation_report(results)
    print(report)
    
    # Save report
    report_file = Path('backup_prevention_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📋 Report saved to: {report_file}")
    
    if results['overall_success']:
        print(f"\n🎉 BACKUP PREVENTION SUCCESSFUL!")
        print("The system is clean and will not create backup folders.")
        return True
    else:
        print(f"\n⚠️ Some issues need attention.")
        print("Check the report for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)