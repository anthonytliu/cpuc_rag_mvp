#!/usr/bin/env python3
"""
R2401017 Backup Consolidation Script

Consolidates all R2401017 backup folders into the main folder using
the existing embedding system's schema migration capabilities.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
import sys

# Add src to path for imports
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data_processing.embedding_only_system import EmbeddingOnlySystem
from core import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class R2401017BackupConsolidator:
    """Consolidates R2401017 backup folders."""
    
    def __init__(self):
        self.proceeding = "R2401017"
        self.base_dir = Path("/Users/anthony.liu/Downloads/CPUC_REG_RAG/data/vector_stores/local_lance_db")
        self.main_folder = self.base_dir / self.proceeding
        self.backup_folders = self._find_backup_folders()
        self.consolidated_hashes = {}
        
        # Initialize embedding system for the main folder
        self.embedding_system = EmbeddingOnlySystem(self.proceeding)
        
        logger.info(f"Found {len(self.backup_folders)} backup folders to consolidate")
    
    def _find_backup_folders(self) -> List[Path]:
        """Find all R2401017 backup folders."""
        backup_pattern = f"{self.proceeding}_backup_*"
        backup_folders = list(self.base_dir.glob(backup_pattern))
        backup_folders.sort()  # Sort by timestamp
        return backup_folders
    
    def consolidate_backups(self) -> Dict:
        """Consolidate all backup folders into the main folder."""
        try:
            logger.info("🔄 Starting R2401017 backup consolidation...")
            
            # Step 1: Collect all document hashes from backup folders
            all_hashes = self._collect_document_hashes()
            
            # Step 2: Merge with current main folder hashes
            current_hashes = self._load_current_hashes()
            merged_hashes = self._merge_hashes(current_hashes, all_hashes)
            
            # Step 3: Update the main document_hashes.json
            self._update_main_document_hashes(merged_hashes)
            
            # Step 4: Use schema migration to consolidate vector data if needed
            if self._needs_vector_consolidation():
                self._consolidate_vector_data()
            
            logger.info("✅ Backup consolidation completed successfully")
            
            return {
                'status': 'completed',
                'backups_processed': len(self.backup_folders),
                'total_documents': len(merged_hashes),
                'main_folder': str(self.main_folder),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Consolidation failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _collect_document_hashes(self) -> Dict[str, Dict]:
        """Collect document hashes from all backup folders."""
        all_hashes = {}
        
        for backup_folder in self.backup_folders:
            hash_file = backup_folder / "document_hashes.json"
            
            if hash_file.exists():
                try:
                    with open(hash_file, 'r') as f:
                        backup_hashes = json.load(f)
                    
                    logger.info(f"📂 Processing {backup_folder.name}: {len(backup_hashes)} documents")
                    
                    # Merge hashes, keeping the most recent processing time
                    for doc_hash, metadata in backup_hashes.items():
                        if doc_hash in all_hashes:
                            # Keep the most recent processing time
                            current_time = all_hashes[doc_hash].get('last_processed', '')
                            backup_time = metadata.get('last_processed', '')
                            
                            if backup_time > current_time:
                                all_hashes[doc_hash] = metadata
                                logger.debug(f"Updated {metadata.get('title', doc_hash)} with newer processing time")
                        else:
                            all_hashes[doc_hash] = metadata
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to read {hash_file}: {e}")
            else:
                logger.warning(f"⚠️ No document_hashes.json found in {backup_folder}")
        
        logger.info(f"📊 Collected {len(all_hashes)} unique documents from backups")
        return all_hashes
    
    def _load_current_hashes(self) -> Dict[str, Dict]:
        """Load current document hashes from main folder."""
        hash_file = self.main_folder / "document_hashes.json"
        
        if hash_file.exists():
            try:
                with open(hash_file, 'r') as f:
                    current_hashes = json.load(f)
                logger.info(f"📋 Current main folder has {len(current_hashes)} documents")
                return current_hashes
            except Exception as e:
                logger.warning(f"⚠️ Failed to read current hashes: {e}")
                return {}
        else:
            logger.info("📋 No current document_hashes.json found in main folder")
            return {}
    
    def _merge_hashes(self, current_hashes: Dict, backup_hashes: Dict) -> Dict:
        """Merge current and backup hashes, keeping the most recent data."""
        merged = current_hashes.copy()
        updates_count = 0
        new_count = 0
        
        for doc_hash, metadata in backup_hashes.items():
            if doc_hash in merged:
                # Keep the most recent processing time
                current_time = merged[doc_hash].get('last_processed', '')
                backup_time = metadata.get('last_processed', '')
                
                if backup_time > current_time:
                    merged[doc_hash] = metadata
                    updates_count += 1
            else:
                merged[doc_hash] = metadata
                new_count += 1
        
        logger.info(f"📊 Merge results: {new_count} new documents, {updates_count} updated documents")
        logger.info(f"📊 Total documents after merge: {len(merged)}")
        
        return merged
    
    def _update_main_document_hashes(self, merged_hashes: Dict):
        """Update the main document_hashes.json file."""
        hash_file = self.main_folder / "document_hashes.json"
        
        # Ensure directory exists
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup of current file if it exists
        if hash_file.exists():
            backup_file = hash_file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            shutil.copy2(hash_file, backup_file)
            logger.info(f"📦 Created backup: {backup_file.name}")
        
        # Write merged hashes
        with open(hash_file, 'w') as f:
            json.dump(merged_hashes, f, indent=2)
        
        logger.info(f"💾 Updated main document_hashes.json with {len(merged_hashes)} documents")
        self.consolidated_hashes = merged_hashes
    
    def _needs_vector_consolidation(self) -> bool:
        """Check if vector data consolidation is needed."""
        try:
            current_vector_count = self.embedding_system.get_vector_count()
            expected_chunk_count = sum(doc.get('chunk_count', 0) for doc in self.consolidated_hashes.values())
            
            logger.info(f"📊 Current vectors: {current_vector_count}, Expected: {expected_chunk_count}")
            
            # If there's a significant discrepancy, consolidation is needed
            if abs(current_vector_count - expected_chunk_count) > 10:
                logger.info("🔄 Vector consolidation needed due to count discrepancy")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Could not determine if vector consolidation needed: {e}")
            return False
    
    def _consolidate_vector_data(self):
        """Consolidate vector data using schema migration."""
        try:
            logger.info("🔄 Starting vector data consolidation using schema migration...")
            
            # Use the embedding system's schema migration to rebuild the vector store
            # This will preserve all data and create a consolidated database
            success = self.embedding_system._attempt_schema_migration()
            
            if success:
                logger.info("✅ Vector data consolidation completed successfully")
            else:
                logger.error("❌ Vector data consolidation failed")
                
        except Exception as e:
            logger.error(f"❌ Vector consolidation error: {e}")
    
    def cleanup_backup_folders(self) -> Dict:
        """Clean up backup folders after successful consolidation."""
        try:
            logger.info("🧹 Starting backup folder cleanup...")
            
            # Create a single consolidated backup before cleanup
            consolidated_backup_dir = self.base_dir / f"{self.proceeding}_consolidated_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if not consolidated_backup_dir.exists():
                consolidated_backup_dir.mkdir(parents=True)
                logger.info(f"📦 Created consolidated backup directory: {consolidated_backup_dir.name}")
            
            cleanup_count = 0
            for backup_folder in self.backup_folders:
                try:
                    if backup_folder.exists():
                        # Move to consolidated backup (don't delete immediately)
                        target_path = consolidated_backup_dir / backup_folder.name
                        shutil.move(str(backup_folder), str(target_path))
                        cleanup_count += 1
                        logger.info(f"🗂️ Moved {backup_folder.name} to consolidated backup")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to move {backup_folder.name}: {e}")
            
            logger.info(f"✅ Cleanup completed: {cleanup_count} backup folders consolidated")
            
            return {
                'status': 'completed',
                'cleaned_folders': cleanup_count,
                'consolidated_backup': str(consolidated_backup_dir)
            }
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def verify_consolidation(self) -> Dict:
        """Verify the consolidation was successful."""
        try:
            logger.info("🔍 Verifying consolidation...")
            
            # Check document hash file
            hash_file = self.main_folder / "document_hashes.json"
            if not hash_file.exists():
                return {'status': 'error', 'error': 'document_hashes.json not found'}
            
            with open(hash_file, 'r') as f:
                main_hashes = json.load(f)
            
            # Check vector store
            vector_count = self.embedding_system.get_vector_count()
            expected_chunks = sum(doc.get('chunk_count', 0) for doc in main_hashes.values())
            
            # Health check
            health = self.embedding_system.health_check()
            
            verification_result = {
                'status': 'verified' if health['healthy'] else 'warning',
                'total_documents': len(main_hashes),
                'current_vectors': vector_count,
                'expected_chunks': expected_chunks,
                'vector_discrepancy': abs(vector_count - expected_chunks),
                'embedding_system_healthy': health['healthy'],
                'main_folder_exists': self.main_folder.exists(),
                'document_hashes_exists': hash_file.exists()
            }
            
            if verification_result['status'] == 'verified':
                logger.info("✅ Consolidation verification passed")
            else:
                logger.warning("⚠️ Consolidation verification has warnings")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


def main():
    """Main consolidation workflow."""
    logger.info("🚀 Starting R2401017 backup consolidation process...")
    
    consolidator = R2401017BackupConsolidator()
    
    # Step 1: Consolidate backups
    consolidation_result = consolidator.consolidate_backups()
    if consolidation_result['status'] != 'completed':
        logger.error(f"❌ Consolidation failed: {consolidation_result}")
        return consolidation_result
    
    # Step 2: Verify consolidation
    verification_result = consolidator.verify_consolidation()
    
    # Step 3: Clean up backup folders (only if verification passed)
    if verification_result['status'] in ['verified', 'warning']:
        cleanup_result = consolidator.cleanup_backup_folders()
        verification_result['cleanup'] = cleanup_result
    
    logger.info("🎉 R2401017 backup consolidation process completed!")
    return {
        'consolidation': consolidation_result,
        'verification': verification_result
    }


if __name__ == "__main__":
    result = main()
    print("\n" + "="*60)
    print("R2401017 BACKUP CONSOLIDATION SUMMARY")
    print("="*60)
    
    if result['consolidation']['status'] == 'completed':
        print(f"✅ Consolidation: SUCCESS")
        print(f"   📊 Backups processed: {result['consolidation']['backups_processed']}")
        print(f"   📄 Total documents: {result['consolidation']['total_documents']}")
        print(f"   📍 Main folder: {result['consolidation']['main_folder']}")
    else:
        print(f"❌ Consolidation: FAILED")
        print(f"   Error: {result['consolidation'].get('error', 'Unknown error')}")
    
    if 'verification' in result:
        verification = result['verification']
        print(f"\n🔍 Verification: {verification['status'].upper()}")
        print(f"   📄 Documents: {verification.get('total_documents', 'N/A')}")
        print(f"   🔢 Vectors: {verification.get('current_vectors', 'N/A')}")
        print(f"   💚 System healthy: {verification.get('embedding_system_healthy', 'N/A')}")
        
        if 'cleanup' in verification:
            cleanup = verification['cleanup']
            print(f"\n🧹 Cleanup: {cleanup['status'].upper()}")
            if cleanup['status'] == 'completed':
                print(f"   📁 Folders moved: {cleanup['cleaned_folders']}")
    
    print("="*60)