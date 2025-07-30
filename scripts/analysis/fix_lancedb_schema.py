#!/usr/bin/env python3
"""
LanceDB Schema Fix Script

This script fixes LanceDB schema compatibility issues by rebuilding
the vector database with the correct schema.

Usage:
    python fix_lancedb_schema.py [proceeding_id]
    
Example:
    python fix_lancedb_schema.py R1202009
"""

import sys
import shutil
import logging
from pathlib import Path
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fix_lancedb_schema(proceeding_id: str):
    """Fix LanceDB schema by rebuilding the database"""
    
    logger.info(f"🔧 Fixing LanceDB schema for proceeding: {proceeding_id}")
    
    # Get paths
    paths = config.get_proceeding_file_paths(proceeding_id)
    
    # Check if vector DB exists
    if not paths['vector_db'].exists():
        logger.info("✅ No existing vector DB found. Schema fix not needed.")
        return True
    
    # Show what will be deleted
    db_files = list(paths['vector_db'].rglob("*"))
    logger.info(f"📊 Found {len(db_files)} files in vector DB directory")
    
    # Ask for confirmation
    print(f"\n⚠️  WARNING: This will delete the existing LanceDB for {proceeding_id}")
    print(f"📁 Directory: {paths['vector_db']}")
    print(f"📄 Files to delete: {len(db_files)}")
    print("\n🔄 After deletion, you'll need to rebuild by running:")
    print(f"   python standalone_data_processor.py {proceeding_id}")
    
    response = input("\n❓ Do you want to continue? (y/N): ").strip().lower()
    
    if response != 'y':
        logger.info("❌ Schema fix cancelled by user")
        return False
    
    try:
        # Delete vector DB directory
        logger.info(f"🗑️  Deleting vector DB directory...")
        shutil.rmtree(paths['vector_db'])
        logger.info("✅ Vector DB directory deleted successfully")
        
        # Optionally clean document hashes too
        if paths['document_hashes'].exists():
            response = input("\n❓ Also delete document hashes? (y/N): ").strip().lower()
            if response == 'y':
                paths['document_hashes'].unlink()
                logger.info("✅ Document hashes deleted")
        
        print("\n🎉 Schema fix completed!")
        print(f"🚀 Now run: python standalone_data_processor.py {proceeding_id}")
        print("   This will rebuild the vector database with the correct schema.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix schema: {e}")
        return False

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage: python fix_lancedb_schema.py [proceeding_id]")
        print("Example: python fix_lancedb_schema.py R1202009")
        sys.exit(1)
    
    proceeding_id = sys.argv[1]
    
    print("🔧 LanceDB Schema Fix Tool")
    print("=" * 50)
    print(f"📋 Proceeding: {proceeding_id}")
    print("\nThis tool fixes PyArrow schema casting errors by rebuilding the LanceDB")
    print("vector database with the correct schema structure.")
    
    success = fix_lancedb_schema(proceeding_id)
    
    if success:
        print("\n✅ Schema fix completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Schema fix failed or was cancelled")
        sys.exit(1)

if __name__ == '__main__':
    main()