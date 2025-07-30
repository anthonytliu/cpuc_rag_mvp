#!/usr/bin/env python3
"""
Test script to check vector store population status for any given proceeding.
This script diagnoses directory structure issues and provides clear status reports.
"""

import sys
import json
import lancedb
from pathlib import Path
from typing import Dict, Optional

def check_vector_store_status(proceeding: str) -> Dict:
    """
    Check the vector store status for a given proceeding.
    
    Args:
        proceeding: Proceeding number (e.g., R2207005)
        
    Returns:
        Dictionary with comprehensive status information
    """
    project_root = Path(__file__).parent
    results = {
        'proceeding': proceeding,
        'directories_found': [],
        'vector_stores': [],
        'document_hashes': {},
        'recommendations': []
    }
    
    # Check for various directory patterns that might exist
    possible_paths = [
        project_root / "local_lance_db" / proceeding,
        project_root / "local_lance_db" / "local_lance_db" / proceeding,
        project_root / f"cpuc_proceedings/{proceeding}/local_lance_db",
        project_root / f"cpuc_proceedings/{proceeding}/vector_db"
    ]
    
    print(f"🔍 Checking vector store status for proceeding: {proceeding}")
    print("=" * 60)
    
    # Check each possible directory
    for db_path in possible_paths:
        if db_path.exists():
            results['directories_found'].append(str(db_path))
            print(f"📁 Found directory: {db_path}")
            
            # Check for LanceDB data
            if "lance" in str(db_path):
                lance_status = check_lancedb_status(db_path, proceeding)
                if lance_status:
                    results['vector_stores'].append(lance_status)
    
    # Check for document hashes
    hash_file_paths = [
        project_root / f"cpuc_proceedings/{proceeding}/document_hashes.json",
        project_root / "local_lance_db" / proceeding / "document_hashes.json",
        project_root / "local_lance_db" / "local_lance_db" / proceeding / "document_hashes.json"
    ]
    
    for hash_path in hash_file_paths:
        if hash_path.exists():
            try:
                with open(hash_path, 'r') as f:
                    hashes = json.load(f)
                    results['document_hashes'][str(hash_path)] = {
                        'count': len(hashes),
                        'sample_keys': list(hashes.keys())[:3]
                    }
                    print(f"📋 Found document hashes: {hash_path} ({len(hashes)} entries)")
            except Exception as e:
                print(f"❌ Error reading hash file {hash_path}: {e}")
    
    # Generate recommendations
    results['recommendations'] = generate_recommendations(results)
    
    # Print summary
    print("\n📊 SUMMARY:")
    print(f"Directories found: {len(results['directories_found'])}")
    print(f"Vector stores found: {len(results['vector_stores'])}")
    print(f"Document hash files: {len(results['document_hashes'])}")
    
    # Print recommendations
    if results['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
    
    return results

def check_lancedb_status(db_path: Path, proceeding: str) -> Optional[Dict]:
    """Check LanceDB status at given path."""
    try:
        db = lancedb.connect(str(db_path))
        tables = db.table_names()
        table_name = f"{proceeding}_documents"
        
        status = {
            'type': 'LanceDB',
            'path': str(db_path),
            'tables': tables,
            'target_table': table_name,
            'has_target_table': table_name in tables,
            'row_count': 0
        }
        
        if table_name in tables:
            try:
                table = db.open_table(table_name)
                status['row_count'] = len(table.to_pandas())
                print(f"✅ LanceDB at {db_path}: {status['row_count']} rows in {table_name}")
            except Exception as e:
                print(f"⚠️ LanceDB table {table_name} exists but error reading: {e}")
                status['error'] = str(e)
        else:
            print(f"❌ LanceDB at {db_path}: No table {table_name} found")
            
        return status
        
    except Exception as e:
        print(f"❌ Error checking LanceDB at {db_path}: {e}")
        return None


def generate_recommendations(results: Dict) -> list:
    """Generate actionable recommendations based on findings."""
    recommendations = []
    
    # Check for directory path issues
    lance_dirs = [d for d in results['directories_found'] if 'lance' in d]
    if len(lance_dirs) > 1:
        recommendations.append(
            "Multiple LanceDB directories found. This indicates a directory path issue. "
            "Check rag_core.py _load_existing_lance_vector_store() method."
        )
    
    # Check for data in wrong location
    working_stores = [vs for vs in results['vector_stores'] if vs.get('row_count', 0) > 0]
    if not working_stores and results['vector_stores']:
        recommendations.append(
            "Vector store directories exist but contain no data. "
            "Run standalone_data_processor.py to populate the vector store."
        )
    elif len(working_stores) > 1:
        recommendations.append(
            "Multiple populated vector stores found. Consolidate to avoid confusion."
        )
    
    # Check for hash/data mismatch
    if results['document_hashes'] and not working_stores:
        recommendations.append(
            "Document hashes exist but no populated vector store found. "
            "There may be a path mismatch between hash storage and vector store location."
        )
    
    # Check for no data at all
    if not results['document_hashes'] and not working_stores:
        recommendations.append(
            "No document hashes or vector store data found. "
            "Run standalone_scraper.py first, then standalone_data_processor.py."
        )
    
    return recommendations

def main():
    """Main function to run the test."""
    proceeding = sys.argv[1] if len(sys.argv) > 1 else "R2207005"
    
    print(f"🧪 Vector Store Status Test")
    print(f"Proceeding: {proceeding}")
    print(f"Time: {Path(__file__).stat().st_mtime}")
    print()
    
    results = check_vector_store_status(proceeding)
    
    # Write results to file for debugging
    output_file = Path(__file__).parent / f"vector_store_status_{proceeding}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Full results saved to: {output_file}")
    
    # Exit with appropriate code
    working_stores = [vs for vs in results['vector_stores'] if vs.get('row_count', 0) > 0]
    if working_stores:
        print(f"\n✅ Vector store is populated and ready for {proceeding}")
        return 0
    else:
        print(f"\n❌ Vector store is not ready for {proceeding}")
        return 1

if __name__ == "__main__":
    exit(main())