#!/usr/bin/env python3
"""
Debug script to examine exact metadata structure being generated
"""

import json
import logging
from pathlib import Path
import data_processing

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_metadata():
    """Debug metadata structure to find None values"""
    
    test_proceeding = 'R1202009'
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M213/K120/213120477.PDF"
    test_title = "213120477"
    
    logger.info(f"Debugging metadata for URL: {test_url}")
    
    try:
        chunks = data_processing.extract_and_chunk_with_docling_url(
            test_url, test_title, test_proceeding
        )
        
        if chunks:
            first_chunk = chunks[0]
            metadata = first_chunk.metadata
            
            print("\n🔍 METADATA ANALYSIS:")
            print("=" * 50)
            
            for key, value in metadata.items():
                value_type = type(value).__name__
                is_none = value is None
                is_empty_string = value == ""
                
                print(f"{key:25} = {repr(value):30} | Type: {value_type:10} | None: {is_none} | Empty: {is_empty_string}")
            
            print("\n📊 METADATA SUMMARY:")
            print(f"Total fields: {len(metadata)}")
            none_fields = [k for k, v in metadata.items() if v is None]
            empty_fields = [k for k, v in metadata.items() if v == ""]
            
            if none_fields:
                print(f"❌ Fields with None values: {none_fields}")
            else:
                print("✅ No None values found")
                
            if empty_fields:
                print(f"📝 Fields with empty strings: {empty_fields}")
            
            # Test PyArrow compatibility
            print("\n🧪 PYARROW COMPATIBILITY TEST:")
            try:
                import pyarrow as pa
                
                # Create a simple PyArrow table to test schema compatibility
                data_dict = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        data_dict[key] = [value]
                    else:
                        data_dict[key] = [str(value)]
                
                table = pa.table(data_dict)
                print("✅ PyArrow table creation successful")
                print(f"Schema: {table.schema}")
                
            except Exception as e:
                print(f"❌ PyArrow compatibility issue: {e}")
            
            return metadata
        else:
            logger.error("No chunks returned")
            return None
            
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        return None

if __name__ == '__main__':
    print("🐛 Debugging Metadata Structure")
    print("=" * 60)
    
    metadata = debug_metadata()
    
    if metadata:
        print("\n✅ Metadata debug completed successfully")
    else:
        print("\n❌ Metadata debug failed")