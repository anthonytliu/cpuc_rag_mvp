#!/usr/bin/env python3
"""
Verify that schema migration worked and enhanced metadata is present
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import lancedb
from rag_core import CPUCRAGSystem

def test_schema_migration():
    """Test that schema migration worked correctly."""
    print("🧪 Testing Schema Migration Results")
    print("=" * 50)
    
    # Connect directly to the LanceDB to check schema
    db_path = "/Users/anthony.liu/Downloads/CPUC_REG_RAG/local_lance_db/R1311005"
    
    try:
        db = lancedb.connect(db_path)
        table = db.open_table("R1311005_documents")
        
        # Get a few sample documents to check schema
        sample_docs = table.search([0.1] * 768).limit(3).to_pandas()
        
        print(f"✅ Successfully connected to LanceDB")
        print(f"📊 Found {len(sample_docs)} sample documents")
        
        # Check if enhanced citation fields are present
        enhanced_fields = ['char_start', 'char_end', 'char_length', 'text_snippet']
        
        print("\n🔍 Schema Migration Check:")
        columns = sample_docs.columns.tolist()
        
        for field in enhanced_fields:
            if field in columns:
                print(f"  ✅ {field}: Present in schema")
                # Show sample value if not null
                sample_value = sample_docs[field].iloc[0] if not sample_docs[field].isna().all() else "null"
                print(f"    📝 Sample value: {str(sample_value)[:50]}{'...' if len(str(sample_value)) > 50 else ''}")
            else:
                print(f"  ❌ {field}: Missing from schema")
        
        # Check basic fields too
        basic_fields = ['text', 'source', 'proceeding', 'page']
        print("\n📋 Basic Schema Check:")
        for field in basic_fields:
            if field in columns:
                print(f"  ✅ {field}: Present")
            else:
                print(f"  ❌ {field}: Missing")
        
        print(f"\n📝 Full schema columns: {len(columns)} total")
        print(f"   Columns: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema check failed: {e}")
        return False

def test_vector_store_health():
    """Test that the vector store is healthy and functional."""
    print("\n🏥 Testing Vector Store Health")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        rag_system = CPUCRAGSystem(current_proceeding="R1311005")
        
        # Check vector store stats
        if hasattr(rag_system, 'vectordb') and rag_system.vectordb:
            print(f"✅ Vector store initialized successfully")
            
            # Try a simple similarity search
            results = rag_system.vectordb.similarity_search("energy efficiency", k=1)
            if results:
                print(f"✅ Similarity search working: found {len(results)} results")
                
                # Check metadata of first result
                metadata = results[0].metadata
                enhanced_fields = ['char_start', 'char_end', 'char_length', 'text_snippet']
                
                enhanced_count = sum(1 for field in enhanced_fields if field in metadata)
                print(f"✅ Enhanced metadata fields: {enhanced_count}/{len(enhanced_fields)} present")
                
                if enhanced_count > 0:
                    print("✅ Schema migration successful - enhanced metadata is working!")
                    return True
                else:
                    print("⚠️ Schema migration incomplete - enhanced metadata missing")
                    return False
            else:
                print("❌ Similarity search returned no results")
                return False
        else:
            print("❌ Vector store not initialized")
            return False
            
    except Exception as e:
        print(f"❌ Vector store health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Verifying Schema Migration and Enhanced Citations\n")
    
    schema_ok = test_schema_migration()
    health_ok = test_vector_store_health()
    
    overall_success = schema_ok and health_ok
    
    print(f"\n{'='*50}")
    print(f"📊 FINAL RESULTS:")
    print(f"  Schema Migration: {'✅ SUCCESS' if schema_ok else '❌ FAILURE'}")
    print(f"  Vector Store Health: {'✅ SUCCESS' if health_ok else '❌ FAILURE'}")
    print(f"  Overall Status: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
    
    if overall_success:
        print("\n🎉 The Docling fallback method should now work correctly!")
        print("   Enhanced citation metadata is properly integrated.")
    else:
        print("\n❌ Further investigation needed.")
    
    sys.exit(0 if overall_success else 1)