#!/usr/bin/env python3
# Verify that optimizations are properly configured

import logging
import os
import config
from data_processing import doc_converter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_optimizations():
    """Verify that all optimizations are properly configured"""
    
    logger.info("🔍 Verifying optimization configurations...")
    logger.info("="*50)
    
    # Check config settings
    logger.info("📋 Configuration Settings:")
    logger.info(f"  URL_PARALLEL_WORKERS: {getattr(config, 'URL_PARALLEL_WORKERS', 'NOT SET')}")
    logger.info(f"  VECTOR_STORE_BATCH_SIZE: {getattr(config, 'VECTOR_STORE_BATCH_SIZE', 'NOT SET')}")
    logger.info(f"  DOCLING_FAST_MODE: {getattr(config, 'DOCLING_FAST_MODE', 'NOT SET')}")
    logger.info(f"  DOCLING_THREADS: {getattr(config, 'DOCLING_THREADS', 'NOT SET')}")
    logger.info(f"  EMBEDDING_BATCH_SIZE: {getattr(config, 'EMBEDDING_BATCH_SIZE', 'NOT SET')}")
    
    # Check environment variables
    logger.info("\n🌍 Environment Variables:")
    logger.info(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
    
    # Verify Docling configuration
    logger.info("\n⚙️  Docling Configuration:")
    try:
        # Check if fast mode is enabled
        pipeline_options = doc_converter.format_options[config.InputFormat.PDF].pipeline_options
        table_mode = pipeline_options.table_structure_options.mode
        logger.info(f"  Table processing mode: {table_mode}")
        
        if hasattr(pipeline_options, 'page_boundary'):
            logger.info(f"  Page boundary: {pipeline_options.page_boundary}")
        else:
            logger.info("  Page boundary: Not set (no limit)")
            
    except Exception as e:
        logger.warning(f"  Could not verify Docling config: {e}")
    
    # Performance expectations
    logger.info("\n🎯 Expected Performance Improvements:")
    logger.info("  • Parallel processing: 2-3x for multiple URLs")
    logger.info("  • Fast table mode: 10-20% Docling speedup") 
    logger.info("  • Larger batches: 50% vector store speedup")
    logger.info("  • Overall target: 20-40% total improvement")
    
    # Recommendations
    logger.info("\n💡 Optimization Status:")
    
    checks = []
    
    # Check parallel workers
    if hasattr(config, 'URL_PARALLEL_WORKERS') and config.URL_PARALLEL_WORKERS > 1:
        checks.append("✅ Parallel URL processing enabled")
    else:
        checks.append("❌ Parallel URL processing not configured")
    
    # Check batch size
    if hasattr(config, 'VECTOR_STORE_BATCH_SIZE') and config.VECTOR_STORE_BATCH_SIZE > 64:
        checks.append("✅ Optimized vector store batch size")
    else:
        checks.append("❌ Vector store batch size not optimized")
        
    # Check fast mode
    if hasattr(config, 'DOCLING_FAST_MODE') and config.DOCLING_FAST_MODE:
        checks.append("✅ Docling fast mode enabled")
    else:
        checks.append("❌ Docling fast mode not enabled")
        
    # Check threading
    if os.environ.get('OMP_NUM_THREADS'):
        checks.append("✅ Docling threading configured")
    else:
        checks.append("❌ Docling threading not configured")
    
    for check in checks:
        logger.info(f"  {check}")
    
    success_count = sum(1 for check in checks if check.startswith("✅"))
    total_count = len(checks)
    
    logger.info(f"\n📊 Optimization Score: {success_count}/{total_count} ({success_count/total_count*100:.0f}%)")
    
    if success_count == total_count:
        logger.info("🎉 All optimizations are properly configured!")
        return True
    else:
        logger.warning("⚠️  Some optimizations may not be active")
        return False

if __name__ == "__main__":
    verify_optimizations()
    
    logger.info("\n📈 To test performance improvements:")
    logger.info("  python test_performance_optimization.py")
    logger.info("  python test_parallel_benefits.py")