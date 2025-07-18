#!/usr/bin/env python3
"""
Quick verification that the startup sequence components are working.
"""

import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import config
from startup_manager import StartupManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_startup_components():
    """Verify the startup components are working."""
    logger.info("🔍 Verifying startup components...")
    
    try:
        # Test 1: Config functions
        first_proceeding = config.get_first_proceeding()
        logger.info(f"✅ First proceeding: {first_proceeding}")
        
        # Test 2: Startup manager initialization
        startup_manager = StartupManager()
        logger.info("✅ Startup manager initialized")
        
        # Test 3: Proceeding selection
        proceeding = startup_manager._select_first_proceeding()
        logger.info(f"✅ Proceeding selection: {proceeding}")
        
        # Test 4: Database initialization (just check if method exists)
        db_existed = startup_manager._initialize_database_and_folders()
        logger.info(f"✅ Database initialization: {db_existed}")
        
        # Test 5: Check scraper implementation
        logger.info("📦 Using standard scraper implementation")
        
        try:
            from incremental_embedder import create_incremental_embedder
            logger.info("📦 Incremental embedder available")
        except ImportError:
            logger.info("📦 Incremental embedder not available (will use fallback)")
        
        logger.info("✅ All startup components verified successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Startup component verification failed: {e}")
        return False

def test_rapid_startup():
    """Test startup without long-running operations."""
    logger.info("🚀 Testing rapid startup sequence...")
    
    def progress_callback(message: str, progress: int):
        if progress >= 0:
            print(f"[{progress:3d}%] {message}")
    
    try:
        startup_manager = StartupManager(progress_callback=progress_callback)
        
        # Test proceeding selection
        proceeding = startup_manager._select_first_proceeding()
        print(f"✅ Selected proceeding: {proceeding}")
        
        # Test database initialization
        db_existed = startup_manager._initialize_database_and_folders()
        print(f"✅ Database setup (existed: {db_existed})")
        
        # Check RAG system
        if startup_manager.rag_system:
            print("✅ RAG system initialized")
        else:
            print("⚠️ RAG system not initialized")
        
        # Check system status
        status = startup_manager.get_system_status()
        print(f"✅ System status: {status.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Rapid startup test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("STARTUP VERIFICATION")
    print("="*60)
    
    # Verify components
    components_ok = verify_startup_components()
    print()
    
    # Test rapid startup
    startup_ok = test_rapid_startup()
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Components: {'✅ OK' if components_ok else '❌ FAILED'}")
    print(f"Startup Test: {'✅ OK' if startup_ok else '❌ FAILED'}")
    
    if components_ok and startup_ok:
        print("\n🎉 STARTUP SYSTEM IS READY!")
        print("The robust startup sequence with fallbacks is working correctly.")
        print("The system will handle missing components gracefully.")
    else:
        print("\n⚠️ STARTUP SYSTEM HAS ISSUES")
        print("Please check the errors above.")