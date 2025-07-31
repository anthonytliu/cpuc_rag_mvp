#!/usr/bin/env python3
"""
MPS Pin Memory Warning Suppression

Suppresses the repetitive pin_memory warnings on Apple Silicon MPS devices.
This is a system-wide fix for the Docling processing warnings.
"""

import warnings
import logging
import sys
from pathlib import Path

def suppress_mps_pin_memory_warnings():
    """Suppress MPS pin_memory warnings that appear during Docling processing."""
    
    # Suppress the specific torch dataloader warning
    warnings.filterwarnings(
        "ignore", 
        message=".*pin_memory.*argument is set as true but not supported on MPS now.*",
        category=UserWarning,
        module="torch.utils.data.dataloader"
    )
    
    # Also suppress at the logging level
    logging.getLogger("torch.utils.data.dataloader").setLevel(logging.ERROR)
    
    print("🔇 MPS pin_memory warnings suppressed for current session")

def apply_system_wide_suppression():
    """Apply system-wide warning suppression for MPS pin_memory issues."""
    
    # Add src to path
    src_dir = Path(__file__).parent / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Suppress warnings
    suppress_mps_pin_memory_warnings()
    
    # Test the suppression
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS device detected - warnings suppressed")
            
            # Test that warnings are suppressed
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.warn(
                    "'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.",
                    UserWarning
                )
                
                if len(w) == 0:
                    print("✅ Warning suppression working correctly")
                else:
                    print(f"⚠️ {len(w)} warnings still showing")
        else:
            print("ℹ️ MPS not available - suppression applied anyway")
            
    except ImportError:
        print("⚠️ PyTorch not available for testing")

if __name__ == "__main__":
    apply_system_wide_suppression()