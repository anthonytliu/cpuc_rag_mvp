#!/usr/bin/env python3
"""
Enhanced Progress Tracker for Massive PDF Processing

Provides real-time progress updates, ETA calculations, and stage tracking
for large document processing operations.

Features:
- Stage-by-stage progress tracking
- Real-time ETA calculations  
- Memory usage monitoring
- Cancellation capability
- Visual progress indicators

Author: Claude Code
"""

import time
import threading
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class ProcessingStage(Enum):
    """Stages of document processing."""
    INITIALIZING = "Initializing"
    DOWNLOADING = "Downloading PDF"
    EXTRACTING = "Extracting Text"
    CHUNKING = "Creating Chunks"
    EMBEDDING = "Generating Embeddings"
    STORING = "Storing to Database"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

@dataclass
class StageProgress:
    """Progress information for a processing stage."""
    stage: ProcessingStage
    current: int = 0
    total: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    message: str = ""
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total == 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100)
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time for this stage."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Calculate estimated time to completion."""
        if self.current == 0 or self.total == 0 or self.start_time is None:
            return None
        
        elapsed = self.elapsed_time
        rate = self.current / elapsed if elapsed > 0 else 0
        
        if rate > 0:
            remaining_items = self.total - self.current
            return remaining_items / rate
        
        return None

class EnhancedProgressTracker:
    """Enhanced progress tracker for large document processing."""
    
    def __init__(self, document_title: str = "Document", estimated_size_mb: float = 0):
        """
        Initialize progress tracker.
        
        Args:
            document_title: Title of document being processed
            estimated_size_mb: Estimated size in MB for better ETA calculations
        """
        self.document_title = document_title
        self.estimated_size_mb = estimated_size_mb
        self.start_time = time.time()
        
        # Stage tracking
        self.current_stage = ProcessingStage.INITIALIZING
        self.stages: Dict[ProcessingStage, StageProgress] = {}
        
        # Progress callbacks
        self.progress_callbacks = []
        
        # Cancellation
        self.cancelled = False
        self.cancel_requested = False
        
        # Memory tracking
        self.process = psutil.Process(os.getpid())
        self.initial_memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        # Threading for periodic updates
        self.update_thread = None
        self.update_interval = 2.0  # seconds
        
        print(f"🚀 Starting processing: {document_title}")
        if estimated_size_mb > 0:
            print(f"📏 Estimated size: {estimated_size_mb:.1f} MB")
            
        self._initialize_stages()
    
    def _initialize_stages(self):
        """Initialize all processing stages."""
        stages_to_init = [
            ProcessingStage.DOWNLOADING,
            ProcessingStage.EXTRACTING, 
            ProcessingStage.CHUNKING,
            ProcessingStage.EMBEDDING,
            ProcessingStage.STORING
        ]
        
        for stage in stages_to_init:
            self.stages[stage] = StageProgress(stage=stage)
    
    def start_stage(self, stage: ProcessingStage, total_items: int = 0, message: str = ""):
        """Start a new processing stage."""
        if self.cancel_requested:
            self.cancelled = True
            return False
            
        self.current_stage = stage
        
        if stage not in self.stages:
            self.stages[stage] = StageProgress(stage=stage)
        
        stage_progress = self.stages[stage]
        stage_progress.start_time = time.time()
        stage_progress.total = total_items
        stage_progress.current = 0
        stage_progress.message = message
        
        self._print_stage_start(stage, message)
        
        # Start periodic updates if not already running
        if self.update_thread is None or not self.update_thread.is_alive():
            self._start_periodic_updates()
        
        return True
    
    def update_progress(self, current: int, message: str = ""):
        """Update progress for current stage."""
        if self.cancelled:
            return False
            
        if self.current_stage in self.stages:
            stage_progress = self.stages[self.current_stage]
            stage_progress.current = current
            if message:
                stage_progress.message = message
            
            # Call progress callbacks
            for callback in self.progress_callbacks:
                try:
                    callback(self.current_stage, stage_progress)
                except Exception:
                    pass  # Don't let callback errors break processing
        
        return not self.cancel_requested
    
    def complete_stage(self, stage: ProcessingStage, message: str = ""):
        """Mark a stage as completed."""
        if stage in self.stages:
            stage_progress = self.stages[stage]
            stage_progress.end_time = time.time()
            stage_progress.current = stage_progress.total
            if message:
                stage_progress.message = message
            
            elapsed = stage_progress.elapsed_time
            print(f"✅ {stage.value} completed in {elapsed:.1f}s - {message}")
    
    def fail_stage(self, stage: ProcessingStage, error_message: str):
        """Mark a stage as failed."""
        if stage in self.stages:
            stage_progress = self.stages[stage]
            stage_progress.end_time = time.time()
            stage_progress.message = f"ERROR: {error_message}"
            
        self.current_stage = ProcessingStage.FAILED
        print(f"❌ {stage.value} failed: {error_message}")
    
    def request_cancellation(self):
        """Request cancellation of processing."""
        self.cancel_requested = True
        print("🛑 Cancellation requested...")
    
    def add_progress_callback(self, callback: Callable[[ProcessingStage, StageProgress], None]):
        """Add a callback for progress updates."""
        self.progress_callbacks.append(callback)
    
    def _start_periodic_updates(self):
        """Start background thread for periodic progress updates."""
        def update_loop():
            while not self.cancelled and self.current_stage not in [
                ProcessingStage.COMPLETED, ProcessingStage.FAILED, ProcessingStage.CANCELLED
            ]:
                if self.cancel_requested:
                    self.cancelled = True
                    self.current_stage = ProcessingStage.CANCELLED
                    break
                    
                self._print_periodic_update()
                time.sleep(self.update_interval)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def _print_stage_start(self, stage: ProcessingStage, message: str):
        """Print stage start information."""
        print(f"\n🔄 {stage.value}")
        print("-" * 50)
        if message:
            print(f"📝 {message}")
    
    def _print_periodic_update(self):
        """Print periodic progress update."""
        if self.current_stage not in self.stages:
            return
            
        stage_progress = self.stages[self.current_stage]
        
        # Build progress line
        progress_line = f"⏳ {self.current_stage.value}: "
        
        if stage_progress.total > 0:
            progress_line += f"{stage_progress.current:,}/{stage_progress.total:,} "
            progress_line += f"({stage_progress.progress_percent:.1f}%) "
        
        # Add ETA if available
        eta = stage_progress.eta_seconds
        if eta is not None and eta > 5:  # Only show ETA if > 5 seconds
            eta_str = self._format_duration(eta)
            progress_line += f"ETA: {eta_str} "
        
        # Add elapsed time
        elapsed = stage_progress.elapsed_time
        elapsed_str = self._format_duration(elapsed)
        progress_line += f"Elapsed: {elapsed_str}"
        
        # Add memory usage
        current_memory_mb = self.process.memory_info().rss / 1024 / 1024
        memory_delta = current_memory_mb - self.initial_memory_mb
        progress_line += f" Memory: {current_memory_mb:.0f}MB (+{memory_delta:.0f}MB)"
        
        print(f"\r{progress_line}", end="", flush=True)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress summary."""
        total_elapsed = time.time() - self.start_time
        
        # Calculate overall progress based on completed stages
        completed_stages = sum(1 for stage in self.stages.values() 
                             if stage.end_time is not None)
        total_stages = len(self.stages)
        overall_percent = (completed_stages / total_stages) * 100 if total_stages > 0 else 0
        
        return {
            "current_stage": self.current_stage.value,
            "overall_percent": overall_percent,
            "total_elapsed": total_elapsed,  
            "stages": {stage.stage.value: {
                "progress_percent": stage.progress_percent,
                "elapsed_time": stage.elapsed_time,
                "eta_seconds": stage.eta_seconds,
                "message": stage.message
            } for stage in self.stages.values()},
            "cancelled": self.cancelled,
            "memory_usage_mb": self.process.memory_info().rss / 1024 / 1024
        }
    
    def finish(self, success: bool = True, message: str = ""):
        """Finish processing."""
        total_elapsed = time.time() - self.start_time
        
        if success and not self.cancelled:
            self.current_stage = ProcessingStage.COMPLETED
            print(f"\n\n🎉 Processing completed successfully!")
        elif self.cancelled:
            self.current_stage = ProcessingStage.CANCELLED
            print(f"\n\n🛑 Processing was cancelled")
        else:
            self.current_stage = ProcessingStage.FAILED
            print(f"\n\n❌ Processing failed")
        
        if message:
            print(f"📝 {message}")
            
        print(f"⏱️  Total time: {self._format_duration(total_elapsed)}")
        
        final_memory_mb = self.process.memory_info().rss / 1024 / 1024
        memory_delta = final_memory_mb - self.initial_memory_mb
        print(f"💾 Memory usage: {final_memory_mb:.0f}MB (+{memory_delta:.0f}MB)")
        
        # Stop update thread
        if self.update_thread and self.update_thread.is_alive():
            # Thread will stop naturally when stage changes to completed/failed/cancelled
            pass

# Example usage demonstration
if __name__ == "__main__":
    # Simulate processing a large document
    tracker = EnhancedProgressTracker("Test Large PDF", estimated_size_mb=22.3)
    
    try:
        # Simulate download stage
        if tracker.start_stage(ProcessingStage.DOWNLOADING, total_items=100, 
                             message="Downloading 22.3MB PDF..."):
            for i in range(0, 101, 10):
                if not tracker.update_progress(i, f"Downloaded {i}%"):
                    break
                time.sleep(0.5)
            tracker.complete_stage(ProcessingStage.DOWNLOADING, "Download complete")
        
        # Simulate extraction stage  
        if not tracker.cancelled and tracker.start_stage(ProcessingStage.EXTRACTING, 
                                                        total_items=334, 
                                                        message="Extracting text from 334 pages..."):
            for i in range(0, 335, 25):
                if not tracker.update_progress(i, f"Processed page {i}"):
                    break
                time.sleep(0.3)
            tracker.complete_stage(ProcessingStage.EXTRACTING, "Text extraction complete")
        
        # Simulate chunking
        if not tracker.cancelled and tracker.start_stage(ProcessingStage.CHUNKING,
                                                        total_items=2500,
                                                        message="Creating text chunks..."):
            for i in range(0, 2501, 100):
                if not tracker.update_progress(i, f"Created {i} chunks"):
                    break
                time.sleep(0.2)
            tracker.complete_stage(ProcessingStage.CHUNKING, "Chunking complete")
        
        tracker.finish(success=True, message="All stages completed successfully")
        
    except KeyboardInterrupt:
        tracker.request_cancellation()
        tracker.finish(success=False, message="Processing interrupted by user")
    except Exception as e:
        tracker.fail_stage(tracker.current_stage, str(e))
        tracker.finish(success=False, message=f"Processing failed: {e}")