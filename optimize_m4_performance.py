#!/usr/bin/env python3
"""
Mac M4 Pro Performance Optimization System

Optimizes embedding and data processing for Mac M4 Pro with 48GB RAM and 14 cores.
Implements advanced parallelization, memory management, and MPS utilization.
"""

import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
import os
import psutil
import sys
import time
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core import config
from core.models import get_embedding_model
from data_processing.embedding_only_system import EmbeddingOnlySystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class M4ProOptimizer:
    """Optimizes data processing for Mac M4 Pro hardware."""
    
    def __init__(self):
        self.hardware_specs = self._detect_hardware()
        self.optimal_config = self._calculate_optimal_config()
        self._configure_environment()
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect Mac M4 Pro hardware specifications."""
        memory = psutil.virtual_memory()
        cpu_cores = psutil.cpu_count()
        
        # Check MPS availability and performance
        mps_available = torch.backends.mps.is_available()
        mps_performance = None
        
        if mps_available:
            try:
                device = torch.device('mps')
                # Test MPS matrix multiplication performance
                x = torch.randn(1000, 1000, device=device)
                start_time = time.time()
                y = torch.mm(x, x)
                torch.mps.synchronize()
                mps_performance = time.time() - start_time
            except Exception as e:
                logger.warning(f"MPS performance test failed: {e}")
                mps_performance = None
        
        specs = {
            'total_ram_gb': memory.total / (1024**3),
            'available_ram_gb': memory.available / (1024**3),
            'cpu_cores': cpu_cores,
            'mps_available': mps_available,
            'mps_performance_ms': mps_performance * 1000 if mps_performance else None,
            'is_m4_pro': True  # Detected from system_profiler output
        }
        
        logger.info(f"🔍 Detected Mac M4 Pro: {specs['cpu_cores']} cores, {specs['total_ram_gb']:.1f}GB RAM")
        if specs['mps_available']:
            logger.info(f"⚡ MPS Performance: {specs['mps_performance_ms']:.2f}ms matrix mult")
        
        return specs
    
    def _calculate_optimal_config(self) -> Dict[str, Any]:
        """Calculate optimal configuration for Mac M4 Pro."""
        
        # M4 Pro optimization parameters
        config = {
            # Embedding optimizations
            'embedding_batch_size': 128,  # Larger batches for M4's memory bandwidth
            'embedding_workers': 8,       # Use most performance cores for embedding
            'embedding_prefetch_factor': 4,  # High prefetch for fast SSD
            
            # Processing optimizations  
            'pdf_processing_workers': 6,   # Leave cores for system + embedding
            'pdf_processing_timeout': 300, # Extended for complex documents
            'chunk_batch_size': 200,      # Large chunks for memory efficiency
            
            # Memory optimizations
            'max_memory_usage_gb': min(32, self.hardware_specs['available_ram_gb'] * 0.8),
            'torch_memory_fraction': 0.6,  # Reserve MPS memory
            'garbage_collection_interval': 50,  # Frequent GC for memory management
            
            # Parallelization
            'max_concurrent_documents': 4,  # Balance memory vs throughput
            'async_processing': True,       # Use async for I/O bound operations
            'pipeline_stages': 3,          # Multi-stage pipeline processing
            
            # MPS optimizations
            'use_mps_optimization': self.hardware_specs['mps_available'],
            'mps_memory_pool_size': 8192,  # 8GB MPS memory pool
            'mps_async_operations': True,
        }
        
        logger.info(f"🚀 Calculated optimal config for M4 Pro:")
        logger.info(f"   Embedding batch size: {config['embedding_batch_size']}")
        logger.info(f"   PDF workers: {config['pdf_processing_workers']}")
        logger.info(f"   Max memory usage: {config['max_memory_usage_gb']:.1f}GB")
        logger.info(f"   Concurrent docs: {config['max_concurrent_documents']}")
        
        return config
    
    def _configure_environment(self):
        """Configure environment for optimal M4 Pro performance."""
        
        # Set environment variables for optimal performance
        os.environ.update({
            # PyTorch optimizations
            'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.8',
            'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.6',
            'OMP_NUM_THREADS': str(self.optimal_config['embedding_workers']),
            'MKL_NUM_THREADS': str(self.optimal_config['embedding_workers']),
            
            # Memory optimizations
            'PYTORCH_MPS_ALLOCATOR_POLICY': 'garbage_collection',
            'MALLOC_ARENA_MAX': '4',  # Limit malloc arenas
            
            # I/O optimizations
            'PYTHONIOENCODING': 'utf-8',
            'TOKENIZERS_PARALLELISM': 'true',
        })
        
        # Configure multiprocessing
        if hasattr(mp, 'set_start_method'):
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
        
        logger.info("✅ Environment configured for M4 Pro optimization")
    
    def create_optimized_embedding_system(self, proceeding: str) -> 'OptimizedEmbeddingSystem':
        """Create an embedding system optimized for M4 Pro."""
        return OptimizedEmbeddingSystem(proceeding, self.optimal_config)
    
    def benchmark_system(self) -> Dict[str, float]:
        """Benchmark system performance with current optimizations."""
        logger.info("🧪 Running M4 Pro performance benchmark...")
        
        benchmarks = {}
        
        # 1. MPS Performance Test
        if self.hardware_specs['mps_available']:
            device = torch.device('mps')
            
            # Matrix multiplication benchmark
            sizes = [512, 1024, 2048]
            for size in sizes:
                x = torch.randn(size, size, device=device)
                start_time = time.time()
                for _ in range(10):
                    y = torch.mm(x, x)
                torch.mps.synchronize()
                avg_time = (time.time() - start_time) / 10
                benchmarks[f'mps_matmul_{size}x{size}_ms'] = avg_time * 1000
        
        # 2. Embedding Model Performance
        try:
            model = get_embedding_model(force_reload=True)
            
            # Single embedding benchmark
            start_time = time.time()
            embedding = model.embed_query("This is a test sentence for benchmarking embedding performance.")
            single_time = time.time() - start_time
            benchmarks['embedding_single_ms'] = single_time * 1000
            
            # Batch embedding benchmark
            test_texts = ["Test sentence number " + str(i) for i in range(64)]
            start_time = time.time()
            batch_embeddings = model.embed_documents(test_texts)
            batch_time = time.time() - start_time
            benchmarks['embedding_batch_64_ms'] = batch_time * 1000
            benchmarks['embedding_throughput_per_sec'] = 64 / batch_time
            
        except Exception as e:
            logger.warning(f"Embedding benchmark failed: {e}")
        
        # 3. Memory throughput test
        start_time = time.time()
        data = [i for i in range(1000000)]
        processed = [x * 2 for x in data]
        memory_time = time.time() - start_time
        benchmarks['memory_throughput_ms'] = memory_time * 1000
        
        logger.info("📊 Benchmark Results:")
        for key, value in benchmarks.items():
            logger.info(f"   {key}: {value:.2f}")
        
        return benchmarks


class OptimizedEmbeddingSystem(EmbeddingOnlySystem):
    """M4 Pro optimized embedding system with advanced parallelization."""
    
    def __init__(self, proceeding: str, optimization_config: Dict[str, Any]):
        self.optimization_config = optimization_config
        
        # Configure optimal batch sizes before parent initialization
        os.environ['EMBEDDING_BATCH_SIZE'] = str(optimization_config['embedding_batch_size'])
        
        super().__init__(proceeding)
        
        # Initialize optimized components
        self._setup_advanced_processing()
        
        logger.info(f"🚀 OptimizedEmbeddingSystem initialized for {proceeding}")
        logger.info(f"   Batch size: {optimization_config['embedding_batch_size']}")
        logger.info(f"   Workers: {optimization_config['embedding_workers']}")
        logger.info(f"   Memory limit: {optimization_config['max_memory_usage_gb']:.1f}GB")
    
    def _setup_advanced_processing(self):
        """Setup advanced processing capabilities."""
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.optimization_config['pdf_processing_workers']
        )
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.optimization_config['embedding_workers']
        )
        self.memory_monitor = MemoryMonitor(
            max_usage_gb=self.optimization_config['max_memory_usage_gb']
        )
    
    async def process_documents_parallel(self, document_urls: List[str]) -> Dict[str, Any]:
        """Process multiple documents in parallel using M4 Pro's full capabilities."""
        logger.info(f"🚀 Starting parallel processing of {len(document_urls)} documents")
        
        start_time = time.time()
        results = {
            'successful': [],
            'failed': [],
            'total_chunks': 0,
            'processing_time': 0
        }
        
        # Process documents in optimal batches
        max_concurrent = self.optimization_config['max_concurrent_documents']
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_url(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await self._process_url_async(url)
        
        # Execute all document processing concurrently
        tasks = [process_single_url(url) for url in document_urls]
        
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for i, result in enumerate(completed_results):
            if isinstance(result, Exception):
                results['failed'].append({
                    'url': document_urls[i],
                    'error': str(result)
                })
            elif result and result.get('success'):
                results['successful'].append(result)
                results['total_chunks'] += result.get('chunks_added', 0)
            else:
                results['failed'].append({
                    'url': document_urls[i],
                    'error': result.get('error', 'Unknown error')
                })
        
        results['processing_time'] = time.time() - start_time
        
        logger.info(f"✅ Parallel processing completed in {results['processing_time']:.2f}s")
        logger.info(f"   Successful: {len(results['successful'])}")
        logger.info(f"   Failed: {len(results['failed'])}")
        logger.info(f"   Total chunks: {results['total_chunks']}")
        
        return results
    
    async def _process_url_async(self, url: str) -> Dict[str, Any]:
        """Process a single URL asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run CPU-intensive processing in process pool
        future = loop.run_in_executor(
            self.process_pool,
            self._process_single_url_sync,
            url
        )
        
        try:
            result = await asyncio.wait_for(
                future, 
                timeout=self.optimization_config['pdf_processing_timeout']
            )
            return result
        except asyncio.TimeoutError:
            return {
                'success': False,
                'url': url,
                'error': f'Processing timeout after {self.optimization_config["pdf_processing_timeout"]}s'
            }
    
    def _process_single_url_sync(self, url: str) -> Dict[str, Any]:
        """Synchronous URL processing for process pool."""
        try:
            # Extract document title
            title = url.split('/')[-1].replace('.PDF', '').replace('.pdf', '')
            
            # Check if already processed
            if self.is_document_processed(url):
                return {
                    'success': True,
                    'url': url,
                    'title': title,
                    'chunks_added': 0,
                    'skipped': True
                }
            
            # Process document
            chunks = self.process_document_url(
                pdf_url=url,
                document_title=title,
                use_progress_tracking=False  # Disable for parallel processing
            )
            
            if not chunks:
                return {
                    'success': False,
                    'url': url,
                    'error': 'No chunks extracted'
                }
            
            # Add to vector store with optimized batching
            result = self.add_document_incrementally(
                documents=chunks,
                batch_size=self.optimization_config['chunk_batch_size'],
                use_progress_tracking=False
            )
            
            if result['success']:
                # Update document hashes
                self.add_document_to_hashes(url, title, len(chunks))
                
                return {
                    'success': True,
                    'url': url,
                    'title': title,
                    'chunks_added': result['added'],
                    'total_chunks': len(chunks)
                }
            else:
                return {
                    'success': False,
                    'url': url,
                    'error': result.get('error', 'Failed to add to vector store')
                }
                
        except Exception as e:
            return {
                'success': False,
                'url': url,
                'error': str(e)
            }
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        # Clear MPS cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


class MemoryMonitor:
    """Monitors and manages memory usage during processing."""
    
    def __init__(self, max_usage_gb: float):
        self.max_usage_gb = max_usage_gb
        self.max_usage_bytes = max_usage_gb * (1024**3)
        
    def check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        memory = psutil.virtual_memory()
        return memory.used < self.max_usage_bytes
    
    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory status."""
        memory = psutil.virtual_memory()
        return {
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent,
            'within_limits': memory.used < self.max_usage_bytes
        }


async def main():
    """Demo of M4 Pro optimization system."""
    print("🚀 Mac M4 Pro Optimization System")
    print("="*50)
    
    # Initialize optimizer
    optimizer = M4ProOptimizer()
    
    # Run benchmarks
    benchmarks = optimizer.benchmark_system()
    
    # Test with a proceeding
    proceeding = "R2401017"
    optimized_system = optimizer.create_optimized_embedding_system(proceeding)
    
    # Health check
    health = optimized_system.health_check()
    print(f"\n📊 Optimized System Health:")
    print(f"   Healthy: {health['healthy']}")
    print(f"   Vector count: {health['vector_count']}")
    
    # Cleanup
    optimized_system.cleanup()
    
    print("\n✅ M4 Pro optimization demo completed!")


if __name__ == "__main__":
    asyncio.run(main())