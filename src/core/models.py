import logging
import os
import gc
import threading
import warnings
from typing import Optional, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Suppress MPS pin_memory warnings at module level
warnings.filterwarnings(
    "ignore", 
    message=".*pin_memory.*argument is set as true but not supported on MPS.*",
    category=UserWarning
)

# Try relative imports first, fall back to absolute
try:
    from . import config
except ImportError:
    from core import config

logger = logging.getLogger(__name__)

# Global model instances for memory management
_embedding_model_instance = None
_embedding_model_lock = threading.Lock()
_cuda_memory_manager = None


class CUDAMemoryManager:
    """Manages CUDA memory allocation and cleanup for optimal performance."""
    
    def __init__(self):
        self.torch = None
        self.device_info = None
        self._initialize_cuda()
    
    def _initialize_cuda(self):
        """Initialize CUDA environment and get device information."""
        try:
            import torch
            self.torch = torch
            
            if torch.cuda.is_available():
                self.device_info = {
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(),
                    'memory_allocated': torch.cuda.memory_allocated(),
                    'memory_reserved': torch.cuda.memory_reserved(),
                    'max_memory_allocated': torch.cuda.max_memory_allocated()
                }
                logger.info(f"CUDA initialized: {self.device_info['device_name']} (Device {self.device_info['current_device']})")
            else:
                logger.info("CUDA not available, using CPU/MPS")
                
        except ImportError:
            logger.warning("PyTorch not available for CUDA management")
    
    def optimize_memory(self):
        """Optimize CUDA memory usage."""
        if self.torch and self.torch.cuda.is_available():
            try:
                # Clear cache and run garbage collection
                self.torch.cuda.empty_cache()
                gc.collect()
                
                # Log memory status
                current_memory = self.torch.cuda.memory_allocated()
                reserved_memory = self.torch.cuda.memory_reserved()
                
                logger.debug(f"CUDA memory optimized - Allocated: {current_memory / 1024**2:.1f}MB, Reserved: {reserved_memory / 1024**2:.1f}MB")
                
            except Exception as e:
                logger.warning(f"CUDA memory optimization failed: {e}")
    
    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """Calculate optimal batch size based on available CUDA memory."""
        if not self.torch or not self.torch.cuda.is_available():
            return base_batch_size
        
        try:
            # Get available memory in GB
            available_memory = self.torch.cuda.get_device_properties(0).total_memory
            available_gb = available_memory / (1024**3)
            
            # Adjust batch size based on available memory
            if available_gb >= 12:  # High-end GPU
                optimal_batch_size = min(base_batch_size * 2, 64)
            elif available_gb >= 8:  # Mid-range GPU
                optimal_batch_size = base_batch_size
            else:  # Lower-end GPU
                optimal_batch_size = max(base_batch_size // 2, 8)
            
            logger.info(f"Optimal batch size calculated: {optimal_batch_size} (Available VRAM: {available_gb:.1f}GB)")
            return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return base_batch_size
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current CUDA memory statistics."""
        if not self.torch or not self.torch.cuda.is_available():
            return {'cuda_available': False}
        
        try:
            return {
                'cuda_available': True,
                'device_name': self.torch.cuda.get_device_name(),
                'total_memory': self.torch.cuda.get_device_properties(0).total_memory,
                'allocated_memory': self.torch.cuda.memory_allocated(),
                'reserved_memory': self.torch.cuda.memory_reserved(),
                'free_memory': self.torch.cuda.get_device_properties(0).total_memory - self.torch.cuda.memory_allocated()
            }
        except Exception as e:
            logger.error(f"Failed to get CUDA memory stats: {e}")
            return {'cuda_available': False, 'error': str(e)}


def get_cuda_memory_manager() -> CUDAMemoryManager:
    """Get global CUDA memory manager instance."""
    global _cuda_memory_manager
    if _cuda_memory_manager is None:
        _cuda_memory_manager = CUDAMemoryManager()
    return _cuda_memory_manager


def get_embedding_model(force_reload: bool = False):
    """
    Initialize and return the embedding model with CUDA optimization and memory management.
    
    This function creates a singleton HuggingFaceEmbeddings instance using the
    BAAI/bge-base-en-v1.5 model, optimized for the best available device
    (CUDA GPU, MPS for Apple Silicon, or CPU) with intelligent memory management.
    
    Args:
        force_reload: If True, force recreation of the model instance
    
    Returns:
        HuggingFaceEmbeddings: A configured embedding model instance
                              ready for generating vector representations
                              of text documents.
                              
    Note:
        The model automatically detects the best device and optimizes batch size
        based on available CUDA memory. Uses singleton pattern for memory efficiency.
    """
    global _embedding_model_instance
    
    with _embedding_model_lock:
        if _embedding_model_instance is None or force_reload:
            import torch
            
            # Get CUDA memory manager
            cuda_manager = get_cuda_memory_manager()
            
            # Auto-detect best device with enhanced logic
            if torch.cuda.is_available():
                device = "cuda"
                # Optimize CUDA memory before model loading
                cuda_manager.optimize_memory()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            
            # Get optimized batch size - Mac M4 Pro specific optimization
            base_batch_size = int(os.environ.get('EMBEDDING_BATCH_SIZE', '32'))
            
            # Mac M4 Pro has 48GB RAM and excellent memory bandwidth - use larger batches
            if device == "mps":
                # Optimize for M4 Pro: larger batches for better throughput
                import psutil
                memory_gb = psutil.virtual_memory().total / (1024**3)
                if memory_gb >= 32:  # M4 Pro with 48GB
                    optimal_batch_size = min(base_batch_size * 4, 128)
                    logger.info(f"M4 Pro detected with {memory_gb:.0f}GB RAM - using optimized batch size: {optimal_batch_size}")
                else:
                    optimal_batch_size = base_batch_size * 2
            else:
                optimal_batch_size = cuda_manager.get_optimal_batch_size(base_batch_size)
            
            # Model configuration with device-specific optimizations
            model_kwargs = {"device": device}
            encode_kwargs = {"batch_size": optimal_batch_size}
            
            # Add device-specific optimizations
            if device == "cuda":
                model_kwargs.update({
                    "device_map": "auto",
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
                })
                encode_kwargs.update({
                    "convert_to_tensor": True,
                    "normalize_embeddings": True  # Better for similarity search
                })
            elif device == "mps":
                # MPS-specific optimizations for Mac M4 Pro
                model_kwargs.update({
                    "model_kwargs": {"torch_dtype": torch.float32},  # M4 optimized precision
                })
                encode_kwargs.update({
                    "convert_to_tensor": True,
                    "normalize_embeddings": True,
                    "convert_to_numpy": False,   # Keep tensors on MPS
                })
                
                # M4 Pro memory optimizations
                os.environ.update({
                    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.8',
                    'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.6',
                    'OMP_NUM_THREADS': '8',  # Use performance cores
                    'MKL_NUM_THREADS': '8',
                })
                
                logger.info(f"M4 Pro MPS optimizations applied - batch size: {optimal_batch_size}")
            else:
                # CPU optimizations
                encode_kwargs.update({
                    "convert_to_tensor": True,
                    "normalize_embeddings": True
                })
            
            try:
                _embedding_model_instance = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-base-en-v1.5",
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                
                logger.info(f"Embedding model initialized on {device} with batch size {optimal_batch_size}")
                
                # Log CUDA memory stats if available
                if device == "cuda":
                    memory_stats = cuda_manager.get_memory_stats()
                    if memory_stats.get('cuda_available'):
                        allocated_mb = memory_stats['allocated_memory'] / (1024**2)
                        total_mb = memory_stats['total_memory'] / (1024**2)
                        logger.info(f"CUDA memory after model load: {allocated_mb:.1f}MB / {total_mb:.1f}MB")
                
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                # Fallback to CPU if CUDA fails
                if device != "cpu":
                    logger.warning("Falling back to CPU for embedding model")
                    _embedding_model_instance = HuggingFaceEmbeddings(
                        model_name="BAAI/bge-base-en-v1.5",
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"batch_size": 16}  # Smaller batch for CPU
                    )
                else:
                    raise
    
    return _embedding_model_instance


def cleanup_embedding_model():
    """Clean up embedding model and free CUDA memory."""
    global _embedding_model_instance
    
    with _embedding_model_lock:
        if _embedding_model_instance is not None:
            try:
                # Clear the model instance
                _embedding_model_instance = None
                
                # Optimize CUDA memory
                cuda_manager = get_cuda_memory_manager()
                cuda_manager.optimize_memory()
                
                logger.info("Embedding model cleaned up and CUDA memory optimized")
                
            except Exception as e:
                logger.warning(f"Error during embedding model cleanup: {e}")


def get_llm():
    """
    Initializes and returns the OpenAI Language Model for text generation.
    
    This function creates and configures a ChatOpenAI instance using the model
    specified in the configuration. It performs validation of the API key and
    tests the connection to ensure the model is accessible.
    
    Returns:
        ChatOpenAI or None: A configured OpenAI language model instance if successful,
                           None if initialization fails due to missing API key or
                           connection errors.
                           
    Raises:
        Logs errors for missing API keys or connection failures.
        
    Configuration:
        - Model: Uses OPENAI_MODEL_NAME from config or environment
        - Temperature: Set to 0 for deterministic outputs
        - Max tokens: Capped at 4096 for controlled response length
        
    Examples:
        >>> llm_instance = get_llm()
        >>> if llm_instance:
        ...     response = llm_instance.invoke("What is CPUC?")
        ...     print(response.content)
    """
    # Get API key from environment variables
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        print("❌ OpenAI API key is not configured. Please set it in environment variables.")
        return None

    # Get model name from config or environment
    model_name = getattr(config, 'OPENAI_MODEL_NAME', os.environ.get('OPENAI_MODEL_NAME', 'gpt-4-turbo-preview'))

    try:
        llm_instance = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key,
            max_tokens=4096
        )
        # Test connection with a simple query
        llm_instance.invoke("Hi")
        logger.info(f"Successfully connected to OpenAI with model: {model_name}")
        return llm_instance
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI API: {e}")
        print(f"❌ Failed to initialize OpenAI model: {e}")
        return None