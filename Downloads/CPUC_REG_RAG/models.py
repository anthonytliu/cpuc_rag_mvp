# 📁 models.py
# Initializes and provides access to LLM and embedding models.

import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from config import EMBEDDING_MODEL_NAME, LLM_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

def get_embedding_model():
    """Initializes and returns the HuggingFace embedding model."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_llm():
    """Initializes and returns the Ollama LLM."""
    try:
        llm = Ollama(
            model=LLM_MODEL,
            temperature=0,
            base_url=OLLAMA_BASE_URL
        )
        # Test connection
        llm.invoke("Hi")
        logger.info(f"Successfully connected to Ollama with model: {LLM_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}. Please ensure Ollama is running.")
        logger.error(f"Error: {e}")
        return None

def setup_local_models(self):
    """Setup local models optimized for M4 MacBook"""
    logger.info("Setting up local models for M4 MacBook...")

    # EMBEDDINGS - Optimized for M4 MacBook
    self.embeddings = self.embedding_model

    # LLM - Local Language Model
    try:
        self.llm = Ollama(
            model=self.llm_model,
            temperature=0,
            base_url="http://localhost:11434"
        )
        logger.info(f"Using Ollama model: {self.llm_model}")
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        logger.info("Falling back to simple text processing")
        self.llm = None

    logger.info(f"Local embeddings model: {self.embedding_model}")