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
    # ### FIX: You can add device mapping for GPU/MPS acceleration if needed
    # model_kwargs = {'device': 'mps'} # for Apple Silicon
    # return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs)
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