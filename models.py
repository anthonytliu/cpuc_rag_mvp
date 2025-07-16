import logging
import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

import config
from config import EMBEDDING_MODEL_NAME, OPENAI_MODEL_NAME, OPENAI_API_KEY

logger = logging.getLogger(__name__)


def get_embedding_model():
    """
    Initialize and return the embedding model for vector representations.
    
    This function creates a SentenceTransformerEmbeddings instance using the
    BAAI/bge-base-en-v1.5 model, which is optimized for English text embeddings.
    The model is configured to run on Metal Performance Shaders (MPS) for
    Apple Silicon devices.
    
    Returns:
        SentenceTransformerEmbeddings: A configured embedding model instance
                                      ready for generating vector representations
                                      of text documents.
                                      
    Note:
        The model uses MPS device acceleration on Apple Silicon Macs.
        This may need to be adjusted for other hardware configurations.
    """
    return SentenceTransformerEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "mps"},
        encode_kwargs={
            "batch_size": config.EMBEDDING_BATCH_SIZE if hasattr(config, 'EMBEDDING_BATCH_SIZE') else 32
        }
    )


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
        Displays Streamlit error messages for user feedback.
        
    Configuration:
        - Model: Uses OPENAI_MODEL_NAME from config
        - Temperature: Set to 0 for deterministic outputs
        - Max tokens: Capped at 4096 for controlled response length
        
    Examples:
        >>> llm = get_llm()
        >>> if llm:
        ...     response = llm.invoke("What is CPUC?")
        ...     print(response.content)
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        st.error("OpenAI API key is not configured. Please set it in your .env file.")
        return None

    try:
        llm = ChatOpenAI(
            model=config.OPENAI_MODEL_NAME,
            temperature=0,
            api_key=OPENAI_API_KEY,
            max_tokens=4096
        )
        llm.invoke("Hi")
        logger.info(f"Successfully connected to OpenAI with model: {OPENAI_MODEL_NAME}")
        return llm
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI API: {e}")
        st.error(f"Failed to initialize OpenAI model: {e}")
        return None