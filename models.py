# 📁 models.py (Code Changes Only)

import logging
import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import torch

import config
from config import EMBEDDING_MODEL_NAME, OPENAI_MODEL_NAME, OPENAI_API_KEY

logger = logging.getLogger(__name__)

def get_optimal_device():
    """Detects and returns the optimal PyTorch device for this machine."""
    # Check for NVIDIA CUDA GPU
    if torch.cuda.is_available():
        logger.info("✅ Found NVIDIA CUDA GPU. Using 'cuda'.")
        return "cuda"
    # Check for Apple Silicon GPU
    if torch.backends.mps.is_available():
        logger.info("✅ Found Apple Silicon GPU. Using 'mps'.")
        return "mps"
    # Fallback to CPU
    logger.info("⚠️ No GPU found. Falling back to 'cpu'.")
    return "cpu"

# Determine the device once when the module is loaded.
DEVICE = get_optimal_device()


def get_embedding_model():

    return SentenceTransformerEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": DEVICE}
    )


def get_llm():
    """Initializes and returns the OpenAI LLM."""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        st.error("OpenAI API key is not configured. Please set it in your .env file.")
        return None

    try:
        llm = ChatOpenAI(
            model=config.OPENAI_MODEL_NAME,
            temperature=0,
            api_key=OPENAI_API_KEY,
            max_tokens=4096  # gpt-4o-mini has a large context window, but we can cap output
        )
        # A lightweight test to ensure the key is valid
        llm.invoke("Hi")
        logger.info(f"Successfully connected to OpenAI with model: {OPENAI_MODEL_NAME}")
        return llm
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI API: {e}")
        # This will also catch authentication errors from a bad key
        st.error(f"Failed to initialize OpenAI model: {e}")
        return None