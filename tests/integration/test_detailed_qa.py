#!/usr/bin/env python3
"""
Detailed QA test to examine response content and citations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

from rag_core import CPUCRAGSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_detailed_response():
    """Test RAG system with detailed output examination"""
    
    # Initialize RAG system
    rag = CPUCRAGSystem("R2207005")
    
    # Test question about fixed charges
    question = "What are the residential fixed charges proposed in this proceeding?"
    
    print(f"Question: {question}")
    print("=" * 80)
    
    # Get the response
    for response in rag.query(question):
        if isinstance(response, dict):
            print("TECHNICAL ANSWER:")
            print(response.get("answer", "No answer"))
            print("\n" + "=" * 80)
            
            print("RAW ANSWER (for analysis):")
            print(response.get("raw_part1_answer", "No raw answer"))
            print("\n" + "=" * 80)
            
            print("SOURCES:")
            sources = response.get("sources", [])
            for i, source in enumerate(sources):
                print(f"{i+1}. {source}")
            print("\n" + "=" * 80)
            
            print("CONFIDENCE INDICATORS:")
            conf = response.get("confidence_indicators", {})
            for key, value in conf.items():
                print(f"  {key}: {value}")
            
            break
        else:
            print(f"Processing: {response}")

if __name__ == "__main__":
    test_detailed_response()