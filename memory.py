import re
from datetime import datetime
from typing import List, Dict

class ConversationMemory:
    """
    Manages conversation history and extracts entities from regulatory documents.
    This class provides functionality to maintain query context and extract
    regulatory entities from text.
    """
    
    def __init__(self):
        """
        Initialize the conversation memory with empty history and entity patterns.
        
        The entity patterns are designed to capture regulatory-specific information
        like deadlines, parties, proceedings, and requirements.
        """
        self.conversation_history = []
        self.entity_patterns = {
            'deadlines': r'(?:due|deadline|by|no later than|within)\s+([^.\n]+?)(?:\.|;|\n)',
            'parties': r'(?:applicant|petitioner|respondent|intervenor|party)\s+([^.\n]+?)(?:\.|;|\n)',
            'proceedings': r'[A-Z]?\d{2}-\d{2}-\d{3}',
            'requirements': r'(?:shall|must|required to|obligated to)\s+([^.\n]+?)(?:\.|;|\n)'
        }