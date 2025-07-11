# 📁 memory.py
# Manages conversation history and extracts entities.

import re
from datetime import datetime
from typing import List, Dict

class ConversationMemory:
    def __init__(self):
        self.conversation_history = []
        self.entity_patterns = {
            'deadlines': r'(?:due|deadline|by|no later than|within)\s+([^.\n]+?)(?:\.|;|\n)',
            'parties': r'(?:applicant|petitioner|respondent|intervenor|party)\s+([^.\n]+?)(?:\.|;|\n)',
            'proceedings': r'[A-Z]?\d{2}-\d{2}-\d{3}',
            'requirements': r'(?:shall|must|required to|obligated to)\s+([^.\n]+?)(?:\.|;|\n)'
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extracts regulatory entities from text."""
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set([m.strip() for m in matches]))
        return entities

    def add_query_context(self, query: str, answer: str, sources: List[Dict], extracted_entities: Dict):
        """Adds a query, its answer, and context to the conversation history."""
        context = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer,
            'sources': [s.get('document', 'Unknown') for s in sources],
            'entities': extracted_entities
        }
        self.conversation_history.append(context)

    def get_relevant_history(self, current_query: str) -> str:
        """Gets a summary of relevant parts of the conversation history."""
        if not self.conversation_history:
            return ""

        relevant_context = []
        current_words = set(current_query.lower().split())

        # Look at the last 3 interactions
        for context in self.conversation_history[-3:]:
            previous_words = set(context['query'].lower().split())
            if len(current_words.intersection(previous_words)) > 1:
                relevant_context.append(f"Previously, you asked: \"{context['query']}\"")
                if context['entities']:
                    relevant_context.append(f"Key entities found were: {context['entities']}")

        return "\n".join(relevant_context) if relevant_context else ""