# 📁 memory.py
# Manages conversation history and extracts entities.

import re
from datetime import datetime
from typing import List, Dict


class ConversationMemory:
    def __init__(self):
        self.conversation_history = []
        self.extracted_entities = {}
        self.regulatory_context = {}
        self.entity_patterns = {
            'deadlines': r'(?:due|deadline|by|no later than|within)\s+([^.]+?)(?:\.|,|;)',
            'parties': r'(?:applicant|petitioner|respondent|intervenor|party)\s+([^.]+?)(?:\.|,|;)',
            'proceedings': r'(?:proceeding|docket|case)\s+(?:no\.?|number)?\s*([A-Z]?\d{2}-\d{2}-\d{3})',
            'orders': r'(?:order|decision|ruling)\s+(\d{2}-\d{2}-\d{3})',
            'requirements': r'(?:shall|must|required to|obligated to)\s+([^.]+?)(?:\.|,|;)'
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract regulatory entities from text"""
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        return entities

    def add_query_context(self, query: str, answer: str, sources: List[Dict], extracted_entities: Dict):
        """Add query context to memory"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer,
            'sources': [s['document'] for s in sources],
            'entities': extracted_entities
        }
        self.conversation_history.append(context)

        for entity_type, values in extracted_entities.items():
            if entity_type not in self.extracted_entities:
                self.extracted_entities[entity_type] = []
            self.extracted_entities[entity_type].extend(values)

    def get_relevant_history(self, current_query: str) -> str:
        """Get relevant conversation history for current query"""
        if not self.conversation_history:
            return ""

        relevant_context = []
        query_words = set(current_query.lower().split())

        for context in self.conversation_history[-5:]:  # Last 5 interactions
            previous_words = set(context['query'].lower().split())
            overlap = len(query_words.intersection(previous_words))

            if overlap > 0:
                relevant_context.append(f"Previous Query: {context['query']}")
                relevant_context.append(f"Key Entities Found: {context['entities']}")

        return "\n".join(relevant_context) if relevant_context else ""
