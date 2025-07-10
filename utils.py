# 📁 utils.py
# Helper functions for text processing, date enhancement, and context manipulation.

import re
import hashlib
from datetime import datetime
from typing import List, Dict

from langchain.docstore.document import Document


def extract_and_enhance_dates(text: str) -> str:
    """Advanced date extraction with regulatory timeline intelligence"""
    current_date = datetime.now()

    # Enhanced date patterns
    date_patterns = {
        'standard': r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',
        'written': r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
        'deadline': r'(?:due|deadline|by|no later than)\s+([^.]+?)(?:\.|,|;)',
        'effective': r'(?:effective|takes effect|becomes effective)\s+([^.]+?)(?:\.|,|;)',
        'filing_window': r'(?:filing window|comment period|from|between)\s+([^.]+?)(?:\.|,|;)',
        'conditional': r'(?:if|unless|provided that|subject to).*?([^.]+?)(?:\.|,|;)'
    }

    enhanced_text = text
    timeline_items = []

    for pattern_type, pattern in date_patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            original_text = match.group(0)

            # Try to parse dates from the match
            try:
                if pattern_type == 'standard':
                    month, day, year = match.groups()
                    parsed_date = datetime(int(year), int(month), int(day))
                elif pattern_type == 'written':
                    month_name, day, year = match.groups()
                    month_num = datetime.strptime(month_name, '%B').month
                    parsed_date = datetime(int(year), month_num, int(day))
                else:
                    # For complex patterns, try to extract dates from the captured group
                    date_text = match.group(1)
                    # Simple date extraction (you could make this more sophisticated)
                    date_match = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', date_text)
                    if date_match:
                        month, day, year = date_match.groups()
                        parsed_date = datetime(int(year), int(month), int(day))
                    else:
                        continue

                # Calculate temporal context
                days_diff = (parsed_date - current_date).days

                if days_diff < -365:
                    context = f" [EXPIRED: {abs(days_diff)} days ago]"
                elif days_diff < 0:
                    context = f" [PAST: {abs(days_diff)} days ago]"
                elif days_diff == 0:
                    context = " [TODAY - URGENT]"
                elif days_diff <= 7:
                    context = f" [URGENT: {days_diff} days remaining]"
                elif days_diff <= 30:
                    context = f" [UPCOMING: {days_diff} days from now]"
                else:
                    context = f" [FUTURE: {days_diff} days from now]"

                # Add regulatory urgency markers
                if pattern_type == 'deadline' and days_diff <= 30:
                    context += " ⚠️"
                elif pattern_type == 'effective' and abs(days_diff) <= 7:
                    context += " 🔥"

                enhanced_text = enhanced_text.replace(original_text, f"{original_text}{context}")

                timeline_items.append({
                    'type': pattern_type,
                    'date': parsed_date,
                    'text': original_text,
                    'days_from_now': days_diff
                })

            except (ValueError, IndexError):
                continue

    return enhanced_text


def highlight_regulatory_terms(text: str, question: str) -> str:
    """Highlight important regulatory terms in context"""

    # Key regulatory terms to emphasize
    key_terms = [
        "shall", "must", "required", "deadline", "due date", "effective date",
        "compliance", "violation", "penalty", "approval", "authorization",
        "proceeding", "docket", "order", "decision", "ruling", "finding",
        "public interest", "just and reasonable", "burden of proof"
    ]

    # Question-specific terms
    question_words = [word.lower() for word in question.split() if len(word) > 3]

    # Combine and deduplicate
    all_terms = list(set(key_terms + question_words))

    # Simple emphasis (you could make this more sophisticated)
    highlighted_text = text
    for term in all_terms:
        if term in text.lower():
            # Add emphasis markers (the LLM will see these)
            highlighted_text = re.sub(
                f'\\b{re.escape(term)}\\b',
                f'**{term}**',
                highlighted_text,
                flags=re.IGNORECASE
            )

    return highlighted_text


def find_cross_references(documents: List[Document]) -> Dict[str, List[str]]:
    """Find cross-references between documents"""
    cross_refs = {}

    # Patterns for regulatory cross-references
    ref_patterns = [
        r'(?:proceeding|docket|case)\s+(?:no\.?|number)?\s*([A-Z]?\d{2}-\d{2}-\d{3})',
        r'(?:decision|ruling|order)\s+(\d{2}-\d{2}-\d{3})',
        r'(?:application|petition)\s+(\d{2}-\d{2}-\d{3})',
        r'(?:see|refer to|pursuant to)\s+([A-Z]?\d{2}-\d{2}-\d{3})'
    ]

    for doc in documents:
        source = doc.metadata.get('source', 'Unknown')
        refs = []

        for pattern in ref_patterns:
            matches = re.findall(pattern, doc.page_content, re.IGNORECASE)
            refs.extend(matches)

        if refs:
            cross_refs[source] = list(set(refs))

    return cross_refs


def create_fallback_answer(context: str, question: str) -> str:
    """Create a structured fallback answer when LLM is unavailable"""
    current_date = datetime.now().strftime("%B %d, %Y")

    return f"""Based on retrieved CPUC documents (analyzed as of {current_date}):

    QUESTION: {question}

    RELEVANT INFORMATION FOUND:
    {context[:3000]}{"..." if len(context) > 3000 else ""}

    ANALYSIS NOTE: This response is based on document retrieval without LLM processing. 
    Please review the source documents above for:
    - Specific deadlines and dates
    - Regulatory requirements and obligations  
    - Cross-references to other proceedings
    - Current applicability and status

    For a complete regulatory analysis, consider reviewing the full source documents listed below."""


def check_source_consistency(documents: List[Document]) -> bool:
    """Check if sources are from consistent proceedings/documents"""
    proceedings = set()
    for doc in documents:
        proceedings.add(doc.metadata.get("proceeding", "unknown"))
    return len(proceedings) <= 2
