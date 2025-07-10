# 📁 utils.py
# Helper functions for text processing, date enhancement, and context manipulation.

import re
import hashlib
from datetime import datetime
from typing import List, Dict

from langchain.docstore.document import Document


def _extract_and_enhance_dates(text: str) -> str:
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


def _highlight_regulatory_terms(text: str, question: str) -> str:
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


def _find_cross_references(documents: List[Document]) -> Dict[str, List[str]]:
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


def _rank_by_regulatory_relevance(documents: List[Document], query: str) -> List[Document]:
    """Rank documents by regulatory relevance"""

    # Regulatory importance weights
    section_weights = {
        'order': 1.0,
        'finding': 0.9,
        'rule': 0.8,
        'requirement': 0.7,
        'definition': 0.6,
        'parent': 0.8,
        'child': 0.6
    }

    # Query-specific weights
    query_lower = query.lower()
    temporal_queries = any(term in query_lower for term in ['deadline', 'when', 'date', 'time', 'schedule'])
    compliance_queries = any(term in query_lower for term in ['requirement', 'must', 'shall', 'comply'])

    scored_docs = []

    for doc in documents:
        score = 0.0

        # Base section type score
        section_type = doc.metadata.get('section_type', 'unknown')
        chunk_type = doc.metadata.get('chunk_type', 'regular')

        score += section_weights.get(section_type, 0.5)
        score += section_weights.get(chunk_type, 0.5)

        # Temporal relevance boost
        if temporal_queries:
            if any(term in doc.page_content.lower() for term in ['deadline', 'due', 'effective', 'expires']):
                score += 0.3

        # Compliance relevance boost
        if compliance_queries:
            if any(term in doc.page_content.lower() for term in ['shall', 'must', 'required', 'obligated']):
                score += 0.3

        # Recency boost
        try:
            last_modified = doc.metadata.get('last_modified', '')
            if last_modified:
                doc_date = datetime.fromisoformat(last_modified.replace('Z', '+00:00').replace('+00:00', ''))
                days_old = (datetime.now() - doc_date).days
                if days_old < 365:
                    score += 0.2
        except:
            pass

        scored_docs.append((doc, score))

    # Sort by score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs]


def _create_fallback_answer(context: str, question: str) -> str:
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


def _assess_confidence(self, documents: List[Document], answer: str, question: str) -> Dict:
    """Enhanced confidence assessment with regulatory-specific indicators"""

    # Basic indicators
    indicators = {
        "num_sources": len(documents),
        "has_exact_quotes": '"' in answer,
        "has_page_references": any("page" in str(doc.metadata) for doc in documents),
        "source_consistency": _check_source_consistency(documents),
        "model_type": "Local (Ollama)" if self.llm else "Retrieval Only"
    }

    # Regulatory-specific confidence indicators

    # Date relevance
    current_date = datetime.now()
    recent_docs = 0
    for doc in documents:
        try:
            last_modified = doc.metadata.get("last_modified", "")
            if last_modified:
                doc_date = datetime.fromisoformat(last_modified.replace('Z', '+00:00').replace('+00:00', ''))
                if (current_date - doc_date).days < 365:  # Within last year
                    recent_docs += 1
        except:
            pass

    indicators["recent_documents"] = recent_docs
    indicators["date_relevance"] = "High" if recent_docs > len(
        documents) * 0.5 else "Medium" if recent_docs > 0 else "Low"

    # Content quality indicators
    total_content_length = sum(len(doc.page_content) for doc in documents)
    indicators[
        "content_depth"] = "High" if total_content_length > 10000 else "Medium" if total_content_length > 3000 else "Low"

    # Question-answer alignment
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    alignment_score = len(question_words.intersection(answer_words)) / len(question_words) if question_words else 0
    indicators["question_alignment"] = alignment_score

    # Regulatory term coverage
    regulatory_terms = [
        "deadline", "requirement", "compliance", "proceeding", "decision",
        "ruling", "order", "effective", "filing", "comment", "approval"
    ]

    found_terms = sum(1 for term in regulatory_terms if term in answer.lower())
    indicators["regulatory_coverage"] = found_terms / len(regulatory_terms)

    # Specificity indicators
    indicators["has_dates"] = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', answer))
    indicators["has_specific_references"] = bool(re.search(r'(page|section|paragraph|docket)', answer.lower()))
    indicators["has_deadlines"] = any(
        word in answer.lower() for word in ["deadline", "due", "expires", "effective"])

    # Overall confidence score
    confidence_factors = [
        indicators["num_sources"] >= 3,
        indicators["source_consistency"],
        indicators["content_depth"] in ["High", "Medium"],
        indicators["question_alignment"] > 0.3,
        indicators["regulatory_coverage"] > 0.2,
        indicators["has_specific_references"]
    ]

    confidence_score = sum(confidence_factors) / len(confidence_factors)
    indicators[
        "overall_confidence"] = "High" if confidence_score > 0.7 else "Medium" if confidence_score > 0.4 else "Low"

    return indicators


def _check_source_consistency(documents: List[Document]) -> bool:
    """Check if sources are from consistent proceedings/documents"""
    proceedings = set()
    for doc in documents:
        proceedings.add(doc.metadata.get("proceeding", "unknown"))
    return len(proceedings) <= 2
