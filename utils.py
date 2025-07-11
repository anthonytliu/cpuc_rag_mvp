# 📁 utils.py
# Helper functions for text processing, date enhancement, and context manipulation.

import re
from datetime import datetime
from typing import List, Dict
import pytesseract
from PIL import Image
import io

from langchain.docstore.document import Document

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extracts text from an image using OCR (Tesseract)."""
    try:
        image = Image.open(io.BytesIO(image_bytes))  # Open from bytes
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")  # Log the error
        return ""

def extract_and_enhance_dates(text: str) -> str:
    """Advanced date extraction with regulatory timeline intelligence."""
    current_date = datetime.now()
    enhanced_text = text

    # Combined regex to find various date formats
    date_regex = re.compile(
        r"""
        \b
        (?:
            (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) # Month name
            \s+
            \d{1,2},?
            \s+
            \d{4}
        )
        |
        (?:
            \d{1,2}[-/]\d{1,2}[-/]\d{2,4} # MM/DD/YYYY or M/D/YY
        )
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    for match in date_regex.finditer(text):
        date_str = match.group(0)
        try:
            # Attempt to parse the found date string
            # This requires a more robust parser for different formats
            parsed_date = None
            if re.match(r'\w+\s+\d{1,2},?\s+\d{4}', date_str, re.IGNORECASE):
                date_str_clean = date_str.replace(',', '')
                parsed_date = datetime.strptime(date_str_clean, '%B %d %Y')
            elif re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', date_str):
                parsed_date = datetime.strptime(date_str.replace('-', '/'), '%m/%d/%Y')
            elif re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2}', date_str):
                 parsed_date = datetime.strptime(date_str.replace('-', '/'), '%m/%d/%y')

            if parsed_date:
                days_diff = (parsed_date - current_date).days
                if days_diff < -365:      context = f" [EXPIRED: {abs(days_diff)} days ago]"
                elif days_diff < 0:       context = f" [PAST: {abs(days_diff)} days ago]"
                elif days_diff == 0:      context = " [TODAY - URGENT]"
                elif days_diff <= 7:      context = f" [URGENT: in {days_diff} days]"
                elif days_diff <= 30:     context = f" [UPCOMING: in {days_diff} days]"
                else:                     context = f" [FUTURE: in {days_diff} days]"
                enhanced_text = enhanced_text.replace(date_str, f"{date_str}{context}")

        except (ValueError, IndexError):
            continue # Ignore if parsing fails

    return enhanced_text


def highlight_regulatory_terms(text: str, question: str) -> str:
    """Highlights important regulatory terms in context for the LLM's attention."""
    key_terms = [
        "shall", "must", "required", "deadline", "due date", "effective date",
        "compliance", "violation", "penalty", "approval", "authorization",
        "proceeding", "docket", "order", "decision", "ruling", "finding"
    ]
    question_words = [word.lower() for word in question.split() if len(word) > 3]
    all_terms = list(set(key_terms + question_words))

    highlighted_text = text
    for term in all_terms:
        # Use word boundaries to avoid matching parts of words
        highlighted_text = re.sub(
            f'\\b({re.escape(term)})\\b',
            r'**\1**',
            highlighted_text,
            flags=re.IGNORECASE
        )
    return highlighted_text


def find_cross_references(documents: List[Document]) -> Dict[str, List[str]]:
    """Finds cross-references (e.g., docket numbers) within documents."""
    cross_refs = {}
    ref_pattern = r'(?:[A-Z]\.)?\d{2}-\d{2}-\d{3}'

    for doc in documents:
        source = doc.metadata.get('source', 'Unknown')
        matches = re.findall(ref_pattern, doc.page_content)
        if matches:
            if source not in cross_refs:
                cross_refs[source] = []
            cross_refs[source].extend(matches)

    for source in cross_refs:
        cross_refs[source] = list(set(cross_refs[source]))

    return cross_refs

def create_fallback_answer(context: str, question: str) -> str:
    """Creates a structured fallback answer when the LLM is unavailable."""
    current_date = datetime.now().strftime("%B %d, %Y")
    return f"""LLM is not available. Based on retrieved documents (analyzed as of {current_date}):

**Question:** {question}

**Relevant Information Found:**
{context[:3000]}{"..." if len(context) > 3000 else ""}

**Analysis Note:** This is a raw data retrieval. Please review the source documents for specific details, deadlines, and requirements."""


def check_source_consistency(documents: List[Document]) -> bool:
    """Checks if retrieved documents are from a small number of proceedings."""
    if not documents:
        return True
    proceedings = {doc.metadata.get("proceeding", "unknown") for doc in documents}
    return len(proceedings) <= 3