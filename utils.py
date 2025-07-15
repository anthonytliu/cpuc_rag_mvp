import re
from datetime import datetime
from typing import List, Dict

from langchain.docstore.document import Document

def extract_and_enhance_dates(text: str) -> str:
    """
    Advanced date extraction with regulatory timeline intelligence.
    
    This function parses text to find date patterns and enhances them with context
    about whether they are past, present, or future dates relative to today.
    It's specifically designed for regulatory documents where timing context is crucial.
    
    Args:
        text (str): The input text containing potential date references
        
    Returns:
        str: The enhanced text with dates annotated with timing context
             such as [EXPIRED], [URGENT], [UPCOMING], etc.
             
    Examples:
        >>> extract_and_enhance_dates("Deadline: December 15, 2023")
        "Deadline: December 15, 2023 [PAST: 45 days ago]"
        
        >>> extract_and_enhance_dates("Filing due 12/15/2025")
        "Filing due 12/15/2025 [FUTURE: in 365 days]"
    """
    current_date = datetime.now()
    enhanced_text = text

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
            continue

    return enhanced_text


def highlight_regulatory_terms(text: str, question: str) -> str:
    """
    Highlights important regulatory terms in context for the LLM's attention.
    
    This function processes text to emphasize regulatory keywords and question-specific
    terms by wrapping them in markdown bold formatting. This helps the LLM focus on
    the most relevant information when processing regulatory documents.
    
    Args:
        text (str): The input text to be processed
        question (str): The user's question to extract relevant terms from
        
    Returns:
        str: The text with regulatory terms and question keywords highlighted
             using markdown bold formatting (**term**)
             
    Examples:
        >>> highlight_regulatory_terms("The deadline shall be met", "deadline requirements")
        "The **deadline** **shall** be met"
        
        >>> highlight_regulatory_terms("CPUC ruling on compliance", "ruling")
        "CPUC **ruling** on **compliance**"
    """
    key_terms = [
        "shall", "must", "required", "deadline", "due date", "effective date",
        "compliance", "violation", "penalty", "approval", "authorization",
        "proceeding", "docket", "order", "decision", "ruling", "finding"
    ]
    question_words = [word.lower() for word in question.split() if len(word) > 3]
    all_terms = list(set(key_terms + question_words))

    highlighted_text = text
    for term in all_terms:
        highlighted_text = re.sub(
            f'\\b({re.escape(term)})\\b',
            r'**\1**',
            highlighted_text,
            flags=re.IGNORECASE
        )
    return highlighted_text




def check_source_consistency(documents: List[Document]) -> bool:
    """
    Checks if retrieved documents are from a small number of proceedings.
    
    This function evaluates whether the retrieved documents come from a limited
    number of regulatory proceedings, which indicates good source consistency.
    Documents from fewer proceedings suggest more focused and coherent results.
    
    Args:
        documents (List[Document]): List of retrieved documents to analyze
        
    Returns:
        bool: True if documents come from 3 or fewer proceedings, False otherwise.
              Returns True for empty document lists.
              
    Examples:
        >>> docs = [doc1, doc2, doc3]  # All from same proceeding
        >>> check_source_consistency(docs)
        True
        
        >>> docs = [doc1, doc2, doc3, doc4]  # From 4 different proceedings
        >>> check_source_consistency(docs)
        False
    """
    if not documents:
        return True
    proceedings = {doc.metadata.get("proceeding", "unknown") for doc in documents}
    return len(proceedings) <= 3