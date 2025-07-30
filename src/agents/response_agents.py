# 📁 response_agents.py
# Specialized response agents for the CPUC regulatory search engine

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

# Try relative imports first, fall back to absolute
try:
    from ..core import config
except ImportError:
    from core import config

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Standardized response structure for all agents."""
    content: str
    agent_type: str
    confidence_score: float
    sources_used: List[str]
    processing_time: float
    metadata: Dict[str, Any]


class BaseResponseAgent(ABC):
    """Base class for all specialized response agents."""
    
    def __init__(self, llm, current_proceeding: str = None):
        self.llm = llm
        self.current_proceeding = current_proceeding or config.DEFAULT_PROCEEDING
        self.search_tool = DuckDuckGoSearchResults(num_results=5)
        
    @abstractmethod
    def generate_response(self, question: str, documents: List[Document], raw_technical_answer: str = None) -> AgentResponse:
        """Generate a specialized response based on the agent's expertise."""
        pass
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation references from text."""
        citation_pattern = re.compile(r"\[CITE:\s*([^,]+),\s*page_(\d+)\]")
        citations = citation_pattern.findall(text)
        return [f"{filename} (page {page})" for filename, page in citations]
    
    def _calculate_confidence_score(self, documents: List[Document], answer: str, question: str) -> float:
        """Calculate confidence score based on various factors."""
        if not documents or not answer:
            return 0.0
        
        factors = []
        
        # Number of sources factor
        source_factor = min(len(documents) / 5.0, 1.0)  # Max at 5 sources
        factors.append(source_factor)
        
        # Citation factor
        citations = self._extract_citations(answer)
        citation_factor = min(len(citations) / 3.0, 1.0)  # Max at 3 citations
        factors.append(citation_factor)
        
        # Length and completeness factor
        length_factor = min(len(answer) / 1000.0, 1.0)  # Max at 1000 chars
        factors.append(length_factor)
        
        # Question-answer alignment (basic keyword matching)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        alignment_factor = len(question_words.intersection(answer_words)) / len(question_words) if question_words else 0
        factors.append(min(alignment_factor, 1.0))
        
        return sum(factors) / len(factors)


class TechnicalIndustryExpertAgent(BaseResponseAgent):
    """
    Provides responses in specific, exact regulatory and technical language.
    Focuses on precise legal citations and regulatory requirements.
    Maintains strict compliance terminology.
    """
    
    def __init__(self, llm, current_proceeding: str = None):
        super().__init__(llm, current_proceeding)
        self.prompt_template = PromptTemplate.from_template(config.TECHNICAL_EXPERT_PROMPT_TEMPLATE)
        
    def generate_response(self, question: str, documents: List[Document], raw_technical_answer: str = None) -> AgentResponse:
        """Generate a technical expert response with precise regulatory language."""
        start_time = datetime.now()
        
        try:
            # Enhanced context preparation for technical analysis
            context = self._prepare_technical_context(documents, question)
            
            # Generate technical response
            prompt = self.prompt_template.format(
                context=context,
                question=question,
                current_date=datetime.now().strftime("%B %d, %Y"),
                current_proceeding=self.current_proceeding
            )
            
            response_message = self.llm.invoke(prompt)
            technical_content = response_message.content
            
            # Post-process for regulatory compliance
            processed_content = self._enhance_regulatory_compliance(technical_content, documents)
            
            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence_score = self._calculate_confidence_score(documents, processed_content, question)
            sources_used = [doc.metadata.get('source', 'Unknown') for doc in documents]
            citations = self._extract_citations(processed_content)
            
            return AgentResponse(
                content=processed_content,
                agent_type="Technical Industry Expert",
                confidence_score=confidence_score,
                sources_used=sources_used,
                processing_time=processing_time,
                metadata={
                    "regulatory_terms_count": self._count_regulatory_terms(processed_content),
                    "citation_count": len(citations),
                    "compliance_indicators": self._identify_compliance_indicators(processed_content),
                    "proceeding_context": self.current_proceeding
                }
            )
            
        except Exception as e:
            logger.error(f"Technical expert agent failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=f"Technical analysis could not be completed due to processing error: {str(e)}",
                agent_type="Technical Industry Expert",
                confidence_score=0.0,
                sources_used=[],
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _prepare_technical_context(self, documents: List[Document], question: str) -> str:
        """Prepare enhanced context with regulatory emphasis."""
        context_parts = []
        
        for i, doc in enumerate(documents):
            source_info = f"[Regulatory Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}]"
            
            # Extract and highlight regulatory content
            content = doc.page_content
            content = self._highlight_regulatory_elements(content)
            content = self._extract_compliance_requirements(content)
            
            context_parts.append(f"{source_info}\n{content}")
        
        return "\n\n---REGULATORY DOCUMENT SEPARATOR---\n\n".join(context_parts)
    
    def _highlight_regulatory_elements(self, content: str) -> str:
        """Highlight key regulatory elements in the content."""
        # Regulatory terms that should be emphasized
        regulatory_terms = [
            r'shall\s+(?:be\s+)?(?:required|comply|maintain|provide|submit|file)',
            r'must\s+(?:be\s+)?(?:comply|maintain|provide|submit|file)',
            r'required\s+to\s+(?:comply|maintain|provide|submit|file)',
            r'pursuant\s+to',
            r'in\s+accordance\s+with',
            r'as\s+defined\s+in',
            r'compliance\s+with',
            r'violation\s+of',
            r'penalty\s+of',
            r'fine\s+not\s+to\s+exceed',
            r'deadline\s+of',
            r'effective\s+date',
            r'implementation\s+schedule'
        ]
        
        for pattern in regulatory_terms:
            content = re.sub(pattern, f"**{pattern.upper()}**", content, flags=re.IGNORECASE)
        
        return content
    
    def _extract_compliance_requirements(self, content: str) -> str:
        """Extract and emphasize compliance requirements."""
        # Look for numbered requirements, deadlines, and specific obligations
        requirement_patterns = [
            r'(\d+\.\s*[A-Z][^.]*(?:shall|must|required)[^.]*\.)',
            r'((?:The\s+)?[A-Z][^.]*(?:deadline|due date|effective date)[^.]*\.)',
            r'((?:The\s+)?[A-Z][^.]*(?:penalty|fine|violation)[^.]*\.)'
        ]
        
        enhanced_content = content
        for pattern in requirement_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                enhanced_content = enhanced_content.replace(match, f"\n**REGULATORY REQUIREMENT:** {match}\n")
        
        return enhanced_content
    
    def _enhance_regulatory_compliance(self, content: str, documents: List[Document]) -> str:
        """Enhance content with regulatory compliance indicators."""
        # Add regulatory framework context
        framework_info = self._extract_regulatory_framework(documents)
        if framework_info:
            content = f"{framework_info}\n\n{content}"
        
        # Add cross-references to related proceedings
        cross_refs = self._identify_cross_references(content)
        if cross_refs:
            content += f"\n\n**Cross-References to Related Proceedings:**\n{cross_refs}"
        
        return content
    
    def _extract_regulatory_framework(self, documents: List[Document]) -> str:
        """Extract regulatory framework information from documents."""
        framework_terms = ['Public Utilities Code', 'General Order', 'Resolution', 'Decision', 'Rulemaking']
        frameworks_found = set()
        
        for doc in documents:
            content = doc.page_content.lower()
            for term in framework_terms:
                if term.lower() in content:
                    frameworks_found.add(term)
        
        if frameworks_found:
            return f"**Regulatory Framework Context:** This analysis is governed by: {', '.join(frameworks_found)}"
        return ""
    
    def _identify_cross_references(self, content: str) -> str:
        """Identify cross-references to other proceedings or regulations."""
        # Pattern for proceeding references (R.XX-XX-XXX format)
        proceeding_pattern = r'R\.\d{2}-\d{2}-\d{3}'
        proceedings = re.findall(proceeding_pattern, content)
        
        # Pattern for decision references (D.XX-XX-XXX format)
        decision_pattern = r'D\.\d{2}-\d{2}-\d{3}'
        decisions = re.findall(decision_pattern, content)
        
        cross_refs = []
        if proceedings:
            cross_refs.append(f"• Related Proceedings: {', '.join(set(proceedings))}")
        if decisions:
            cross_refs.append(f"• Related Decisions: {', '.join(set(decisions))}")
        
        return '\n'.join(cross_refs)
    
    def _count_regulatory_terms(self, content: str) -> int:
        """Count regulatory terms in the content."""
        regulatory_terms = [
            'shall', 'must', 'required', 'compliance', 'violation', 'penalty',
            'pursuant', 'accordance', 'regulation', 'rule', 'order', 'decision'
        ]
        
        count = 0
        content_lower = content.lower()
        for term in regulatory_terms:
            count += content_lower.count(term)
        
        return count
    
    def _identify_compliance_indicators(self, content: str) -> List[str]:
        """Identify compliance indicators in the content."""
        indicators = []
        
        if re.search(r'deadline|due date|effective date', content, re.IGNORECASE):
            indicators.append("time_sensitive")
        
        if re.search(r'penalty|fine|violation', content, re.IGNORECASE):
            indicators.append("enforcement_related")
        
        if re.search(r'shall|must|required', content, re.IGNORECASE):
            indicators.append("mandatory_requirement")
        
        if re.search(r'may|optional|discretionary', content, re.IGNORECASE):
            indicators.append("discretionary_provision")
        
        return indicators


class LaymenInterpretationAgent(BaseResponseAgent):
    """
    Translates complex regulatory language into clear, understandable terms.
    Explains policy implications for general audience.
    Simplifies technical jargon while maintaining accuracy.
    """
    
    def __init__(self, llm, current_proceeding: str = None):
        super().__init__(llm, current_proceeding)
        self.prompt_template = PromptTemplate.from_template(config.LAYMEN_INTERPRETATION_PROMPT_TEMPLATE)
        
    def generate_response(self, question: str, documents: List[Document], raw_technical_answer: str = None) -> AgentResponse:
        """Generate a simplified, accessible explanation for general audiences."""
        start_time = datetime.now()
        
        try:
            # Use the technical answer as primary input for translation
            if not raw_technical_answer:
                raw_technical_answer = "Complex regulatory information provided in source documents."
            
            # Generate laymen interpretation
            prompt = self.prompt_template.format(
                technical_answer=raw_technical_answer,
                question=question,
                current_proceeding=self.current_proceeding
            )
            
            response_message = self.llm.invoke(prompt)
            laymen_content = response_message.content
            
            # Enhance with practical examples and analogies
            enhanced_content = self._add_practical_examples(laymen_content, question)
            enhanced_content = self._add_impact_analysis(enhanced_content, documents)
            
            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence_score = self._calculate_laymen_confidence(enhanced_content, raw_technical_answer)
            sources_used = [doc.metadata.get('source', 'Unknown') for doc in documents]
            
            return AgentResponse(
                content=enhanced_content,
                agent_type="Laymen Interpretation",
                confidence_score=confidence_score,
                sources_used=sources_used,
                processing_time=processing_time,
                metadata={
                    "readability_score": self._calculate_readability_score(enhanced_content),
                    "jargon_terms_simplified": self._count_simplified_terms(enhanced_content),
                    "practical_examples_added": self._count_practical_examples(enhanced_content),
                    "proceeding_context": self.current_proceeding
                }
            )
            
        except Exception as e:
            logger.error(f"Laymen interpretation agent failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=f"Simplified explanation could not be generated due to processing error: {str(e)}",
                agent_type="Laymen Interpretation",
                confidence_score=0.0,
                sources_used=[],
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _add_practical_examples(self, content: str, question: str) -> str:
        """Add practical examples and real-world scenarios."""
        examples_map = {
            'rate': 'For example, if your monthly electricity bill is $100, this change might increase it by about $5.',
            'tariff': 'This is like a pricing menu that shows how much you pay for electricity at different times.',
            'demand response': 'Think of this like a program where you get paid to use less electricity when the grid is stressed.',
            'microgrid': 'Imagine a mini power grid for your neighborhood that can work independently during outages.',
            'interconnection': 'This is the process of connecting your solar panels to the main power grid.',
            'net metering': 'This lets you sell extra solar power back to the utility, like running your meter backwards.',
            'time-of-use': 'Electricity costs different amounts depending on the time of day, like surge pricing for rideshares.'
        }
        
        enhanced_content = content
        question_lower = question.lower()
        
        for term, example in examples_map.items():
            if term in question_lower and term in content.lower():
                # Add example after first mention of the term
                pattern = rf'\b{re.escape(term)}\b'
                match = re.search(pattern, enhanced_content, re.IGNORECASE)
                if match:
                    insert_pos = match.end()
                    enhanced_content = (enhanced_content[:insert_pos] + 
                                      f" ({example})" + 
                                      enhanced_content[insert_pos:])
                    break
        
        return enhanced_content
    
    def _add_impact_analysis(self, content: str, documents: List[Document]) -> str:
        """Add analysis of real-world impacts on consumers and businesses."""
        impact_indicators = []
        
        # Analyze documents for impact indicators
        for doc in documents:
            doc_content = doc.page_content.lower()
            
            if any(word in doc_content for word in ['cost', 'rate', 'fee', 'charge']):
                impact_indicators.append("financial_impact")
            
            if any(word in doc_content for word in ['customer', 'consumer', 'residential']):
                impact_indicators.append("consumer_impact")
            
            if any(word in doc_content for word in ['business', 'commercial', 'industrial']):
                impact_indicators.append("business_impact")
            
            if any(word in doc_content for word in ['deadline', 'effective', 'timeline']):
                impact_indicators.append("timing_impact")
        
        # Add impact section if indicators found
        if impact_indicators:
            impact_text = "\n\n<h4>Real-World Impact</h4>\n<ul>"
            
            if "financial_impact" in impact_indicators:
                impact_text += "<li><strong>Financial:</strong> This regulation may affect your utility bills or costs.</li>"
            
            if "consumer_impact" in impact_indicators:
                impact_text += "<li><strong>For Consumers:</strong> This primarily affects residential customers and homeowners.</li>"
            
            if "business_impact" in impact_indicators:
                impact_text += "<li><strong>For Businesses:</strong> Commercial and industrial customers should pay attention to these changes.</li>"
            
            if "timing_impact" in impact_indicators:
                impact_text += "<li><strong>Important Dates:</strong> There are specific deadlines or effective dates to be aware of.</li>"
            
            impact_text += "</ul>"
            content += impact_text
        
        return content
    
    def _calculate_laymen_confidence(self, laymen_content: str, technical_content: str) -> float:
        """Calculate confidence score specific to laymen interpretation."""
        factors = []
        
        # Length appropriateness (not too short, not too long)
        length_factor = 1.0 if 200 <= len(laymen_content) <= 800 else 0.5
        factors.append(length_factor)
        
        # Simplified language factor (fewer complex terms)
        complex_terms = ['pursuant', 'aforementioned', 'heretofore', 'whereas', 'therein']
        complex_count = sum(1 for term in complex_terms if term in laymen_content.lower())
        simplicity_factor = max(0.0, 1.0 - (complex_count * 0.2))
        factors.append(simplicity_factor)
        
        # Practical information factor
        practical_words = ['example', 'like', 'means', 'think of', 'imagine', 'for instance']
        practical_count = sum(1 for word in practical_words if word in laymen_content.lower())
        practical_factor = min(practical_count * 0.3, 1.0)
        factors.append(practical_factor)
        
        # Coverage factor (addresses key points from technical content)
        if technical_content:
            tech_key_words = set(re.findall(r'\b\w{4,}\b', technical_content.lower()))
            laymen_words = set(re.findall(r'\b\w{4,}\b', laymen_content.lower()))
            coverage_factor = len(tech_key_words.intersection(laymen_words)) / max(len(tech_key_words), 1)
            factors.append(min(coverage_factor, 1.0))
        
        return sum(factors) / len(factors)
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate approximate readability score."""
        # Simple readability approximation
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(content.split())
        
        if sentences == 0:
            return 0.0
        
        avg_sentence_length = words / sentences
        
        # Score based on sentence length (shorter is better for readability)
        if avg_sentence_length <= 15:
            return 1.0
        elif avg_sentence_length <= 20:
            return 0.8
        elif avg_sentence_length <= 25:
            return 0.6
        else:
            return 0.4
    
    def _count_simplified_terms(self, content: str) -> int:
        """Count terms that have been simplified."""
        simplification_indicators = [
            'this means', 'in other words', 'simply put', 'basically',
            'for example', 'like', 'think of it as', 'imagine'
        ]
        
        count = 0
        content_lower = content.lower()
        for indicator in simplification_indicators:
            count += content_lower.count(indicator)
        
        return count
    
    def _count_practical_examples(self, content: str) -> int:
        """Count practical examples in the content."""
        example_patterns = [
            r'for example[^.]*\.',
            r'imagine[^.]*\.',
            r'like[^.]*\.',
            r'think of[^.]*\.'
        ]
        
        count = 0
        for pattern in example_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            count += len(matches)
        
        return count


class FurtherSourcesResearcherAgent(BaseResponseAgent):
    """
    Conducts web searches for additional context and recent developments.
    Finds related news articles, industry analysis, and commentary.
    Identifies relevant external resources and authorities.
    """
    
    def __init__(self, llm, current_proceeding: str = None):
        super().__init__(llm, current_proceeding)
        self.prompt_template = PromptTemplate.from_template(config.FURTHER_SOURCES_PROMPT_TEMPLATE)
        self.search_tool = DuckDuckGoSearchResults(num_results=8)  # More results for research
        
    def generate_response(self, question: str, documents: List[Document], raw_technical_answer: str = None) -> AgentResponse:
        """Generate additional sources and research context."""
        start_time = datetime.now()
        
        try:
            # Generate search queries based on the question and proceeding
            search_queries = self._generate_search_queries(question, self.current_proceeding)
            
            # Perform web searches
            search_results = self._perform_web_searches(search_queries)
            
            # Generate research summary
            research_content = self._generate_research_summary(question, search_results, documents)
            
            # Add expert commentary and analysis sources
            enhanced_content = self._add_expert_sources(research_content, question)
            enhanced_content = self._add_regulatory_resources(enhanced_content, self.current_proceeding)
            
            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence_score = self._calculate_research_confidence(search_results, enhanced_content)
            sources_used = [doc.metadata.get('source', 'Unknown') for doc in documents]
            
            return AgentResponse(
                content=enhanced_content,
                agent_type="Further Sources Researcher",
                confidence_score=confidence_score,
                sources_used=sources_used,
                processing_time=processing_time,
                metadata={
                    "search_queries_performed": len(search_queries),
                    "web_sources_found": len(search_results),
                    "external_links_provided": self._count_external_links(enhanced_content),
                    "resource_categories": self._identify_resource_categories(enhanced_content),
                    "proceeding_context": self.current_proceeding
                }
            )
            
        except Exception as e:
            logger.error(f"Further sources researcher agent failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=f"Additional research sources could not be generated due to processing error: {str(e)}",
                agent_type="Further Sources Researcher",
                confidence_score=0.0,
                sources_used=[],
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _generate_search_queries(self, question: str, proceeding: str) -> List[str]:
        """Generate targeted search queries for web research."""
        base_queries = []
        
        # Primary query with proceeding context
        proceeding_formatted = config.format_proceeding_for_search(proceeding)
        base_queries.append(f"CPUC {proceeding_formatted} {question}")
        
        # News and recent developments
        base_queries.append(f"CPUC {proceeding_formatted} news recent developments")
        
        # Industry analysis
        base_queries.append(f"California utility regulation {question} analysis")
        
        # Extract key terms from question for targeted searches
        key_terms = self._extract_key_terms(question)
        if key_terms:
            base_queries.append(f"CPUC {' '.join(key_terms[:3])} industry impact")
        
        # Stakeholder perspectives
        base_queries.append(f"utility stakeholders {question} California")
        
        return base_queries[:4]  # Limit to avoid excessive searches
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from the question for search optimization."""
        # Remove common words and extract meaningful terms
        stop_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off',
            'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        words = re.findall(r'\b\w{4,}\b', question.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        return key_terms[:5]  # Return top 5 key terms
    
    def _perform_web_searches(self, queries: List[str]) -> List[Dict]:
        """Perform web searches and collect results."""
        all_results = []
        
        for query in queries:
            try:
                logger.info(f"Searching for: {query}")
                results = self.search_tool.run(query)
                
                # Parse DuckDuckGo results (they come as a formatted string)
                if isinstance(results, str):
                    # Extract structured information from the results string
                    parsed_results = self._parse_search_results(results, query)
                    all_results.extend(parsed_results)
                
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue
        
        return all_results[:10]  # Limit total results
    
    def _parse_search_results(self, results_string: str, query: str) -> List[Dict]:
        """Parse DuckDuckGo search results string into structured data."""
        parsed_results = []
        
        # Split results by line and extract information
        lines = results_string.split('\n')
        current_result = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_result:
                    current_result['query'] = query
                    parsed_results.append(current_result)
                    current_result = {}
                continue
            
            # Look for URL patterns
            if line.startswith('http'):
                current_result['url'] = line
            # Look for title patterns (usually the first non-URL line)
            elif 'title' not in current_result and line:
                current_result['title'] = line[:100]  # Truncate long titles
            # Everything else is snippet/description
            elif line and 'snippet' not in current_result:
                current_result['snippet'] = line[:200]  # Truncate long snippets
        
        # Add the last result if it exists
        if current_result:
            current_result['query'] = query
            parsed_results.append(current_result)
        
        return parsed_results
    
    def _generate_research_summary(self, question: str, search_results: List[Dict], documents: List[Document]) -> str:
        """Generate a comprehensive research summary."""
        if not search_results:
            return self._generate_fallback_resources(question)
        
        # Create context from search results
        search_context = "\n".join([
            f"Source: {result.get('title', 'Unknown')}\nURL: {result.get('url', 'N/A')}\nSummary: {result.get('snippet', 'No summary available')}\n"
            for result in search_results
        ])
        
        # Generate summary using LLM
        prompt = self.prompt_template.format(
            question=question,
            search_results=search_context,
            proceeding=self.current_proceeding,
            documents_context=f"Primary analysis based on {len(documents)} regulatory documents"
        )
        
        response_message = self.llm.invoke(prompt)
        return response_message.content
    
    def _generate_fallback_resources(self, question: str) -> str:
        """Generate fallback resources when web search fails."""
        proceeding_formatted = config.format_proceeding_for_search(self.current_proceeding)
        
        return f"""<h4>Additional Research Resources</h4>
        
<p>While current web search results are not available, here are key resources for further research on this topic:</p>

<h5>Official CPUC Resources:</h5>
<ul>
<li><strong>CPUC Proceeding Page:</strong> <a href="https://apps.cpuc.ca.gov/apex/f?p=401:56:::NO:RP,57,RIR:P5_PROCEEDING_SELECT:{self.current_proceeding}" target="_blank">{proceeding_formatted} Official Page</a></li>
<li><strong>CPUC Document Search:</strong> <a href="https://docs.cpuc.ca.gov/SearchRes.aspx" target="_blank">Search CPUC Documents</a></li>
<li><strong>CPUC Decisions Database:</strong> <a href="https://docs.cpuc.ca.gov/DecisionsIndex.aspx" target="_blank">Recent Decisions and Orders</a></li>
</ul>

<h5>Industry Analysis Sources:</h5>
<ul>
<li><strong>California Energy Commission:</strong> <a href="https://www.energy.ca.gov/" target="_blank">CEC Official Website</a></li>
<li><strong>California ISO:</strong> <a href="https://www.caiso.com/" target="_blank">Grid Operations and Market Data</a></li>
<li><strong>Public Advocates Office:</strong> <a href="https://www.publicadvocates.cpuc.ca.gov/" target="_blank">Consumer Advocacy Perspective</a></li>
</ul>

<p><em>Note: For the most current information, please check these official sources directly.</em></p>"""
    
    def _add_expert_sources(self, content: str, question: str) -> str:
        """Add expert commentary and analysis sources."""
        expert_sources = {
            'energy': [
                "Stanford Energy Corporate Affiliates Program",
                "UC Berkeley Energy Institute",
                "RAND Corporation Energy and Environment Research"
            ],
            'regulation': [
                "National Association of Regulatory Utility Commissioners (NARUC)",
                "Energy Law Journal",
                "Utility Regulatory Policy"
            ],
            'utility': [
                "American Gas Association (AGA)",
                "Edison Electric Institute (EEI)",
                "Solar Energy Industries Association (SEIA)"
            ]
        }
        
        # Determine relevant categories based on question content
        relevant_categories = []
        question_lower = question.lower()
        
        if any(term in question_lower for term in ['energy', 'power', 'electricity', 'solar', 'renewable']):
            relevant_categories.append('energy')
        
        if any(term in question_lower for term in ['regulation', 'rule', 'policy', 'compliance']):
            relevant_categories.append('regulation')
        
        if any(term in question_lower for term in ['utility', 'rate', 'tariff', 'billing']):
            relevant_categories.append('utility')
        
        if not relevant_categories:
            relevant_categories = ['regulation']  # Default fallback
        
        # Add expert sources section
        expert_section = "\n\n<h5>Expert Analysis and Commentary Sources:</h5>\n<ul>"
        
        for category in relevant_categories:
            if category in expert_sources:
                for source in expert_sources[category][:2]:  # Limit to 2 per category
                    expert_section += f"<li><strong>{source}:</strong> Provides expert analysis on regulatory and policy matters</li>"
        
        expert_section += "</ul>"
        
        return content + expert_section
    
    def _add_regulatory_resources(self, content: str, proceeding: str) -> str:
        """Add proceeding-specific regulatory resources."""
        proceeding_urls = config.get_proceeding_urls(proceeding)
        
        regulatory_section = f"""
<h5>Official Regulatory Resources:</h5>
<ul>
<li><strong>CPUC Proceeding Portal:</strong> <a href="{proceeding_urls['cpuc_apex']}" target="_blank">Official {proceeding} Page</a></li>
<li><strong>Document Search:</strong> <a href="{proceeding_urls['cpuc_search']}" target="_blank">Search {proceeding} Documents</a></li>
<li><strong>Public Participation:</strong> <a href="https://www.cpuc.ca.gov/about-cpuc/public-participation" target="_blank">How to Participate in CPUC Proceedings</a></li>
<li><strong>Filing Requirements:</strong> <a href="https://www.cpuc.ca.gov/about-cpuc/divisions/office-of-ratepayer-advocates" target="_blank">Ratepayer Advocate Resources</a></li>
</ul>

<h5>Related Proceedings and Cross-References:</h5>
<p><em>Monitor related proceedings for comprehensive understanding of regulatory landscape.</em></p>
"""
        
        return content + regulatory_section
    
    def _calculate_research_confidence(self, search_results: List[Dict], content: str) -> float:
        """Calculate confidence score for research quality."""
        factors = []
        
        # Number of sources factor
        sources_factor = min(len(search_results) / 5.0, 1.0)
        factors.append(sources_factor)
        
        # Content comprehensiveness (length and structure)
        content_factor = min(len(content) / 1500.0, 1.0)
        factors.append(content_factor)
        
        # Link/resource availability
        link_count = content.count('href=')
        link_factor = min(link_count / 8.0, 1.0)
        factors.append(link_factor)
        
        # Official source preference
        official_sources = ['cpuc.ca.gov', 'energy.ca.gov', 'caiso.com']
        official_count = sum(1 for result in search_results 
                           if any(official in result.get('url', '') for official in official_sources))
        official_factor = min(official_count / 2.0, 1.0)
        factors.append(official_factor)
        
        return sum(factors) / len(factors)
    
    def _count_external_links(self, content: str) -> int:
        """Count external links in the content."""
        return content.count('href=')
    
    def _identify_resource_categories(self, content: str) -> List[str]:
        """Identify categories of resources provided."""
        categories = []
        content_lower = content.lower()
        
        if 'official' in content_lower or 'cpuc' in content_lower:
            categories.append('official_regulatory')
        
        if 'expert' in content_lower or 'analysis' in content_lower:
            categories.append('expert_analysis')
        
        if 'news' in content_lower or 'recent' in content_lower:
            categories.append('news_updates')
        
        if 'industry' in content_lower or 'stakeholder' in content_lower:
            categories.append('industry_perspective')
        
        return categories


# Agent Registry for easy access
AGENT_REGISTRY = {
    'technical_expert': TechnicalIndustryExpertAgent,
    'laymen_interpreter': LaymenInterpretationAgent,
    'further_sources': FurtherSourcesResearcherAgent
}


def get_agent(agent_type: str, llm, current_proceeding: str = None) -> BaseResponseAgent:
    """Factory function to get specific agent instances."""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}")
    
    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class(llm, current_proceeding)


def generate_multi_agent_response(question: str, documents: List[Document], llm, 
                                current_proceeding: str = None, 
                                enabled_agents: List[str] = None) -> Dict[str, AgentResponse]:
    """
    Generate responses from multiple specialized agents.
    
    Args:
        question: User's question
        documents: Retrieved documents for context
        llm: Language model instance
        current_proceeding: Current proceeding context
        enabled_agents: List of agent types to enable (default: all)
    
    Returns:
        Dictionary mapping agent types to their responses
    """
    if enabled_agents is None:
        enabled_agents = list(AGENT_REGISTRY.keys())
    
    responses = {}
    
    # Generate technical response first (used by other agents)
    technical_agent = get_agent('technical_expert', llm, current_proceeding)
    technical_response = technical_agent.generate_response(question, documents)
    responses['technical_expert'] = technical_response
    
    # Generate laymen interpretation using technical response
    if 'laymen_interpreter' in enabled_agents:
        laymen_agent = get_agent('laymen_interpreter', llm, current_proceeding)
        laymen_response = laymen_agent.generate_response(question, documents, technical_response.content)
        responses['laymen_interpreter'] = laymen_response
    
    # Generate further sources research
    if 'further_sources' in enabled_agents:
        research_agent = get_agent('further_sources', llm, current_proceeding)
        research_response = research_agent.generate_response(question, documents, technical_response.content)
        responses['further_sources'] = research_response
    
    return responses