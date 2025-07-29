# 📁 agents/agent_tools.py
# Agent Tools Interface for CPUC Regulatory Search Engine

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from langchain.docstore.document import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.vectorstores import LanceDB
import requests
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from any search tool."""
    content: str
    source: str
    title: str = ""
    url: str = ""
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_document(self) -> Document:
        """Convert to LangChain Document."""
        return Document(
            page_content=self.content,
            metadata={
                "source": self.source,
                "title": self.title,
                "url": self.url,
                "page": self.page_number,
                "relevance_score": self.relevance_score,
                "timestamp": self.timestamp.isoformat(),
                **self.metadata
            }
        )


@dataclass
class CitationInfo:
    """Represents citation information extracted from documents."""
    document_id: str
    page_number: int
    text_snippet: str
    confidence: float
    context_before: str = ""
    context_after: str = ""
    regulatory_reference: Optional[str] = None
    date_referenced: Optional[datetime] = None
    
    def format_citation(self, style: str = "cpuc", char_start: int = None, char_end: int = None, 
                       line_number: int = None, text_snippet: str = None) -> str:
        """Format citation according to specified style with enhanced precision."""
        if style == "cpuc":
            citation_parts = [f"CITE:{self.document_id}", f"page_{self.page_number}"]
            
            # Add character positions if available
            if char_start is not None and char_end is not None:
                citation_parts.append(f"chars_{char_start}-{char_end}")
            
            # Add line number if available
            if line_number is not None:
                citation_parts.append(f"line_{line_number}")
            
            # Add text snippet for verification if available
            if text_snippet:
                snippet = text_snippet[:50].replace('"', "'").replace('\n', ' ').strip()
                citation_parts.append(f'"{snippet}..."')
            
            return f"[{','.join(citation_parts)}]"
            
        elif style == "academic":
            return f"({self.document_id}, p. {self.page_number})"
        else:
            return f"{self.document_id}:{self.page_number}"


class BaseTool(ABC):
    """Abstract base class for all agent tools."""
    
    def __init__(self, name: str, description: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base tool.
        
        Args:
            name: Tool name
            description: Tool description
            config: Tool-specific configuration
        """
        self.name = name
        self.description = description
        self.config = config or {}
        self.usage_count = 0
        self.error_count = 0
        self.last_used = None
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        pass
    
    def _log_usage(self, success: bool = True) -> None:
        """Log tool usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.now()
        if not success:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.usage_count),
            "last_used": self.last_used.isoformat() if self.last_used else None
        }


class VectorStoreTool(BaseTool):
    """Tool for searching and retrieving documents from the vector store."""
    
    def __init__(self, vector_store: LanceDB, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the vector store tool.
        
        Args:
            vector_store: LanceDB vector store instance
            config: Tool configuration
        """
        super().__init__(
            name="vector_store_search",
            description="Search documents in the regulatory vector database",
            config=config
        )
        self.vector_store = vector_store
        self.default_k = self.config.get("default_k", 10)
        self.max_k = self.config.get("max_k", 50)
    
    def execute(
        self, 
        query: str, 
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "similarity"
    ) -> List[SearchResult]:
        """
        Execute vector store search.
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Metadata filters
            search_type: Type of search (similarity, mmr, etc.)
            
        Returns:
            List of search results
        """
        try:
            k = min(k or self.default_k, self.max_k)
            
            if search_type == "similarity":
                documents = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filters
                )
            elif search_type == "similarity_with_score":
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filters
                )
                documents = [doc for doc, score in docs_with_scores]
                # Add scores to metadata
                for i, (doc, score) in enumerate(docs_with_scores):
                    documents[i].metadata["relevance_score"] = float(score)
            else:
                # Default to similarity search
                documents = self.vector_store.similarity_search(query=query, k=k)
            
            # Convert to SearchResult objects
            results = []
            for doc in documents:
                result = SearchResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    title=doc.metadata.get("title", ""),
                    url=doc.metadata.get("source_url", ""),
                    page_number=doc.metadata.get("page"),
                    relevance_score=doc.metadata.get("relevance_score", 0.0),
                    metadata=doc.metadata
                )
                results.append(result)
            
            self._log_usage(success=True)
            logger.info(f"Vector search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self._log_usage(success=False)
            logger.error(f"Vector store search failed: {e}")
            return []
    
    def get_document_by_id(self, document_id: str) -> Optional[SearchResult]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            SearchResult if found, None otherwise
        """
        try:
            # Search for document with specific ID in metadata
            documents = self.vector_store.similarity_search(
                query="",  # Empty query to get any documents
                k=1000,  # Get a large number to search through
                filter={"chunk_id": document_id}
            )
            
            if documents:
                doc = documents[0]
                result = SearchResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    title=doc.metadata.get("title", ""),
                    url=doc.metadata.get("source_url", ""),
                    page_number=doc.metadata.get("page"),
                    relevance_score=1.0,  # Perfect match
                    metadata=doc.metadata
                )
                self._log_usage(success=True)
                return result
            
            return None
            
        except Exception as e:
            self._log_usage(success=False)
            logger.error(f"Document retrieval failed for ID {document_id}: {e}")
            return None


class WebSearchTool(BaseTool):
    """Tool for searching the web for additional regulatory information."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the web search tool.
        
        Args:
            config: Tool configuration
        """
        super().__init__(
            name="web_search",
            description="Search the web for regulatory information",
            config=config
        )
        self.search_engine = DuckDuckGoSearchResults(
            num_results=self.config.get("max_results", 10)
        )
        self.cpuc_domains = [
            "docs.cpuc.ca.gov",
            "apps.cpuc.ca.gov", 
            "www.cpuc.ca.gov"
        ]
    
    def execute(
        self, 
        query: str, 
        site_filter: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Execute web search.
        
        Args:
            query: Search query
            site_filter: Specific site to search within
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Modify query for site filtering
            if site_filter:
                search_query = f"site:{site_filter} {query}"
            else:
                search_query = query
            
            # Add CPUC-specific terms to improve relevance
            if "cpuc" not in search_query.lower():
                search_query = f"CPUC {search_query}"
            
            # Execute search
            raw_results = self.search_engine.run(search_query)
            
            # Parse results (DuckDuckGo returns string format)
            results = []
            if isinstance(raw_results, str):
                # Parse the string format
                result_blocks = self._parse_duckduckgo_results(raw_results)
                
                for block in result_blocks:
                    result = SearchResult(
                        content=block.get("snippet", ""),
                        source="Web Search",
                        title=block.get("title", ""),
                        url=block.get("link", ""),
                        relevance_score=self._calculate_cpuc_relevance(block),
                        metadata={
                            "search_engine": "DuckDuckGo",
                            "query": search_query
                        }
                    )
                    results.append(result)
            
            # Sort by CPUC relevance
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Limit results
            if max_results:
                results = results[:max_results]
            
            self._log_usage(success=True)
            logger.info(f"Web search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self._log_usage(success=False)
            logger.error(f"Web search failed: {e}")
            return []
    
    def _parse_duckduckgo_results(self, raw_results: str) -> List[Dict[str, str]]:
        """Parse DuckDuckGo search results from string format."""
        results = []
        
        # Split by result blocks (typically separated by newlines)
        blocks = raw_results.split('\n\n')
        
        for block in blocks:
            if block.strip():
                # Extract title, link, and snippet using regex
                title_match = re.search(r'title: (.+)', block, re.IGNORECASE)
                link_match = re.search(r'link: (.+)', block, re.IGNORECASE)
                snippet_match = re.search(r'snippet: (.+)', block, re.IGNORECASE)
                
                result = {
                    "title": title_match.group(1).strip() if title_match else "",
                    "link": link_match.group(1).strip() if link_match else "",
                    "snippet": snippet_match.group(1).strip() if snippet_match else ""
                }
                
                if result["title"] or result["snippet"]:
                    results.append(result)
        
        return results
    
    def _calculate_cpuc_relevance(self, result: Dict[str, str]) -> float:
        """Calculate relevance score based on CPUC-specific criteria."""
        score = 0.0
        
        url = result.get("link", "").lower()
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        
        # High score for CPUC official domains
        for domain in self.cpuc_domains:
            if domain in url:
                score += 0.8
                break
        
        # Score for regulatory keywords
        regulatory_keywords = [
            "decision", "ruling", "proceeding", "rulemaking", "order",
            "application", "petition", "resolution", "advice letter"
        ]
        
        text_content = f"{title} {snippet}".lower()
        for keyword in regulatory_keywords:
            if keyword in text_content:
                score += 0.1
        
        # Score for proceeding number patterns
        if re.search(r'r\.\d{2}-\d{2}-\d{3}', text_content):
            score += 0.3
        
        return min(score, 1.0)
    
    def search_cpuc_specific(self, query: str, proceeding: Optional[str] = None) -> List[SearchResult]:
        """
        Search specifically within CPUC domains.
        
        Args:
            query: Search query
            proceeding: Specific proceeding number to include
            
        Returns:
            List of CPUC-specific search results
        """
        if proceeding:
            query = f"{query} {proceeding}"
        
        results = []
        
        # Search each CPUC domain
        for domain in self.cpuc_domains:
            domain_results = self.execute(query, site_filter=domain, max_results=5)
            results.extend(domain_results)
        
        # Remove duplicates and sort by relevance
        unique_results = []
        seen_urls = set()
        
        for result in results:
            if result.url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result.url)
        
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return unique_results


class CitationTool(BaseTool):
    """Tool for analyzing and managing citations in regulatory documents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the citation tool.
        
        Args:
            config: Tool configuration
        """
        super().__init__(
            name="citation_analysis",
            description="Analyze and manage document citations",
            config=config
        )
        self.citation_patterns = [
            # Enhanced citation format with character positions and text snippets
            r"\[CITE:([^,]+),\s*page_(\d+),\s*chars_(\d+)-(\d+),\s*line_(\d+),\s*\"([^\"]*)\"\]",
            r"\[CITE:([^,]+),\s*page_(\d+),\s*chars_(\d+)-(\d+),\s*line_(\d+)\]",
            r"\[CITE:([^,]+),\s*page_(\d+),\s*chars_(\d+)-(\d+)\]",
            r"\[CITE:([^,]+),\s*page_(\d+),\s*line_(\d+)\]",  # Legacy with line
            r"\[CITE:([^,]+),\s*page_(\d+)\]",  # Legacy format
            r"Decision\s+(\d{2}-\d{2}-\d{3})",  # CPUC Decision format
            r"Resolution\s+([A-Z]-\d+)",  # CPUC Resolution format
            r"Advice Letter\s+(\d+[-A-Z]*)",  # Advice Letter format
            r"R\.(\d{2}-\d{2}-\d{3})",  # Rulemaking format
            r"A\.(\d{2}-\d{2}-\d{3})",  # Application format
        ]
    
    def execute(
        self, 
        text: str, 
        extract_type: str = "all"
    ) -> List[CitationInfo]:
        """
        Extract citations from text.
        
        Args:
            text: Text to analyze
            extract_type: Type of citations to extract (all, custom, regulatory)
            
        Returns:
            List of extracted citation information
        """
        try:
            citations = []
            
            if extract_type in ["all", "custom"]:
                citations.extend(self._extract_custom_citations(text))
            
            if extract_type in ["all", "regulatory"]:
                citations.extend(self._extract_regulatory_citations(text))
            
            self._log_usage(success=True)
            logger.info(f"Extracted {len(citations)} citations from text")
            return citations
            
        except Exception as e:
            self._log_usage(success=False)
            logger.error(f"Citation extraction failed: {e}")
            return []
    
    def _extract_custom_citations(self, text: str) -> List[CitationInfo]:
        """Extract custom format citations [CITE:filename,page_X]."""
        citations = []
        pattern = r"\[CITE:([^,]+),\s*page_(\d+)\]"
        
        for match in re.finditer(pattern, text):
            filename = match.group(1).strip()
            page_num = int(match.group(2))
            
            # Get context around citation
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(text), match.end() + 100)
            context = text[start_pos:end_pos]
            
            citation = CitationInfo(
                document_id=filename,
                page_number=page_num,
                text_snippet=match.group(0),
                confidence=0.9,  # High confidence for exact format match
                context_before=text[start_pos:match.start()],
                context_after=text[match.end():end_pos]
            )
            citations.append(citation)
        
        return citations
    
    def _extract_regulatory_citations(self, text: str) -> List[CitationInfo]:
        """Extract regulatory citations (Decisions, Resolutions, etc.)."""
        citations = []
        
        # Decision pattern
        decision_pattern = r"Decision\s+(\d{2}-\d{2}-\d{3})"
        for match in re.finditer(decision_pattern, text, re.IGNORECASE):
            citation = CitationInfo(
                document_id=f"D.{match.group(1)}",
                page_number=0,  # Page not specified in reference
                text_snippet=match.group(0),
                confidence=0.8,
                regulatory_reference="Decision"
            )
            citations.append(citation)
        
        # Rulemaking pattern
        rulemaking_pattern = r"R\.(\d{2}-\d{2}-\d{3})"
        for match in re.finditer(rulemaking_pattern, text):
            citation = CitationInfo(
                document_id=f"R.{match.group(1)}",
                page_number=0,
                text_snippet=match.group(0),
                confidence=0.8,
                regulatory_reference="Rulemaking"
            )
            citations.append(citation)
        
        return citations
    
    def validate_citations(
        self, 
        citations: List[CitationInfo], 
        vector_store_tool: VectorStoreTool
    ) -> List[CitationInfo]:
        """
        Validate citations against the vector store.
        
        Args:
            citations: List of citations to validate
            vector_store_tool: Vector store tool for document lookup
            
        Returns:
            List of validated citations with updated confidence scores
        """
        validated_citations = []
        
        for citation in citations:
            # Try to find the document in vector store
            doc_result = vector_store_tool.get_document_by_id(citation.document_id)
            
            if doc_result:
                # Document exists, increase confidence
                citation.confidence = min(citation.confidence + 0.1, 1.0)
                validated_citations.append(citation)
            else:
                # Document not found, decrease confidence
                citation.confidence = max(citation.confidence - 0.2, 0.1)
                # Still include but mark as potentially invalid
                validated_citations.append(citation)
        
        return validated_citations
    
    def format_citation_list(self, citations: List[CitationInfo], style: str = "cpuc") -> str:
        """
        Format a list of citations into a readable format.
        
        Args:
            citations: List of citations to format
            style: Citation style (cpuc, academic, apa)
            
        Returns:
            Formatted citation list
        """
        if not citations:
            return "No citations found."
        
        formatted = []
        for i, citation in enumerate(citations, 1):
            if style == "cpuc":
                formatted.append(f"{i}. {citation.format_citation('cpuc')} - {citation.text_snippet}")
            elif style == "academic":
                formatted.append(f"{i}. {citation.format_citation('academic')}")
            else:
                formatted.append(f"{i}. {citation.document_id}, Page {citation.page_number}")
        
        return "\n".join(formatted)


class ToolRegistry:
    """Registry for managing and accessing all agent tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, BaseTool] = {}
        self._tool_usage_stats: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: Tool to register
        """
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools."""
        return self.tools.copy()
    
    def get_tools_by_capability(self, capability: str) -> List[BaseTool]:
        """
        Get tools that support a specific capability.
        
        Args:
            capability: Capability name
            
        Returns:
            List of matching tools
        """
        matching_tools = []
        
        capability_mapping = {
            "document_search": ["vector_store_search"],
            "web_search": ["web_search"],
            "citation_analysis": ["citation_analysis"]
        }
        
        tool_names = capability_mapping.get(capability, [])
        
        for tool_name in tool_names:
            tool = self.get_tool(tool_name)
            if tool:
                matching_tools.append(tool)
        
        return matching_tools
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all tools."""
        stats = {
            "total_tools": len(self.tools),
            "tool_stats": {}
        }
        
        for tool_name, tool in self.tools.items():
            stats["tool_stats"][tool_name] = tool.get_stats()
        
        return stats
    
    def create_default_tools(
        self, 
        vector_store: Optional[LanceDB] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create and register default tools.
        
        Args:
            vector_store: Vector store instance for document search
            config: Global configuration for tools
        """
        tool_config = config or {}
        
        # Register vector store tool if available
        if vector_store:
            vector_tool = VectorStoreTool(
                vector_store=vector_store,
                config=tool_config.get("vector_store", {})
            )
            self.register_tool(vector_tool)
        
        # Register web search tool
        web_tool = WebSearchTool(config=tool_config.get("web_search", {}))
        self.register_tool(web_tool)
        
        # Register citation tool
        citation_tool = CitationTool(config=tool_config.get("citation", {}))
        self.register_tool(citation_tool)
        
        logger.info("Default tools registered successfully")