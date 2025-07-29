# 📁 multi_agent_system.py
# Multi-agent system for enhanced CPUC regulatory analysis

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.docstore.document import Document

import config

logger = logging.getLogger(__name__)


class AgentType(Enum):
    QUESTION_INTERPRETATION = "question_interpretation"
    TECHNICAL_ANALYSIS = "technical_analysis"
    LAYMEN_EXPLANATION = "laymen_explanation"
    SOURCES_ANALYSIS = "sources_analysis"
    RESPONSE_SYNTHESIS = "response_synthesis"


@dataclass
class AgentResponse:
    """Container for individual agent responses."""
    agent_type: AgentType
    content: str
    processing_time: float
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


@dataclass
class MultiAgentResult:
    """Container for complete multi-agent analysis results."""
    question: str
    agent_responses: Dict[AgentType, AgentResponse]
    final_response: str
    confidence_score: float
    processing_metrics: Dict[str, Any]
    citations: List[Dict[str, str]]
    timestamp: datetime


class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents for comprehensive CPUC regulatory analysis.
    
    This system coordinates:
    1. Question Interpretation Agent - Analyzes and enhances user queries
    2. Technical Analysis Agent - Provides detailed regulatory analysis
    3. Laymen Explanation Agent - Translates technical content to accessible language
    4. Sources Agent - Validates and curates source information
    5. Response Synthesis Agent - Integrates all outputs into cohesive response
    """
    
    def __init__(self, llm, embedding_model=None):
        self.llm = llm
        self.embedding_model = embedding_model
        
        # Initialize agent prompts
        self.agent_prompts = {
            AgentType.QUESTION_INTERPRETATION: PromptTemplate.from_template(
                config.QUESTION_INTERPRETATION_AGENT_TEMPLATE
            ),
            AgentType.TECHNICAL_ANALYSIS: PromptTemplate.from_template(
                config.TECHNICAL_ANALYSIS_AGENT_TEMPLATE
            ),
            AgentType.LAYMEN_EXPLANATION: PromptTemplate.from_template(
                config.LAYMEN_EXPLANATION_AGENT_TEMPLATE
            ),
            AgentType.SOURCES_ANALYSIS: PromptTemplate.from_template(
                config.SOURCES_AGENT_TEMPLATE
            ),
            AgentType.RESPONSE_SYNTHESIS: PromptTemplate.from_template(
                config.RESPONSE_SYNTHESIS_AGENT_TEMPLATE
            )
        }
        
        # Initialize output parsers
        self.json_parser = JsonOutputParser()
        self.string_parser = StrOutputParser()
        
        logger.info("MultiAgentOrchestrator initialized with all agents")
    
    async def process_query(
        self, 
        question: str, 
        retrieved_docs: List[Document],
        enable_agents: Optional[Dict[str, bool]] = None
    ) -> MultiAgentResult:
        """
        Process a query through the multi-agent system.
        
        Args:
            question: User question to analyze
            retrieved_docs: Documents retrieved from vector store
            enable_agents: Dict to enable/disable specific agents
            
        Returns:
            MultiAgentResult containing all agent outputs and synthesis
        """
        start_time = time.time()
        
        # Use default agent configuration if not provided
        if enable_agents is None:
            enable_agents = config.AGENT_RESPONSE_CONFIG
        
        agent_responses = {}
        processing_metrics = {
            "total_agents": 0,
            "successful_agents": 0,
            "failed_agents": 0,
            "agent_times": {}
        }
        
        try:
            # Step 1: Question Interpretation
            if enable_agents.get("enable_question_interpretation", True):
                query_analysis = await self._run_question_interpretation_agent(question)
                agent_responses[AgentType.QUESTION_INTERPRETATION] = query_analysis
                processing_metrics["agent_times"]["question_interpretation"] = query_analysis.processing_time
                processing_metrics["total_agents"] += 1
                if query_analysis.success:
                    processing_metrics["successful_agents"] += 1
                else:
                    processing_metrics["failed_agents"] += 1
            else:
                # Create default query analysis if agent is disabled
                query_analysis = AgentResponse(
                    agent_type=AgentType.QUESTION_INTERPRETATION,
                    content='{"enhanced_query": "' + question + '", "intent_classification": "general_inquiry"}',
                    processing_time=0.0,
                    metadata={},
                    success=True
                )
            
            # Step 2: Technical Analysis
            if enable_agents.get("enable_technical_analysis", True):
                technical_analysis = await self._run_technical_analysis_agent(
                    question, query_analysis.content, retrieved_docs
                )
                agent_responses[AgentType.TECHNICAL_ANALYSIS] = technical_analysis
                processing_metrics["agent_times"]["technical_analysis"] = technical_analysis.processing_time
                processing_metrics["total_agents"] += 1
                if technical_analysis.success:
                    processing_metrics["successful_agents"] += 1
                else:
                    processing_metrics["failed_agents"] += 1
            else:
                technical_analysis = AgentResponse(
                    agent_type=AgentType.TECHNICAL_ANALYSIS,
                    content="Technical analysis disabled",
                    processing_time=0.0,
                    metadata={},
                    success=False,
                    error_message="Agent disabled"
                )
            
            # Step 3: Laymen Explanation
            if enable_agents.get("enable_laymen_explanation", True) and technical_analysis.success:
                laymen_explanation = await self._run_laymen_explanation_agent(
                    question, technical_analysis.content
                )
                agent_responses[AgentType.LAYMEN_EXPLANATION] = laymen_explanation
                processing_metrics["agent_times"]["laymen_explanation"] = laymen_explanation.processing_time
                processing_metrics["total_agents"] += 1
                if laymen_explanation.success:
                    processing_metrics["successful_agents"] += 1
                else:
                    processing_metrics["failed_agents"] += 1
            else:
                laymen_explanation = AgentResponse(
                    agent_type=AgentType.LAYMEN_EXPLANATION,
                    content="Laymen explanation not available",
                    processing_time=0.0,
                    metadata={},
                    success=False,
                    error_message="Agent disabled or technical analysis failed"
                )
            
            # Step 4: Sources Analysis
            if enable_agents.get("enable_sources_agent", True):
                sources_analysis = await self._run_sources_analysis_agent(
                    question, retrieved_docs
                )
                agent_responses[AgentType.SOURCES_ANALYSIS] = sources_analysis
                processing_metrics["agent_times"]["sources_analysis"] = sources_analysis.processing_time
                processing_metrics["total_agents"] += 1
                if sources_analysis.success:
                    processing_metrics["successful_agents"] += 1
                else:
                    processing_metrics["failed_agents"] += 1
            else:
                sources_analysis = AgentResponse(
                    agent_type=AgentType.SOURCES_ANALYSIS,
                    content="Sources analysis disabled",
                    processing_time=0.0,
                    metadata={},
                    success=False,
                    error_message="Agent disabled"
                )
            
            # Step 5: Response Synthesis
            if enable_agents.get("enable_response_synthesis", True):
                synthesis_response = await self._run_response_synthesis_agent(
                    query_analysis.content if query_analysis.success else "{}",
                    technical_analysis.content if technical_analysis.success else "",
                    laymen_explanation.content if laymen_explanation.success else "",
                    sources_analysis.content if sources_analysis.success else ""
                )
                agent_responses[AgentType.RESPONSE_SYNTHESIS] = synthesis_response
                processing_metrics["agent_times"]["response_synthesis"] = synthesis_response.processing_time
                processing_metrics["total_agents"] += 1
                if synthesis_response.success:
                    processing_metrics["successful_agents"] += 1
                    final_response = synthesis_response.content
                else:
                    processing_metrics["failed_agents"] += 1
                    final_response = self._create_fallback_response(
                        technical_analysis.content if technical_analysis.success else "",
                        laymen_explanation.content if laymen_explanation.success else ""
                    )
            else:
                final_response = self._create_fallback_response(
                    technical_analysis.content if technical_analysis.success else "",
                    laymen_explanation.content if laymen_explanation.success else ""
                )
            
            # Calculate final metrics
            total_time = time.time() - start_time
            processing_metrics["total_processing_time"] = total_time
            processing_metrics["success_rate"] = (
                processing_metrics["successful_agents"] / processing_metrics["total_agents"]
                if processing_metrics["total_agents"] > 0 else 0
            )
            
            # Extract citations from final response
            citations = self._extract_citations(final_response)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(agent_responses, processing_metrics)
            
            return MultiAgentResult(
                question=question,
                agent_responses=agent_responses,
                final_response=final_response,
                confidence_score=confidence_score,
                processing_metrics=processing_metrics,
                citations=citations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Multi-agent processing failed: {e}")
            
            # Return fallback result
            return MultiAgentResult(
                question=question,
                agent_responses=agent_responses,
                final_response=f"Multi-agent analysis encountered an error: {str(e)}",
                confidence_score=0.0,
                processing_metrics=processing_metrics,
                citations=[],
                timestamp=datetime.now()
            )
    
    async def _run_question_interpretation_agent(self, question: str) -> AgentResponse:
        """Run the Question Interpretation Agent."""
        start_time = time.time()
        
        try:
            prompt = self.agent_prompts[AgentType.QUESTION_INTERPRETATION]
            chain = prompt | self.llm | self.string_parser
            
            response = chain.invoke({"question": question})
            
            # Try to parse as JSON, fallback to string if needed
            try:
                parsed_response = json.loads(response)
                content = json.dumps(parsed_response, indent=2)
            except json.JSONDecodeError:
                logger.warning("Question interpretation response not valid JSON, using as string")
                content = response
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                agent_type=AgentType.QUESTION_INTERPRETATION,
                content=content,
                processing_time=processing_time,
                metadata={"response_type": "json_analysis"},
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Question interpretation agent failed: {e}")
            
            return AgentResponse(
                agent_type=AgentType.QUESTION_INTERPRETATION,
                content="",
                processing_time=processing_time,
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    async def _run_technical_analysis_agent(
        self, 
        question: str, 
        query_analysis: str, 
        retrieved_docs: List[Document]
    ) -> AgentResponse:
        """Run the Technical Analysis Agent."""
        start_time = time.time()
        
        try:
            # Format context from retrieved documents
            context = self._format_context_for_agent(retrieved_docs, question)
            
            prompt = self.agent_prompts[AgentType.TECHNICAL_ANALYSIS]
            chain = prompt | self.llm | self.string_parser
            
            response = chain.invoke({
                "query_analysis": query_analysis,
                "context": context,
                "question": question
            })
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                agent_type=AgentType.TECHNICAL_ANALYSIS,
                content=response,
                processing_time=processing_time,
                metadata={
                    "context_length": len(context),
                    "document_count": len(retrieved_docs)
                },
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Technical analysis agent failed: {e}")
            
            return AgentResponse(
                agent_type=AgentType.TECHNICAL_ANALYSIS,
                content="",
                processing_time=processing_time,
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    async def _run_laymen_explanation_agent(
        self, 
        question: str, 
        technical_analysis: str
    ) -> AgentResponse:
        """Run the Laymen Explanation Agent."""
        start_time = time.time()
        
        try:
            prompt = self.agent_prompts[AgentType.LAYMEN_EXPLANATION]
            chain = prompt | self.llm | self.string_parser
            
            response = chain.invoke({
                "technical_analysis": technical_analysis,
                "question": question
            })
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                agent_type=AgentType.LAYMEN_EXPLANATION,
                content=response,
                processing_time=processing_time,
                metadata={"input_length": len(technical_analysis)},
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Laymen explanation agent failed: {e}")
            
            return AgentResponse(
                agent_type=AgentType.LAYMEN_EXPLANATION,
                content="",
                processing_time=processing_time,
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    async def _run_sources_analysis_agent(
        self, 
        question: str, 
        retrieved_docs: List[Document]
    ) -> AgentResponse:
        """Run the Sources Analysis Agent."""
        start_time = time.time()
        
        try:
            # Format sources for analysis
            sources_info = self._format_sources_for_analysis(retrieved_docs)
            
            prompt = self.agent_prompts[AgentType.SOURCES_ANALYSIS]
            chain = prompt | self.llm | self.string_parser
            
            response = chain.invoke({
                "sources": sources_info,
                "question": question
            })
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                agent_type=AgentType.SOURCES_ANALYSIS,
                content=response,
                processing_time=processing_time,
                metadata={
                    "source_count": len(retrieved_docs),
                    "unique_documents": len(set(doc.metadata.get('source', '') for doc in retrieved_docs))
                },
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Sources analysis agent failed: {e}")
            
            return AgentResponse(
                agent_type=AgentType.SOURCES_ANALYSIS,
                content="",
                processing_time=processing_time,
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    async def _run_response_synthesis_agent(
        self,
        query_analysis: str,
        technical_analysis: str,
        laymen_explanation: str,
        sources_analysis: str
    ) -> AgentResponse:
        """Run the Response Synthesis Agent."""
        start_time = time.time()
        
        try:
            prompt = self.agent_prompts[AgentType.RESPONSE_SYNTHESIS]
            chain = prompt | self.llm | self.string_parser
            
            response = chain.invoke({
                "query_analysis": query_analysis,
                "technical_analysis": technical_analysis,
                "laymen_explanation": laymen_explanation,
                "sources_analysis": sources_analysis
            })
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                agent_type=AgentType.RESPONSE_SYNTHESIS,
                content=response,
                processing_time=processing_time,
                metadata={
                    "inputs_provided": sum([
                        bool(query_analysis),
                        bool(technical_analysis),
                        bool(laymen_explanation),
                        bool(sources_analysis)
                    ])
                },
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Response synthesis agent failed: {e}")
            
            return AgentResponse(
                agent_type=AgentType.RESPONSE_SYNTHESIS,
                content="",
                processing_time=processing_time,
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _format_context_for_agent(self, documents: List[Document], question: str) -> str:
        """Format retrieved documents for agent processing."""
        context_parts = []
        
        for i, doc in enumerate(documents):
            source_info = f"[Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}]"
            content = doc.page_content
            context_parts.append(f"{source_info}\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_sources_for_analysis(self, documents: List[Document]) -> str:
        """Format source information for the Sources Analysis Agent."""
        sources_info = []
        
        for i, doc in enumerate(documents, 1):
            source_entry = f"""
Source {i}:
- Document: {doc.metadata.get('source', 'Unknown')}
- Page: {doc.metadata.get('page', 'Unknown')}
- Content Type: {doc.metadata.get('content_type', 'text')}
- Document Date: {doc.metadata.get('document_date', 'Unknown')}
- Proceeding: {doc.metadata.get('proceeding_number', 'Unknown')}
- Excerpt: {doc.page_content[:300]}...
"""
            sources_info.append(source_entry)
        
        return "\n".join(sources_info)
    
    def _create_fallback_response(self, technical_content: str, laymen_content: str) -> str:
        """Create a fallback response when synthesis fails."""
        sections = []
        
        if technical_content:
            sections.append(f"""
<div style="background-color: #f8f9fa; border-radius: 8px; padding: 25px; margin-bottom: 25px; border: 1px solid #dee2e6;">
    <h3 style="color: #0d6efd;">📋 Technical Analysis</h3>
    {technical_content}
</div>
""")
        
        if laymen_content:
            sections.append(f"""
<div style="background-color: #f8f9fa; border-radius: 8px; padding: 25px; margin-bottom: 25px; border: 1px solid #dee2e6;">
    <h3 style="color: #198754;">💡 Simplified Explanation</h3>
    {laymen_content}
</div>
""")
        
        if not sections:
            return "<div style='background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 10px; margin: 10px 0;'><strong>Note:</strong> Multi-agent analysis was unable to process this query. Please try rephrasing your question.</div>"
        
        return "".join(sections)
    
    def _extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from the response text."""
        import re
        
        citations = []
        citation_pattern = re.compile(r'\[CITE:([^,]+),page_(\d+)\]')
        
        for match in citation_pattern.finditer(text):
            citations.append({
                "document": match.group(1),
                "page": match.group(2),
                "citation_text": match.group(0)
            })
        
        return citations
    
    def _calculate_confidence_score(
        self, 
        agent_responses: Dict[AgentType, AgentResponse], 
        processing_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score based on agent performance."""
        score_factors = []
        
        # Success rate factor (0-0.4)
        success_rate = processing_metrics.get("success_rate", 0)
        score_factors.append(success_rate * 0.4)
        
        # Technical analysis quality (0-0.3)
        tech_response = agent_responses.get(AgentType.TECHNICAL_ANALYSIS)
        if tech_response and tech_response.success:
            # Simple heuristic based on content length and citation presence
            content_score = min(len(tech_response.content) / 1000, 1.0) * 0.15
            citation_score = 0.15 if "[CITE:" in tech_response.content else 0
            score_factors.append(content_score + citation_score)
        
        # Sources analysis quality (0-0.2)
        sources_response = agent_responses.get(AgentType.SOURCES_ANALYSIS)
        if sources_response and sources_response.success:
            score_factors.append(0.2)
        
        # Response completeness (0-0.1)
        completeness = len([r for r in agent_responses.values() if r.success]) / len(agent_responses) * 0.1
        score_factors.append(completeness)
        
        return sum(score_factors)


# Synchronous wrapper for backward compatibility
class MultiAgentSystem:
    """Synchronous wrapper for the MultiAgentOrchestrator."""
    
    def __init__(self, llm, embedding_model=None):
        self.orchestrator = MultiAgentOrchestrator(llm, embedding_model)
    
    def process_query(
        self, 
        question: str, 
        retrieved_docs: List[Document],
        enable_agents: Optional[Dict[str, bool]] = None
    ) -> MultiAgentResult:
        """Process a query synchronously."""
        import asyncio
        
        # Run the async method in a synchronous context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.orchestrator.process_query(question, retrieved_docs, enable_agents)
        )