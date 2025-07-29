# 📁 agents/base_agent.py
# Base Agent Framework for CPUC Regulatory Search Engine

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import json

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Defines the core capabilities that agents can possess."""
    DOCUMENT_SEARCH = "document_search"
    WEB_SEARCH = "web_search"
    CITATION_ANALYSIS = "citation_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    LEGAL_INTERPRETATION = "legal_interpretation"
    HISTORICAL_CONTEXT = "historical_context"
    FINANCIAL_ANALYSIS = "financial_analysis"
    QUESTION_ROUTING = "question_routing"
    RESULT_SYNTHESIS = "result_synthesis"
    CONFLICT_RESOLUTION = "conflict_resolution"


class TaskPriority(Enum):
    """Priority levels for agent tasks."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TaskStatus(Enum):
    """Status of agent tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """Represents a task that can be executed by an agent."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    task_type: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def is_expired(self) -> bool:
        """Check if task has exceeded timeout."""
        if self.started_at:
            elapsed = (datetime.now() - self.started_at).total_seconds()
            return elapsed > self.timeout_seconds
        return False


@dataclass
class AgentResult:
    """Represents the result of an agent's execution."""
    agent_id: str
    task_id: str
    success: bool
    result_data: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "success": self.success,
            "result_data": self.result_data,
            "confidence_score": self.confidence_score,
            "sources": self.sources,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "error_message": self.error_message
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the CPUC regulatory search system.
    
    Provides common functionality for agent lifecycle management, task execution,
    result processing, and integration with the broader system infrastructure.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        llm: BaseLanguageModel,
        tools: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and functionality
            capabilities: List of capabilities this agent possesses
            llm: Language model instance for text generation
            tools: Dictionary of tools available to this agent
            config: Agent-specific configuration parameters
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.llm = llm
        self.tools = tools or {}
        self.config = config or {}
        
        # Agent state management
        self.is_active = True
        self.current_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.error_count = 0
        self.max_errors = self.config.get("max_errors", 10)
        
        # Performance metrics
        self.total_tasks_completed = 0
        self.total_execution_time = 0.0
        self.average_confidence_score = 0.0
        
        # Memory and context
        self.memory: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized agent {self.agent_id} ({self.name}) with capabilities: {[cap.value for cap in capabilities]}")
    
    @abstractmethod
    def process_task(self, task: AgentTask) -> AgentResult:
        """
        Process a specific task assigned to this agent.
        
        This method must be implemented by all concrete agent classes.
        It should contain the core logic for handling the agent's specific
        functionality and responsibilities.
        
        Args:
            task: The task to be processed
            
        Returns:
            AgentResult: The result of processing the task
        """
        pass
    
    def can_handle_task(self, task: AgentTask) -> bool:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task: The task to evaluate
            
        Returns:
            bool: True if the agent can handle the task, False otherwise
        """
        task_type = task.task_type
        required_capabilities = task.input_data.get("required_capabilities", [])
        
        # Check if agent has required capabilities
        agent_capability_values = [cap.value for cap in self.capabilities]
        
        if required_capabilities:
            return all(cap in agent_capability_values for cap in required_capabilities)
        
        # Default capability mapping for common task types
        capability_mapping = {
            "document_search": [AgentCapability.DOCUMENT_SEARCH.value],
            "web_search": [AgentCapability.WEB_SEARCH.value],
            "technical_analysis": [AgentCapability.TECHNICAL_ANALYSIS.value],
            "legal_interpretation": [AgentCapability.LEGAL_INTERPRETATION.value],
            "question_routing": [AgentCapability.QUESTION_ROUTING.value],
            "result_synthesis": [AgentCapability.RESULT_SYNTHESIS.value]
        }
        
        if task_type in capability_mapping:
            required_caps = capability_mapping[task_type]
            return any(cap in agent_capability_values for cap in required_caps)
        
        return False
    
    def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a task with comprehensive error handling and monitoring.
        
        Args:
            task: The task to execute
            
        Returns:
            AgentResult: The result of task execution
        """
        if not self.is_active:
            return self._create_error_result(task, "Agent is not active")
        
        if not self.can_handle_task(task):
            return self._create_error_result(task, f"Agent cannot handle task type: {task.task_type}")
        
        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        self.current_tasks[task.task_id] = task
        
        try:
            logger.info(f"Agent {self.agent_id} executing task {task.task_id} ({task.task_type})")
            
            # Check for timeout
            if task.is_expired():
                raise TimeoutError(f"Task {task.task_id} exceeded timeout of {task.timeout_seconds} seconds")
            
            # Execute the actual task
            result = self.process_task(task)
            
            # Update task completion
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.output_data = result.result_data
            
            # Update agent metrics
            self.total_tasks_completed += 1
            if result.execution_time > 0:
                self.total_execution_time += result.execution_time
            
            # Update confidence score moving average
            if result.confidence_score > 0:
                current_avg = self.average_confidence_score
                new_avg = ((current_avg * (self.total_tasks_completed - 1)) + result.confidence_score) / self.total_tasks_completed
                self.average_confidence_score = new_avg
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            del self.current_tasks[task.task_id]
            
            logger.info(f"Agent {self.agent_id} completed task {task.task_id} with confidence {result.confidence_score:.3f}")
            return result
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            self.error_count += 1
            
            # Check if agent should be deactivated due to too many errors
            if self.error_count >= self.max_errors:
                logger.error(f"Agent {self.agent_id} deactivated due to {self.error_count} errors")
                self.is_active = False
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            if task.task_id in self.current_tasks:
                del self.current_tasks[task.task_id]
            
            logger.error(f"Agent {self.agent_id} failed task {task.task_id}: {str(e)}")
            return self._create_error_result(task, str(e))
    
    def _create_error_result(self, task: AgentTask, error_message: str) -> AgentResult:
        """Create an error result for a failed task."""
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            success=False,
            error_message=error_message,
            execution_time=0.0
        )
    
    def get_prompt_template(self, prompt_type: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template for the specified type.
        
        Args:
            prompt_type: The type of prompt template to retrieve
            
        Returns:
            PromptTemplate: The prompt template, or None if not found
        """
        prompts = self.config.get("prompts", {})
        if prompt_type in prompts:
            return PromptTemplate.from_template(prompts[prompt_type])
        return None
    
    def create_llm_chain(self, prompt_template: PromptTemplate) -> Any:
        """
        Create an LLM chain with the given prompt template.
        
        Args:
            prompt_template: The prompt template to use
            
        Returns:
            LLM chain ready for execution
        """
        return prompt_template | self.llm | StrOutputParser()
    
    def update_memory(self, key: str, value: Any, persistent: bool = False) -> None:
        """
        Update agent memory with new information.
        
        Args:
            key: Memory key
            value: Value to store
            persistent: Whether to persist across agent restarts
        """
        self.memory[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "persistent": persistent
        }
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """
        Retrieve value from agent memory.
        
        Args:
            key: Memory key
            default: Default value if key not found
            
        Returns:
            Stored value or default
        """
        memory_item = self.memory.get(key)
        if memory_item:
            return memory_item["value"]
        return default
    
    def add_conversation_entry(self, entry: Dict[str, Any]) -> None:
        """
        Add an entry to the conversation history.
        
        Args:
            entry: Conversation entry containing role, content, metadata
        """
        entry["timestamp"] = datetime.now().isoformat()
        self.conversation_history.append(entry)
        
        # Limit conversation history size
        max_history = self.config.get("max_conversation_history", 100)
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
    
    def get_capabilities_summary(self) -> Dict[str, Any]:
        """Get a summary of agent capabilities and current status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [cap.value for cap in self.capabilities],
            "is_active": self.is_active,
            "current_tasks_count": len(self.current_tasks),
            "total_tasks_completed": self.total_tasks_completed,
            "error_count": self.error_count,
            "average_confidence_score": round(self.average_confidence_score, 3),
            "average_execution_time": round(self.total_execution_time / max(1, self.total_tasks_completed), 3)
        }
    
    def reset_agent(self) -> None:
        """Reset agent state and clear temporary data."""
        self.current_tasks.clear()
        self.error_count = 0
        self.is_active = True
        
        # Clear non-persistent memory
        persistent_memory = {
            k: v for k, v in self.memory.items() 
            if v.get("persistent", False)
        }
        self.memory = persistent_memory
        
        logger.info(f"Agent {self.agent_id} has been reset")
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"Agent({self.agent_id}: {self.name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"Agent(id={self.agent_id}, name={self.name}, "
                f"capabilities={[cap.value for cap in self.capabilities]}, "
                f"active={self.is_active})")


class AgentPool:
    """
    Manages a pool of agents and provides agent selection and load balancing.
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """
        Initialize the agent pool.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks across all agents
        """
        self.agents: Dict[str, BaseAgent] = {}
        self.max_concurrent_tasks = max_concurrent_tasks
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the pool."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id} ({agent.name}) with pool")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the pool."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent {agent_id} from pool")
    
    def get_available_agents(self, task: AgentTask) -> List[BaseAgent]:
        """
        Get all agents capable of handling the given task.
        
        Args:
            task: The task to find agents for
            
        Returns:
            List of capable agents, sorted by availability and performance
        """
        capable_agents = []
        
        for agent in self.agents.values():
            if agent.is_active and agent.can_handle_task(task):
                capable_agents.append(agent)
        
        # Sort by current load (fewer active tasks first) and average confidence
        capable_agents.sort(
            key=lambda a: (len(a.current_tasks), -a.average_confidence_score)
        )
        
        return capable_agents
    
    def select_best_agent(self, task: AgentTask) -> Optional[BaseAgent]:
        """
        Select the best agent for a given task.
        
        Args:
            task: The task to assign
            
        Returns:
            Best available agent, or None if no agent can handle the task
        """
        available_agents = self.get_available_agents(task)
        
        if not available_agents:
            return None
        
        # Select agent with best combination of availability and performance
        return available_agents[0]
    
    def execute_task_async(self, task: AgentTask) -> Optional[Any]:
        """
        Execute a task asynchronously using the best available agent.
        
        Args:
            task: The task to execute
            
        Returns:
            Future object for the task execution
        """
        agent = self.select_best_agent(task)
        if not agent:
            logger.warning(f"No agent available for task {task.task_id} ({task.task_type})")
            return None
        
        # Submit task to thread pool
        future = self.executor.submit(agent.execute_task, task)
        return future
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the agent pool."""
        total_active_agents = sum(1 for agent in self.agents.values() if agent.is_active)
        total_current_tasks = sum(len(agent.current_tasks) for agent in self.agents.values())
        total_completed_tasks = sum(agent.total_tasks_completed for agent in self.agents.values())
        
        agent_summaries = [agent.get_capabilities_summary() for agent in self.agents.values()]
        
        return {
            "total_agents": len(self.agents),
            "active_agents": total_active_agents,
            "total_current_tasks": total_current_tasks,
            "total_completed_tasks": total_completed_tasks,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "agents": agent_summaries
        }
    
    def shutdown(self) -> None:
        """Shutdown the agent pool and cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("Agent pool shutdown complete")