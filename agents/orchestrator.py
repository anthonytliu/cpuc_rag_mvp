# 📁 agents/orchestrator.py
# Agent Orchestration System for CPUC Regulatory Search Engine

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading
import time

from .base_agent import BaseAgent, AgentTask, AgentResult, TaskPriority, TaskStatus, AgentPool
from .agent_tools import ToolRegistry

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExecutionStrategy(Enum):
    """Execution strategies for multi-agent workflows."""
    SEQUENTIAL = "sequential"  # Execute tasks one after another
    PARALLEL = "parallel"     # Execute all tasks simultaneously
    PIPELINE = "pipeline"     # Execute in stages with dependencies
    ADAPTIVE = "adaptive"     # Adapt strategy based on resources and task types


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    step_id: str
    agent_type: str
    task_type: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # IDs of steps this depends on
    timeout_seconds: int = 300
    required_capabilities: List[str] = field(default_factory=list)
    optional: bool = False  # If True, workflow continues even if this step fails
    retry_count: int = 0
    max_retries: int = 2
    
    def can_execute(self, completed_steps: Set[str]) -> bool:
        """Check if this step can execute based on completed dependencies."""
        return all(dep in completed_steps for dep in self.dependencies)


@dataclass
class WorkflowDefinition:
    """Defines a complete workflow for processing queries."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    max_execution_time: int = 1800  # 30 minutes
    created_at: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> List[str]:
        """Validate workflow definition and return any errors."""
        errors = []
        
        if not self.steps:
            errors.append("Workflow must have at least one step")
        
        step_ids = {step.step_id for step in self.steps}
        
        # Check for duplicate step IDs
        if len(step_ids) != len(self.steps):
            errors.append("Duplicate step IDs found")
        
        # Check dependencies
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step {step.step_id} has invalid dependency: {dep}")
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Circular dependencies detected")
        
        return errors
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS."""
        def has_cycle(step_id: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            
            step = next((s for s in self.steps if s.step_id == step_id), None)
            if step:
                for dep in step.dependencies:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(step_id)
            return False
        
        visited = set()
        rec_stack = set()
        
        for step in self.steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id, visited, rec_stack):
                    return True
        
        return False


@dataclass
class WorkflowExecution:
    """Tracks the execution of a workflow instance."""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, AgentResult] = field(default_factory=dict)
    step_tasks: Dict[str, AgentTask] = field(default_factory=dict)
    step_status: Dict[str, TaskStatus] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: float = 0.0
    
    def get_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def get_completed_steps(self) -> Set[str]:
        """Get set of completed step IDs."""
        return {
            step_id for step_id, status in self.step_status.items()
            if status == TaskStatus.COMPLETED
        }
    
    def update_progress(self, total_steps: int) -> None:
        """Update progress percentage based on completed steps."""
        completed = len(self.get_completed_steps())
        self.progress_percentage = (completed / total_steps) * 100 if total_steps > 0 else 0


class AgentOrchestrator:
    """
    Orchestrates multiple agents to execute complex workflows for regulatory search queries.
    
    The orchestrator manages workflow definitions, executes multi-agent workflows,
    handles dependencies between tasks, and provides comprehensive monitoring and
    result aggregation capabilities.
    """
    
    def __init__(
        self,
        agent_pool: AgentPool,
        tool_registry: ToolRegistry,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent orchestrator.
        
        Args:
            agent_pool: Pool of available agents
            tool_registry: Registry of available tools
            config: Orchestrator configuration
        """
        self.agent_pool = agent_pool
        self.tool_registry = tool_registry
        self.config = config or {}
        
        # Workflow management
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        
        # Execution control
        self.max_concurrent_workflows = self.config.get("max_concurrent_workflows", 5)
        self.default_timeout = self.config.get("default_timeout", 1800)
        self.execution_lock = threading.Lock()
        
        # Performance monitoring
        self.total_workflows_executed = 0
        self.total_execution_time = 0.0
        self.average_workflow_duration = 0.0
        
        # Create default workflows
        self._create_default_workflows()
        
        logger.info("Agent orchestrator initialized")
    
    def register_workflow(self, workflow: WorkflowDefinition) -> bool:
        """
        Register a workflow definition.
        
        Args:
            workflow: Workflow definition to register
            
        Returns:
            True if registered successfully, False otherwise
        """
        errors = workflow.validate()
        if errors:
            logger.error(f"Invalid workflow {workflow.workflow_id}: {errors}")
            return False
        
        self.workflow_definitions[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.workflow_id} ({workflow.name})")
        return True
    
    def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> str:
        """
        Execute a workflow asynchronously.
        
        Args:
            workflow_id: ID of workflow to execute
            input_data: Input data for the workflow
            execution_id: Optional custom execution ID
            
        Returns:
            Execution ID for tracking the workflow
        """
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        if len(self.active_executions) >= self.max_concurrent_workflows:
            raise RuntimeError("Maximum concurrent workflows exceeded")
        
        # Create execution instance
        if not execution_id:
            execution_id = f"{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_executions)}"
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            input_data=input_data
        )
        
        with self.execution_lock:
            self.active_executions[execution_id] = execution
        
        # Start execution in background thread
        thread = threading.Thread(
            target=self._execute_workflow_thread,
            args=(execution,),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started workflow execution: {execution_id}")
        return execution_id
    
    def _execute_workflow_thread(self, execution: WorkflowExecution) -> None:
        """Execute workflow in a separate thread."""
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.now()
            
            workflow = self.workflow_definitions[execution.workflow_id]
            
            if workflow.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                self._execute_sequential(workflow, execution)
            elif workflow.execution_strategy == ExecutionStrategy.PARALLEL:
                self._execute_parallel(workflow, execution)
            elif workflow.execution_strategy == ExecutionStrategy.PIPELINE:
                self._execute_pipeline(workflow, execution)
            else:  # ADAPTIVE
                self._execute_adaptive(workflow, execution)
            
            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            
            # Update statistics
            self.total_workflows_executed += 1
            duration = execution.get_duration()
            if duration:
                self.total_execution_time += duration
                self.average_workflow_duration = self.total_execution_time / self.total_workflows_executed
            
            logger.info(f"Workflow {execution.execution_id} completed successfully")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            logger.error(f"Workflow {execution.execution_id} failed: {e}")
        
        finally:
            # Move to history and clean up
            with self.execution_lock:
                if execution.execution_id in self.active_executions:
                    del self.active_executions[execution.execution_id]
                self.execution_history.append(execution)
                
                # Limit history size
                max_history = self.config.get("max_execution_history", 100)
                if len(self.execution_history) > max_history:
                    self.execution_history = self.execution_history[-max_history:]
    
    def _execute_sequential(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> None:
        """Execute workflow steps sequentially."""
        # Sort steps by dependencies (topological sort)
        sorted_steps = self._topological_sort(workflow.steps)
        
        for step in sorted_steps:
            if execution.status != WorkflowStatus.RUNNING:
                break
                
            self._execute_step(step, workflow, execution)
            execution.update_progress(len(workflow.steps))
    
    def _execute_parallel(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> None:
        """Execute all possible steps in parallel."""
        remaining_steps = workflow.steps.copy()
        completed_steps = set()
        
        with ThreadPoolExecutor(max_workers=min(len(remaining_steps), 8)) as executor:
            while remaining_steps and execution.status == WorkflowStatus.RUNNING:
                # Find steps that can execute now
                ready_steps = [
                    step for step in remaining_steps
                    if step.can_execute(completed_steps)
                ]
                
                if not ready_steps:
                    # Check if we're deadlocked
                    if not any(execution.step_status.get(step.step_id) == TaskStatus.IN_PROGRESS 
                             for step in remaining_steps):
                        break
                    time.sleep(0.1)
                    continue
                
                # Submit ready steps for execution
                future_to_step = {}
                for step in ready_steps:
                    future = executor.submit(self._execute_step, step, workflow, execution)
                    future_to_step[future] = step
                    remaining_steps.remove(step)
                
                # Wait for completion
                for future in as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        future.result()  # This will raise if the step failed
                        completed_steps.add(step.step_id)
                    except Exception as e:
                        if not step.optional:
                            raise e
                        logger.warning(f"Optional step {step.step_id} failed: {e}")
                        completed_steps.add(step.step_id)  # Mark as completed to unblock dependencies
                    
                    execution.update_progress(len(workflow.steps))
    
    def _execute_pipeline(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> None:
        """Execute workflow in pipeline stages based on dependencies."""
        remaining_steps = workflow.steps.copy()
        completed_steps = set()
        
        while remaining_steps and execution.status == WorkflowStatus.RUNNING:
            # Find steps that can execute in current stage
            current_stage = [
                step for step in remaining_steps
                if step.can_execute(completed_steps)
            ]
            
            if not current_stage:
                break  # No more steps can execute
            
            # Execute current stage in parallel
            with ThreadPoolExecutor(max_workers=min(len(current_stage), 4)) as executor:
                future_to_step = {
                    executor.submit(self._execute_step, step, workflow, execution): step
                    for step in current_stage
                }
                
                for future in as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        future.result()
                        completed_steps.add(step.step_id)
                    except Exception as e:
                        if not step.optional:
                            raise e
                        logger.warning(f"Optional step {step.step_id} failed: {e}")
                        completed_steps.add(step.step_id)
                    
                    remaining_steps.remove(step)
                    execution.update_progress(len(workflow.steps))
    
    def _execute_adaptive(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> None:
        """Execute workflow using adaptive strategy based on current conditions."""
        # Analyze workflow characteristics
        total_steps = len(workflow.steps)
        parallel_potential = sum(1 for step in workflow.steps if not step.dependencies)
        dependency_ratio = 1 - (parallel_potential / total_steps)
        
        # Choose strategy based on analysis
        if dependency_ratio < 0.3:  # Low dependencies
            self._execute_parallel(workflow, execution)
        elif dependency_ratio > 0.7:  # High dependencies
            self._execute_sequential(workflow, execution)
        else:  # Medium dependencies
            self._execute_pipeline(workflow, execution)
    
    def _execute_step(
        self,
        step: WorkflowStep,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Execute a single workflow step."""
        # Find suitable agent
        task = AgentTask(
            agent_id="",  # Will be set when agent is selected
            task_type=step.task_type,
            input_data={**execution.input_data, **step.input_data},
            timeout_seconds=step.timeout_seconds,
            max_retries=step.max_retries
        )
        
        # Add required capabilities to task
        task.input_data["required_capabilities"] = step.required_capabilities
        
        agent = self.agent_pool.select_best_agent(task)
        if not agent:
            raise RuntimeError(f"No suitable agent found for step {step.step_id}")
        
        task.agent_id = agent.agent_id
        execution.step_tasks[step.step_id] = task
        execution.step_status[step.step_id] = TaskStatus.IN_PROGRESS
        
        # Execute task
        result = agent.execute_task(task)
        
        execution.step_results[step.step_id] = result
        execution.step_status[step.step_id] = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        
        if not result.success and not step.optional:
            raise RuntimeError(f"Step {step.step_id} failed: {result.error_message}")
    
    def _topological_sort(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Sort workflow steps in topological order based on dependencies."""
        # Create adjacency list
        graph = {step.step_id: step.dependencies for step in steps}
        step_map = {step.step_id: step for step in steps}
        
        # Kahn's algorithm
        in_degree = {step_id: 0 for step_id in graph}
        for step_id in graph:
            for dep in graph[step_id]:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            step_id = queue.pop(0)
            result.append(step_map[step_id])
            
            for dependent in graph[step_id]:
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        return result
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of workflow execution.
        
        Args:
            execution_id: Execution ID to check
            
        Returns:
            Status information or None if not found
        """
        # Check active executions
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
        else:
            # Check history
            execution = next(
                (ex for ex in self.execution_history if ex.execution_id == execution_id),
                None
            )
        
        if not execution:
            return None
        
        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "progress_percentage": execution.progress_percentage,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "duration": execution.get_duration(),
            "step_status": {k: v.value for k, v in execution.step_status.items()},
            "error_message": execution.error_message
        }
    
    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active workflow execution.
        
        Args:
            execution_id: Execution ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        with self.execution_lock:
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.now()
                logger.info(f"Cancelled workflow execution: {execution_id}")
                return True
        
        return False
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics."""
        return {
            "total_workflows_defined": len(self.workflow_definitions),
            "active_executions": len(self.active_executions),
            "total_workflows_executed": self.total_workflows_executed,
            "average_workflow_duration": self.average_workflow_duration,
            "max_concurrent_workflows": self.max_concurrent_workflows,
            "agent_pool_status": self.agent_pool.get_pool_status(),
            "tool_registry_stats": self.tool_registry.get_registry_stats()
        }
    
    def _create_default_workflows(self) -> None:
        """Create default workflow definitions for common use cases."""
        
        # Simple query workflow
        simple_query_workflow = WorkflowDefinition(
            workflow_id="simple_query",
            name="Simple Query Processing",
            description="Process simple regulatory queries using document search",
            steps=[
                WorkflowStep(
                    step_id="vector_search",
                    agent_type="search_agent",
                    task_type="document_search",
                    required_capabilities=["document_search"],
                    timeout_seconds=60
                ),
                WorkflowStep(
                    step_id="result_synthesis",
                    agent_type="synthesis_agent",
                    task_type="result_synthesis",
                    dependencies=["vector_search"],
                    required_capabilities=["result_synthesis"],
                    timeout_seconds=120
                )
            ],
            execution_strategy=ExecutionStrategy.SEQUENTIAL
        )
        
        # Complex query workflow
        complex_query_workflow = WorkflowDefinition(
            workflow_id="complex_query",
            name="Complex Query Processing",
            description="Process complex queries requiring multiple analysis types",
            steps=[
                WorkflowStep(
                    step_id="question_analysis",
                    agent_type="interpreter_agent",
                    task_type="question_routing",
                    required_capabilities=["question_routing"],
                    timeout_seconds=30
                ),
                WorkflowStep(
                    step_id="vector_search",
                    agent_type="search_agent",
                    task_type="document_search",
                    dependencies=["question_analysis"],
                    required_capabilities=["document_search"],
                    timeout_seconds=60
                ),
                WorkflowStep(
                    step_id="web_search",
                    agent_type="web_agent",
                    task_type="web_search",
                    dependencies=["question_analysis"],
                    required_capabilities=["web_search"],
                    timeout_seconds=90,
                    optional=True
                ),
                WorkflowStep(
                    step_id="technical_analysis",
                    agent_type="technical_agent",
                    task_type="technical_analysis",
                    dependencies=["vector_search"],
                    required_capabilities=["technical_analysis"],
                    timeout_seconds=180,
                    optional=True
                ),
                WorkflowStep(
                    step_id="legal_analysis",
                    agent_type="legal_agent",
                    task_type="legal_interpretation",
                    dependencies=["vector_search"],
                    required_capabilities=["legal_interpretation"],
                    timeout_seconds=180,
                    optional=True
                ),
                WorkflowStep(
                    step_id="result_coordination",
                    agent_type="coordinator_agent",
                    task_type="result_synthesis",
                    dependencies=["vector_search", "technical_analysis", "legal_analysis"],
                    required_capabilities=["result_synthesis", "conflict_resolution"],
                    timeout_seconds=240
                )
            ],
            execution_strategy=ExecutionStrategy.ADAPTIVE
        )
        
        # Register default workflows
        self.register_workflow(simple_query_workflow)
        self.register_workflow(complex_query_workflow)
        
        logger.info("Default workflows created and registered")