"""
Task Service for Claude Manager Service

This service provides task orchestration and execution management,
coordinating workflows between different services and managing task lifecycles.

Features:
- Task workflow orchestration
- Service coordination and dependency injection
- Task execution tracking and monitoring
- Batch task processing with concurrency
- Performance monitoring and error recovery
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

from logger import get_logger
from performance_monitor import monitor_performance, get_monitor
from error_handler import NetworkError, with_enhanced_error_handling
from task_tracker import get_task_tracker
from services.configuration_service import get_configuration_service
from services.repository_service import RepositoryService
from services.issue_service import get_issue_service


logger = get_logger(__name__)


class TaskType(Enum):
    """Task type enumeration"""
    REPOSITORY_SCAN = "repository_scan"
    ISSUE_CREATION = "issue_creation"
    CODE_ANALYSIS = "code_analysis"
    BULK_OPERATION = "bulk_operation"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    MAINTENANCE = "maintenance"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class TaskDefinition:
    """Definition of a task to be executed"""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # List of task_ids this task depends on
    priority: int = 5  # 1-10, higher is more important
    retry_attempts: int = 3
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    workflow_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    total_duration: float
    success_rate: float
    task_results: List[TaskResult] = field(default_factory=list)


class TaskService:
    """
    Service for task orchestration and workflow management
    
    This service coordinates between different services to execute complex
    workflows and provides task execution management with monitoring.
    """
    
    def __init__(self, max_concurrent_tasks: int = 8, default_timeout: int = 300):
        """
        Initialize task service
        
        Args:
            max_concurrent_tasks: Maximum concurrent task execution
            default_timeout: Default task timeout in seconds
        """
        self.logger = get_logger(__name__)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        
        # Task tracking
        self._active_tasks: Dict[str, TaskResult] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._task_lock = asyncio.Lock()
        
        # Service instances (lazy loaded)
        self._config_service = None
        self._repository_service = None
        self._issue_service = None
        
        # Performance monitoring
        self.performance_monitor = get_monitor()
        
        self.logger.info(f"Task service initialized: max_concurrent={max_concurrent_tasks}")
    
    async def _get_config_service(self):
        """Get configuration service instance"""
        if self._config_service is None:
            self._config_service = await get_configuration_service()
        return self._config_service
    
    async def _get_repository_service(self):
        """Get repository service instance"""
        if self._repository_service is None:
            self._repository_service = RepositoryService()
        return self._repository_service
    
    async def _get_issue_service(self):
        """Get issue service instance"""
        if self._issue_service is None:
            self._issue_service = await get_issue_service()
        return self._issue_service
    
    @with_enhanced_error_handling("execute_task", use_circuit_breaker=True)
    async def execute_task(self, task_def: TaskDefinition) -> TaskResult:
        """
        Execute a single task
        
        Args:
            task_def: Task definition to execute
            
        Returns:
            TaskResult with execution details
        """
        task_result = TaskResult(
            task_id=task_def.task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now()
        )
        
        self.logger.info(f"Starting task execution: {task_def.name} ({task_def.task_type.value})")
        
        try:
            # Register active task
            async with self._task_lock:
                self._active_tasks[task_def.task_id] = task_result
            
            # Execute task based on type
            if task_def.task_type == TaskType.REPOSITORY_SCAN:
                result_data = await self._execute_repository_scan(task_def)
            elif task_def.task_type == TaskType.ISSUE_CREATION:
                result_data = await self._execute_issue_creation(task_def)
            elif task_def.task_type == TaskType.CODE_ANALYSIS:
                result_data = await self._execute_code_analysis(task_def)
            elif task_def.task_type == TaskType.BULK_OPERATION:
                result_data = await self._execute_bulk_operation(task_def)
            elif task_def.task_type == TaskType.WORKFLOW_ORCHESTRATION:
                result_data = await self._execute_workflow_orchestration(task_def)
            elif task_def.task_type == TaskType.MAINTENANCE:
                result_data = await self._execute_maintenance(task_def)
            else:
                raise ValueError(f"Unknown task type: {task_def.task_type}")
            
            # Task completed successfully
            task_result.status = TaskStatus.COMPLETED
            task_result.result_data = result_data
            
            self.logger.info(f"Task completed successfully: {task_def.name}")
            
        except asyncio.TimeoutError:
            task_result.status = TaskStatus.FAILED
            task_result.error_message = f"Task timed out after {task_def.timeout_seconds} seconds"
            self.logger.error(f"Task timed out: {task_def.name}")
            
        except Exception as e:
            task_result.status = TaskStatus.FAILED
            task_result.error_message = str(e)
            self.logger.error(f"Task failed: {task_def.name} - {e}")
            
            # Re-raise for enhanced error handling
            raise NetworkError(f"Task execution failed: {str(e)}", "execute_task", e)
        
        finally:
            # Update task completion
            task_result.completed_at = datetime.now()
            if task_result.started_at:
                task_result.duration = (task_result.completed_at - task_result.started_at).total_seconds()
            
            # Move from active to completed
            async with self._task_lock:
                if task_def.task_id in self._active_tasks:
                    del self._active_tasks[task_def.task_id]
                self._completed_tasks[task_def.task_id] = task_result
        
        return task_result
    
    async def execute_workflow(self, tasks: List[TaskDefinition], 
                             workflow_id: str = None) -> WorkflowResult:
        """
        Execute a workflow of interdependent tasks
        
        Args:
            tasks: List of task definitions
            workflow_id: Optional workflow identifier
            
        Returns:
            WorkflowResult with execution statistics
        """
        if workflow_id is None:
            workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        start_time = asyncio.get_event_loop().time()
        self.logger.info(f"Starting workflow execution: {workflow_id} with {len(tasks)} tasks")
        
        workflow_result = WorkflowResult(
            workflow_id=workflow_id,
            total_tasks=len(tasks),
            completed_tasks=0,
            failed_tasks=0,
            cancelled_tasks=0,
            total_duration=0.0,
            success_rate=0.0
        )
        
        if not tasks:
            workflow_result.total_duration = asyncio.get_event_loop().time() - start_time
            workflow_result.success_rate = 100.0
            return workflow_result
        
        # Sort tasks by priority and dependencies
        sorted_tasks = await self._sort_tasks_by_dependencies(tasks)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        async def execute_single_task(task_def: TaskDefinition) -> TaskResult:
            async with semaphore:
                return await asyncio.wait_for(
                    self.execute_task(task_def),
                    timeout=task_def.timeout_seconds
                )
        
        # Execute tasks with dependency resolution
        with monitor_performance("workflow_execution"):
            task_results = await self._execute_with_dependencies(sorted_tasks, execute_single_task)
        
        # Process results
        for task_result in task_results:
            workflow_result.task_results.append(task_result)
            
            if task_result.status == TaskStatus.COMPLETED:
                workflow_result.completed_tasks += 1
            elif task_result.status == TaskStatus.FAILED:
                workflow_result.failed_tasks += 1
            elif task_result.status == TaskStatus.CANCELLED:
                workflow_result.cancelled_tasks += 1
        
        # Calculate final statistics
        workflow_result.total_duration = asyncio.get_event_loop().time() - start_time
        workflow_result.success_rate = (workflow_result.completed_tasks / workflow_result.total_tasks) * 100 if workflow_result.total_tasks > 0 else 0
        
        self.logger.info(f"Workflow completed: {workflow_id} - {workflow_result.completed_tasks}/{workflow_result.total_tasks} successful ({workflow_result.success_rate:.1f}%)")
        
        return workflow_result
    
    async def create_repository_scan_workflow(self, repo_names: List[str], 
                                            manager_repo: str) -> List[TaskDefinition]:
        """
        Create a workflow for scanning multiple repositories
        
        Args:
            repo_names: List of repository names to scan
            manager_repo: Manager repository for issue creation
            
        Returns:
            List of task definitions for the workflow
        """
        tasks = []
        
        # Create repository scan tasks
        for i, repo_name in enumerate(repo_names):
            task_def = TaskDefinition(
                task_id=f"scan_{repo_name.replace('/', '_')}_{i}",
                task_type=TaskType.REPOSITORY_SCAN,
                name=f"Scan Repository: {repo_name}",
                description=f"Scan repository {repo_name} for TODOs and issues",
                parameters={
                    'repository_name': repo_name,
                    'manager_repository': manager_repo
                },
                priority=7,
                retry_attempts=2,
                timeout_seconds=600
            )
            tasks.append(task_def)
        
        # Create consolidation task that depends on all scans
        consolidation_task = TaskDefinition(
            task_id="consolidate_scan_results",
            task_type=TaskType.BULK_OPERATION,
            name="Consolidate Scan Results",
            description="Consolidate and analyze all repository scan results",
            parameters={
                'operation_type': 'consolidate_scans',
                'repositories': repo_names
            },
            dependencies=[task.task_id for task in tasks],
            priority=9,
            timeout_seconds=300
        )
        tasks.append(consolidation_task)
        
        return tasks
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """
        Get status of a specific task
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskResult if found, None otherwise
        """
        async with self._task_lock:
            # Check active tasks first
            if task_id in self._active_tasks:
                return self._active_tasks[task_id]
            # Check completed tasks
            return self._completed_tasks.get(task_id)
    
    async def get_workflow_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive task and workflow statistics
        
        Returns:
            Dictionary with task statistics
        """
        async with self._task_lock:
            active_count = len(self._active_tasks)
            completed_count = len(self._completed_tasks)
            
            # Analyze completed tasks
            status_counts = {}
            type_counts = {}
            average_duration = 0.0
            total_duration = 0.0
            
            for task_result in self._completed_tasks.values():
                # Count by status
                status = task_result.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Calculate duration statistics
                if task_result.duration:
                    total_duration += task_result.duration
            
            if completed_count > 0:
                average_duration = total_duration / completed_count
            
            success_rate = 0.0
            if completed_count > 0:
                successful = status_counts.get('completed', 0)
                success_rate = (successful / completed_count) * 100
            
            return {
                'active_tasks': active_count,
                'completed_tasks': completed_count,
                'total_tasks': active_count + completed_count,
                'status_counts': status_counts,
                'success_rate': success_rate,
                'average_duration_seconds': average_duration,
                'total_duration_seconds': total_duration
            }
    
    async def _execute_repository_scan(self, task_def: TaskDefinition) -> Dict[str, Any]:
        """Execute repository scan task"""
        repo_service = await self._get_repository_service()
        repo_name = task_def.parameters['repository_name']
        manager_repo = task_def.parameters['manager_repository']
        
        # Get repository info and scan
        repo_infos = await repo_service.get_repositories([repo_name])
        if not repo_infos:
            raise ValueError(f"Repository not found: {repo_name}")
        
        scan_result = await repo_service.scan_repository_todos(repo_infos[0], manager_repo)
        
        return {
            'repository': repo_name,
            'todos_found': scan_result.todos_found,
            'files_scanned': scan_result.files_scanned,
            'scan_duration': scan_result.scan_duration,
            'success': scan_result.success
        }
    
    async def _execute_issue_creation(self, task_def: TaskDefinition) -> Dict[str, Any]:
        """Execute issue creation task"""
        issue_service = await self._get_issue_service()
        
        # Extract parameters
        repo_name = task_def.parameters['repository']
        title = task_def.parameters['title']
        body = task_def.parameters['body']
        labels = task_def.parameters.get('labels', [])
        
        # Create issue
        issue_metadata = await issue_service.create_issue(repo_name, title, body, labels)
        
        return {
            'issue_id': issue_metadata.issue_id,
            'github_issue_number': issue_metadata.github_issue_number,
            'status': issue_metadata.status.value,
            'repository': issue_metadata.repository
        }
    
    async def _execute_code_analysis(self, task_def: TaskDefinition) -> Dict[str, Any]:
        """Execute code analysis task"""
        # Placeholder for code analysis implementation
        analysis_type = task_def.parameters.get('analysis_type', 'basic')
        target = task_def.parameters.get('target', 'unknown')
        
        self.logger.info(f"Performing {analysis_type} code analysis on {target}")
        
        # Simulate analysis work
        await asyncio.sleep(1)
        
        return {
            'analysis_type': analysis_type,
            'target': target,
            'results': {'complexity_score': 7.5, 'issues_found': 3}
        }
    
    async def _execute_bulk_operation(self, task_def: TaskDefinition) -> Dict[str, Any]:
        """Execute bulk operation task"""
        operation_type = task_def.parameters.get('operation_type', 'unknown')
        
        if operation_type == 'consolidate_scans':
            # Consolidate repository scan results
            repositories = task_def.parameters.get('repositories', [])
            
            self.logger.info(f"Consolidating scan results for {len(repositories)} repositories")
            
            # Get scan results from completed tasks
            total_todos = 0
            total_files = 0
            async with self._task_lock:
                for task_result in self._completed_tasks.values():
                    if 'todos_found' in task_result.result_data:
                        total_todos += task_result.result_data['todos_found']
                        total_files += task_result.result_data.get('files_scanned', 0)
            
            return {
                'operation_type': operation_type,
                'repositories_processed': len(repositories),
                'total_todos_found': total_todos,
                'total_files_scanned': total_files
            }
        
        return {'operation_type': operation_type, 'status': 'completed'}
    
    async def _execute_workflow_orchestration(self, task_def: TaskDefinition) -> Dict[str, Any]:
        """Execute workflow orchestration task"""
        workflow_type = task_def.parameters.get('workflow_type', 'custom')
        
        self.logger.info(f"Orchestrating {workflow_type} workflow")
        
        return {
            'workflow_type': workflow_type,
            'orchestration_completed': True
        }
    
    async def _execute_maintenance(self, task_def: TaskDefinition) -> Dict[str, Any]:
        """Execute maintenance task"""
        maintenance_type = task_def.parameters.get('maintenance_type', 'general')
        
        if maintenance_type == 'cleanup_cache':
            # Clear service caches
            repo_service = await self._get_repository_service()
            issue_service = await self._get_issue_service()
            
            await repo_service.clear_cache()
            await issue_service.clear_cache()
            
            return {
                'maintenance_type': maintenance_type,
                'caches_cleared': ['repository', 'issue']
            }
        
        return {
            'maintenance_type': maintenance_type,
            'status': 'completed'
        }
    
    async def _sort_tasks_by_dependencies(self, tasks: List[TaskDefinition]) -> List[TaskDefinition]:
        """Sort tasks by dependencies and priority"""
        # Simple topological sort implementation
        task_map = {task.task_id: task for task in tasks}
        sorted_tasks = []
        processed = set()
        
        def process_task(task_id: str):
            if task_id in processed:
                return
            
            task = task_map.get(task_id)
            if not task:
                return
            
            # Process dependencies first
            for dep_id in task.dependencies:
                process_task(dep_id)
            
            sorted_tasks.append(task)
            processed.add(task_id)
        
        # Process all tasks
        for task in tasks:
            process_task(task.task_id)
        
        # Sort by priority within dependency levels
        return sorted(sorted_tasks, key=lambda t: (-t.priority, t.created_at))
    
    async def _execute_with_dependencies(self, tasks: List[TaskDefinition], 
                                       executor: Callable) -> List[TaskResult]:
        """Execute tasks respecting dependencies"""
        results = []
        completed_tasks = set()
        
        # Execute tasks in dependency order
        for task in tasks:
            # Wait for dependencies to complete
            for dep_id in task.dependencies:
                while dep_id not in completed_tasks:
                    await asyncio.sleep(0.1)
            
            # Execute task
            try:
                result = await executor(task)
                results.append(result)
                completed_tasks.add(task.task_id)
            except Exception as e:
                # Create failed result
                failed_result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error_message=str(e),
                    started_at=datetime.now(),
                    completed_at=datetime.now()
                )
                results.append(failed_result)
                completed_tasks.add(task.task_id)
        
        return results


# Global task service instance
_task_service: Optional[TaskService] = None


async def get_task_service(max_concurrent_tasks: int = 8) -> TaskService:
    """
    Get global task service instance
    
    Args:
        max_concurrent_tasks: Maximum concurrent task execution
        
    Returns:
        Initialized task service
    """
    global _task_service
    
    if _task_service is None:
        _task_service = TaskService(max_concurrent_tasks=max_concurrent_tasks)
    
    return _task_service


# Example usage and testing
async def example_task_service():
    """Example of using task service"""
    try:
        # Get task service
        task_service = await get_task_service()
        
        # Create a simple workflow
        repo_names = ["example/repo1", "example/repo2"]
        manager_repo = "manager/issues"
        
        workflow_tasks = await task_service.create_repository_scan_workflow(repo_names, manager_repo)
        
        # Execute workflow
        workflow_result = await task_service.execute_workflow(workflow_tasks)
        
        # Get statistics
        stats = await task_service.get_workflow_statistics()
        
        logger.info(f"Task service example completed")
        logger.info(f"Workflow: {workflow_result.completed_tasks}/{workflow_result.total_tasks} successful")
        logger.info(f"Statistics: {stats['total_tasks']} total tasks processed")
        
    except Exception as e:
        logger.error(f"Task service example failed: {e}")
        raise


if __name__ == "__main__":
    # Test task service
    asyncio.run(example_task_service())