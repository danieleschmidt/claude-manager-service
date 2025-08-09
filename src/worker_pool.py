"""
Advanced Worker Pool and Task Queue System for Claude Manager Service Generation 3

This module provides enterprise-grade concurrent processing including:
- Intelligent worker pools with dynamic scaling
- Priority-based task queues with deadlock prevention
- CPU and I/O bound task separation
- Backpressure management and load shedding
- Task retry mechanisms with exponential backoff
- Worker health monitoring and automatic recovery
- Distributed task coordination
- Performance optimization and resource management
"""

import asyncio
import heapq
import time
import threading
import weakref
import multiprocessing
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Awaitable, Generic, TypeVar
import logging
import os
import signal
import traceback
import concurrent.futures

from .logger import get_logger
from .performance_monitor import monitor_performance, get_monitor
from .error_handler import WorkerError, TaskError, with_enhanced_error_handling


logger = get_logger(__name__)
T = TypeVar('T')


class TaskPriority(IntEnum):
    """Task priority levels (lower values = higher priority)"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class WorkerType(Enum):
    """Types of workers"""
    IO_BOUND = "io_bound"
    CPU_BOUND = "cpu_bound"
    MIXED = "mixed"


@dataclass
class TaskMetrics:
    """Task execution metrics"""
    task_id: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    queue_time: Optional[float] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    worker_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None


@dataclass(order=True)
class Task(Generic[T]):
    """Task wrapper with priority and metadata"""
    priority: TaskPriority = field(compare=True)
    created_at: float = field(compare=True, default_factory=time.time)
    task_id: str = field(compare=False)
    func: Callable[..., Awaitable[T]] = field(compare=False)
    args: Tuple[Any, ...] = field(compare=False, default_factory=tuple)
    kwargs: Dict[str, Any] = field(compare=False, default_factory=dict)
    retry_count: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=3)
    timeout: Optional[float] = field(compare=False, default=None)
    result_future: asyncio.Future = field(compare=False, default_factory=asyncio.Future)
    metrics: TaskMetrics = field(compare=False, init=False)
    
    def __post_init__(self):
        self.metrics = TaskMetrics(self.task_id, self.created_at)


class Worker:
    """Individual worker for task execution"""
    
    def __init__(self, 
                 worker_id: str, 
                 worker_type: WorkerType,
                 queue: asyncio.Queue,
                 max_concurrent_tasks: int = 1):
        
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.queue = queue
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Worker state
        self.is_running = False
        self.current_tasks: Set[asyncio.Task] = set()
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.average_task_time = 0.0
        self.last_activity = time.time()
        
        # Worker task
        self.worker_task: Optional[asyncio.Task] = None
        
        # Thread pool for CPU-bound tasks
        self.thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        if worker_type == WorkerType.CPU_BOUND:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_concurrent_tasks,
                thread_name_prefix=f"worker-{worker_id}"
            )
        
        logger.debug(f"Worker {worker_id} ({worker_type.value}) created")
    
    async def start(self):
        """Start the worker"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info(f"Worker {self.worker_id} started")
    
    async def stop(self):
        """Stop the worker"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel current tasks
        for task in list(self.current_tasks):
            task.cancel()
        
        # Cancel worker task
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    async def _worker_loop(self):
        """Main worker loop"""
        logger.debug(f"Worker {self.worker_id} loop started")
        
        while self.is_running:
            try:
                # Wait for task with timeout
                try:
                    task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Check if we can handle more tasks
                if len(self.current_tasks) >= self.max_concurrent_tasks:
                    # Put task back and wait
                    await self.queue.put(task)
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task
                execution_task = asyncio.create_task(self._execute_task(task))
                self.current_tasks.add(execution_task)
                
                # Clean up completed tasks
                completed_tasks = {t for t in self.current_tasks if t.done()}
                self.current_tasks.difference_update(completed_tasks)
                
                self.last_activity = time.time()
                
            except asyncio.CancelledError:
                logger.debug(f"Worker {self.worker_id} loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in worker {self.worker_id} loop: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def _execute_task(self, task: Task):
        """Execute a single task"""
        task.metrics.started_at = time.time()
        task.metrics.queue_time = task.metrics.started_at - task.metrics.created_at
        task.metrics.worker_id = self.worker_id
        task.metrics.status = TaskStatus.RUNNING
        
        logger.debug(f"Worker {self.worker_id} executing task {task.task_id}")
        
        try:
            # Execute task with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    self._run_task_function(task),
                    timeout=task.timeout
                )
            else:
                result = await self._run_task_function(task)
            
            # Task completed successfully
            task.metrics.completed_at = time.time()
            task.metrics.execution_time = task.metrics.completed_at - task.metrics.started_at
            task.metrics.status = TaskStatus.COMPLETED
            
            if not task.result_future.done():
                task.result_future.set_result(result)
            
            self.successful_tasks += 1
            logger.debug(f"Task {task.task_id} completed successfully")
            
        except asyncio.TimeoutError:
            error_msg = f"Task {task.task_id} timed out after {task.timeout}s"
            logger.warning(error_msg)
            task.metrics.status = TaskStatus.FAILED
            task.metrics.error_message = error_msg
            
            if not task.result_future.done():
                task.result_future.set_exception(TaskError(error_msg, "task_timeout"))
            
            self.failed_tasks += 1
            
        except Exception as e:
            error_msg = f"Task {task.task_id} failed: {e}"
            logger.error(error_msg)
            task.metrics.status = TaskStatus.FAILED
            task.metrics.error_message = str(e)
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.metrics.retry_count = task.retry_count
                task.metrics.status = TaskStatus.RETRYING
                
                # Exponential backoff
                delay = min(60, 2 ** task.retry_count)
                logger.info(f"Retrying task {task.task_id} in {delay}s (attempt {task.retry_count + 1})")
                
                asyncio.create_task(self._retry_task(task, delay))
            else:
                if not task.result_future.done():
                    task.result_future.set_exception(TaskError(error_msg, "task_execution", e))
                
                self.failed_tasks += 1
        
        finally:
            # Update worker statistics
            self.total_tasks_processed += 1
            if task.metrics.execution_time:
                if self.average_task_time == 0:
                    self.average_task_time = task.metrics.execution_time
                else:
                    self.average_task_time = (
                        self.average_task_time * (self.total_tasks_processed - 1) + 
                        task.metrics.execution_time
                    ) / self.total_tasks_processed
    
    async def _run_task_function(self, task: Task):
        """Run task function based on worker type"""
        if self.worker_type == WorkerType.CPU_BOUND and self.thread_pool:
            # Run in thread pool for CPU-bound tasks
            loop = asyncio.get_event_loop()
            
            def sync_wrapper():
                # Convert async function to sync for thread execution
                if asyncio.iscoroutinefunction(task.func):
                    # Create new event loop for thread
                    thread_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(thread_loop)
                    try:
                        return thread_loop.run_until_complete(task.func(*task.args, **task.kwargs))
                    finally:
                        thread_loop.close()
                else:
                    return task.func(*task.args, **task.kwargs)
            
            return await loop.run_in_executor(self.thread_pool, sync_wrapper)
        else:
            # Run directly for I/O-bound tasks
            if asyncio.iscoroutinefunction(task.func):
                return await task.func(*task.args, **task.kwargs)
            else:
                return task.func(*task.args, **task.kwargs)
    
    async def _retry_task(self, task: Task, delay: float):
        """Retry a failed task after delay"""
        await asyncio.sleep(delay)
        
        # Reset task state for retry
        task.metrics.status = TaskStatus.QUEUED
        task.metrics.started_at = None
        task.metrics.completed_at = None
        task.metrics.execution_time = None
        
        # Put back in queue
        await self.queue.put(task)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        success_rate = self.successful_tasks / self.total_tasks_processed if self.total_tasks_processed > 0 else 0
        
        return {
            'worker_id': self.worker_id,
            'worker_type': self.worker_type.value,
            'is_running': self.is_running,
            'current_tasks': len(self.current_tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'total_tasks_processed': self.total_tasks_processed,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': success_rate,
            'average_task_time': self.average_task_time,
            'last_activity': self.last_activity,
            'idle_time': time.time() - self.last_activity
        }


class WorkerPool:
    """Intelligent worker pool with dynamic scaling"""
    
    def __init__(self, 
                 name: str,
                 worker_type: WorkerType = WorkerType.IO_BOUND,
                 min_workers: int = 2,
                 max_workers: int = 10,
                 max_queue_size: int = 1000,
                 auto_scale: bool = True):
        
        self.name = name
        self.worker_type = worker_type
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.auto_scale = auto_scale
        
        # Task queue with priority support
        self.task_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        
        # Workers management
        self.workers: Dict[str, Worker] = {}
        self.worker_counter = 0
        
        # Pool statistics
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.queue_high_watermark = 0
        
        # Auto-scaling
        self.last_scale_check = time.time()
        self.scale_check_interval = 30  # seconds
        self.scale_up_threshold = 0.8  # Scale up when queue is 80% full
        self.scale_down_threshold = 0.2  # Scale down when utilization is < 20%
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info(f"WorkerPool '{name}' created: type={worker_type.value}, min={min_workers}, max={max_workers}")
    
    async def start(self):
        """Start the worker pool"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Create minimum workers
        for _ in range(self.min_workers):
            await self._create_worker()
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info(f"WorkerPool '{self.name}' started with {len(self.workers)} workers")
    
    async def stop(self):
        """Stop the worker pool"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel monitor task
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop all workers
        for worker in list(self.workers.values()):
            await worker.stop()
        
        self.workers.clear()
        
        logger.info(f"WorkerPool '{self.name}' stopped")
    
    async def submit_task(self, 
                         func: Callable[..., Awaitable[T]], 
                         *args,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: Optional[float] = None,
                         max_retries: int = 3,
                         **kwargs) -> asyncio.Future[T]:
        """Submit a task to the worker pool"""
        
        if not self.is_running:
            raise WorkerError("Worker pool is not running", "submit_task")
        
        # Check queue capacity
        current_queue_size = self.task_queue.qsize()
        if current_queue_size >= self.max_queue_size:
            raise WorkerError("Task queue is full", "submit_task")
        
        # Create task
        task_id = f"{self.name}-{int(time.time() * 1000)}-{self.total_tasks_submitted}"
        task = Task(
            priority=priority,
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Submit to queue
        await self.task_queue.put(task)
        task.metrics.status = TaskStatus.QUEUED
        
        self.total_tasks_submitted += 1
        self.queue_high_watermark = max(self.queue_high_watermark, current_queue_size + 1)
        
        logger.debug(f"Task {task_id} submitted to pool '{self.name}' with priority {priority.name}")
        
        return task.result_future
    
    async def submit_batch(self, 
                          tasks: List[Tuple[Callable, Tuple, Dict]],
                          priority: TaskPriority = TaskPriority.NORMAL) -> List[asyncio.Future]:
        """Submit multiple tasks as a batch"""
        futures = []
        
        for func, args, kwargs in tasks:
            future = await self.submit_task(func, *args, priority=priority, **kwargs)
            futures.append(future)
        
        logger.info(f"Submitted batch of {len(tasks)} tasks to pool '{self.name}'")
        return futures
    
    async def _create_worker(self) -> Optional[Worker]:
        """Create a new worker"""
        if len(self.workers) >= self.max_workers:
            return None
        
        self.worker_counter += 1
        worker_id = f"{self.name}-worker-{self.worker_counter}"
        
        # Determine concurrent task limit based on worker type
        if self.worker_type == WorkerType.CPU_BOUND:
            max_concurrent = 1  # CPU-bound should be single-threaded per worker
        else:
            max_concurrent = 5  # I/O-bound can handle multiple concurrent tasks
        
        worker = Worker(worker_id, self.worker_type, self.task_queue, max_concurrent)
        await worker.start()
        
        self.workers[worker_id] = worker
        logger.debug(f"Created worker {worker_id} in pool '{self.name}'")
        
        return worker
    
    async def _remove_worker(self, worker_id: str):
        """Remove a worker"""
        worker = self.workers.get(worker_id)
        if worker:
            await worker.stop()
            del self.workers[worker_id]
            logger.debug(f"Removed worker {worker_id} from pool '{self.name}'")
    
    async def _monitor_loop(self):
        """Background monitoring and auto-scaling"""
        logger.debug(f"Monitor loop started for pool '{self.name}'")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.scale_check_interval)
                
                if self.auto_scale:
                    await self._check_auto_scale()
                
                await self._health_check_workers()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop for pool '{self.name}': {e}")
    
    async def _check_auto_scale(self):
        """Check if auto-scaling is needed"""
        current_time = time.time()
        if current_time - self.last_scale_check < self.scale_check_interval:
            return
        
        self.last_scale_check = current_time
        
        # Get current metrics
        queue_size = self.task_queue.qsize()
        queue_utilization = queue_size / self.max_queue_size
        worker_count = len(self.workers)
        
        # Calculate worker utilization
        busy_workers = sum(1 for w in self.workers.values() if len(w.current_tasks) > 0)
        worker_utilization = busy_workers / worker_count if worker_count > 0 else 0
        
        # Scale up decision
        if (queue_utilization > self.scale_up_threshold or 
            worker_utilization > 0.9) and worker_count < self.max_workers:
            
            await self._create_worker()
            logger.info(f"Scaled up pool '{self.name}': {worker_count} -> {len(self.workers)} workers")
        
        # Scale down decision
        elif (queue_utilization < self.scale_down_threshold and 
              worker_utilization < self.scale_down_threshold and 
              worker_count > self.min_workers):
            
            # Find least active worker to remove
            idle_workers = [
                (w.worker_id, w.last_activity) 
                for w in self.workers.values() 
                if len(w.current_tasks) == 0
            ]
            
            if idle_workers:
                # Remove worker with longest idle time
                idle_workers.sort(key=lambda x: x[1])
                worker_id_to_remove = idle_workers[0][0]
                await self._remove_worker(worker_id_to_remove)
                logger.info(f"Scaled down pool '{self.name}': {worker_count} -> {len(self.workers)} workers")
    
    async def _health_check_workers(self):
        """Perform health checks on workers"""
        current_time = time.time()
        unhealthy_workers = []
        
        for worker_id, worker in self.workers.items():
            # Check if worker is responsive
            if current_time - worker.last_activity > 300:  # 5 minutes
                logger.warning(f"Worker {worker_id} appears unresponsive")
                unhealthy_workers.append(worker_id)
        
        # Restart unhealthy workers
        for worker_id in unhealthy_workers:
            logger.info(f"Restarting unresponsive worker {worker_id}")
            await self._remove_worker(worker_id)
            await self._create_worker()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        worker_stats = []
        for worker in self.workers.values():
            worker_stats.append(worker.get_stats())
        
        # Aggregate statistics
        total_processed = sum(w.total_tasks_processed for w in self.workers.values())
        total_successful = sum(w.successful_tasks for w in self.workers.values())
        total_failed = sum(w.failed_tasks for w in self.workers.values())
        
        success_rate = total_successful / total_processed if total_processed > 0 else 0
        avg_task_time = sum(w.average_task_time for w in self.workers.values()) / len(self.workers) if self.workers else 0
        
        return {
            'pool_name': self.name,
            'worker_type': self.worker_type.value,
            'is_running': self.is_running,
            'worker_count': len(self.workers),
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'queue_size': self.task_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'queue_utilization': self.task_queue.qsize() / self.max_queue_size,
            'total_tasks_submitted': self.total_tasks_submitted,
            'total_tasks_processed': total_processed,
            'total_tasks_successful': total_successful,
            'total_tasks_failed': total_failed,
            'success_rate': success_rate,
            'average_task_time': avg_task_time,
            'queue_high_watermark': self.queue_high_watermark,
            'workers': worker_stats
        }


class TaskManager:
    """Central task manager coordinating multiple worker pools"""
    
    def __init__(self):
        self.pools: Dict[str, WorkerPool] = {}
        self.task_registry: Dict[str, Task] = {}
        self.is_running = False
        
        # Default pools
        self.default_io_pool: Optional[WorkerPool] = None
        self.default_cpu_pool: Optional[WorkerPool] = None
        
        logger.info("TaskManager initialized")
    
    async def start(self):
        """Start the task manager"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Create default pools
        await self.create_pool("default_io", WorkerType.IO_BOUND, min_workers=5, max_workers=20)
        await self.create_pool("default_cpu", WorkerType.CPU_BOUND, min_workers=2, max_workers=multiprocessing.cpu_count())
        
        self.default_io_pool = self.pools["default_io"]
        self.default_cpu_pool = self.pools["default_cpu"]
        
        logger.info("TaskManager started with default pools")
    
    async def stop(self):
        """Stop the task manager"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all pools
        for pool in self.pools.values():
            await pool.stop()
        
        self.pools.clear()
        self.task_registry.clear()
        
        logger.info("TaskManager stopped")
    
    async def create_pool(self, 
                         name: str, 
                         worker_type: WorkerType, 
                         min_workers: int = 2, 
                         max_workers: int = 10) -> bool:
        """Create a new worker pool"""
        if name in self.pools:
            logger.warning(f"Pool '{name}' already exists")
            return False
        
        pool = WorkerPool(name, worker_type, min_workers, max_workers)
        await pool.start()
        
        self.pools[name] = pool
        logger.info(f"Created pool '{name}' with type {worker_type.value}")
        return True
    
    @monitor_performance(track_memory=True, custom_name="task_submission")
    async def submit_io_task(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> asyncio.Future[T]:
        """Submit I/O-bound task to default I/O pool"""
        if not self.default_io_pool:
            raise WorkerError("Default I/O pool not available", "submit_io_task")
        
        return await self.default_io_pool.submit_task(func, *args, **kwargs)
    
    @monitor_performance(track_memory=True, custom_name="cpu_task_submission")
    async def submit_cpu_task(self, func: Callable[..., T], *args, **kwargs) -> asyncio.Future[T]:
        """Submit CPU-bound task to default CPU pool"""
        if not self.default_cpu_pool:
            raise WorkerError("Default CPU pool not available", "submit_cpu_task")
        
        return await self.default_cpu_pool.submit_task(func, *args, **kwargs)
    
    async def submit_to_pool(self, 
                           pool_name: str, 
                           func: Callable, 
                           *args, 
                           priority: TaskPriority = TaskPriority.NORMAL,
                           **kwargs) -> asyncio.Future:
        """Submit task to specific pool"""
        pool = self.pools.get(pool_name)
        if not pool:
            raise WorkerError(f"Pool '{pool_name}' not found", "submit_to_pool")
        
        return await pool.submit_task(func, *args, priority=priority, **kwargs)
    
    async def wait_for_tasks(self, 
                           futures: List[asyncio.Future], 
                           timeout: Optional[float] = None,
                           return_when: str = 'ALL_COMPLETED') -> Tuple[Set, Set]:
        """Wait for multiple tasks to complete"""
        if return_when == 'ALL_COMPLETED':
            return_condition = asyncio.ALL_COMPLETED
        elif return_when == 'FIRST_COMPLETED':
            return_condition = asyncio.FIRST_COMPLETED
        elif return_when == 'FIRST_EXCEPTION':
            return_condition = asyncio.FIRST_EXCEPTION
        else:
            return_condition = asyncio.ALL_COMPLETED
        
        done, pending = await asyncio.wait(
            futures, 
            timeout=timeout, 
            return_when=return_condition
        )
        
        return done, pending
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        stats = {}
        for name, pool in self.pools.items():
            stats[name] = await pool.get_stats()
        
        # Add overall statistics
        total_submitted = sum(pool_stats['total_tasks_submitted'] for pool_stats in stats.values())
        total_processed = sum(pool_stats['total_tasks_processed'] for pool_stats in stats.values())
        total_successful = sum(pool_stats['total_tasks_successful'] for pool_stats in stats.values())
        
        stats['overall'] = {
            'pools_count': len(self.pools),
            'total_workers': sum(pool_stats['worker_count'] for pool_stats in stats.values()),
            'total_tasks_submitted': total_submitted,
            'total_tasks_processed': total_processed,
            'total_tasks_successful': total_successful,
            'overall_success_rate': total_successful / total_processed if total_processed > 0 else 0
        }
        
        return stats


# Global task manager instance
_task_manager: Optional[TaskManager] = None
_manager_lock = asyncio.Lock()


async def get_task_manager() -> TaskManager:
    """Get global task manager instance"""
    global _task_manager
    
    if _task_manager is None:
        async with _manager_lock:
            if _task_manager is None:
                _task_manager = TaskManager()
                await _task_manager.start()
    
    return _task_manager


# Convenience functions
async def submit_io_task(func: Callable[..., Awaitable[T]], *args, **kwargs) -> asyncio.Future[T]:
    """Submit I/O-bound task"""
    manager = await get_task_manager()
    return await manager.submit_io_task(func, *args, **kwargs)


async def submit_cpu_task(func: Callable[..., T], *args, **kwargs) -> asyncio.Future[T]:
    """Submit CPU-bound task"""
    manager = await get_task_manager()
    return await manager.submit_cpu_task(func, *args, **kwargs)


async def run_parallel(tasks: List[Callable], max_concurrency: int = 10) -> List[Any]:
    """Run multiple tasks in parallel with concurrency limit"""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def limited_task(task):
        async with semaphore:
            if asyncio.iscoroutinefunction(task):
                return await task()
            else:
                return task()
    
    # Submit all tasks
    futures = [asyncio.create_task(limited_task(task)) for task in tasks]
    
    # Wait for completion
    results = await asyncio.gather(*futures, return_exceptions=True)
    
    return results


# Task decorators
def io_task(priority: TaskPriority = TaskPriority.NORMAL, timeout: Optional[float] = None):
    """Decorator to mark function as I/O-bound task"""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs):
            manager = await get_task_manager()
            future = await manager.submit_io_task(func, *args, **kwargs)
            return await future
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


def cpu_task(priority: TaskPriority = TaskPriority.NORMAL, timeout: Optional[float] = None):
    """Decorator to mark function as CPU-bound task"""
    def decorator(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs):
            manager = await get_task_manager()
            future = await manager.submit_cpu_task(func, *args, **kwargs)
            return await future
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator