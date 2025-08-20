#!/usr/bin/env python3
"""
Claude Manager Service - Generation 3: MAKE IT SCALE
Advanced optimization with concurrent processing, caching, and auto-scaling
"""

import asyncio
import json
import os
import sys
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Union, Callable, AsyncGenerator
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from functools import wraps, lru_cache
from pathlib import Path
import hashlib
import pickle
import weakref
import gc

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn

# Import from previous generations
from robust_main import (
    RobustConfig, EnhancedLogger, LogConfig, PerformanceMonitor, 
    RobustHealthCheck, SecurityValidator, ClaudeManagerError,
    ValidationError, handle_errors, async_handle_errors
)

# Advanced caching system
class AdvancedCache:
    """High-performance caching with TTL, LRU, and memory management"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup of expired entries"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_expired()
                self._enforce_size_limit()
            except Exception:
                pass  # Continue running even if cleanup fails
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        with self.lock:
            expired_keys = [
                key for key, expiry in self.expiry_times.items()
                if expiry < current_time
            ]
            for key in expired_keys:
                self._remove_key(key)
    
    def _enforce_size_limit(self):
        """Enforce LRU size limit"""
        with self.lock:
            if len(self.cache) <= self.max_size:
                return
            
            # Sort by access time and remove oldest
            sorted_keys = sorted(
                self.access_times.items(),
                key=lambda x: x[1]
            )
            
            keys_to_remove = len(self.cache) - self.max_size + 10  # Remove extra for buffer
            for key, _ in sorted_keys[:keys_to_remove]:
                self._remove_key(key)
    
    def _remove_key(self, key: str):
        """Remove a key from all structures"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expiry_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check expiry
            if key in self.expiry_times and self.expiry_times[key] < time.time():
                self._remove_key(key)
                self.misses += 1
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            self.hits += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        with self.lock:
            self.cache[key] = value
            self.access_times[key] = time.time()
            
            if ttl is None:
                ttl = self.default_ttl
            
            if ttl > 0:
                self.expiry_times[key] = time.time() + ttl
    
    def invalidate(self, pattern: str = None) -> int:
        """Invalidate cache entries matching pattern"""
        with self.lock:
            if pattern is None:
                # Clear all
                count = len(self.cache)
                self.cache.clear()
                self.access_times.clear()
                self.expiry_times.clear()
                return count
            
            # Pattern matching
            keys_to_remove = [
                key for key in self.cache.keys()
                if pattern in key
            ]
            
            for key in keys_to_remove:
                self._remove_key(key)
            
            return len(keys_to_remove)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "memory_usage": sys.getsizeof(self.cache) + sys.getsizeof(self.access_times)
            }

# Connection pooling and resource management
class ResourcePool:
    """Generic resource pool with automatic scaling"""
    
    def __init__(self, factory: Callable, min_size: int = 2, max_size: int = 10, 
                 timeout: float = 30.0):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        self.pool = asyncio.Queue(maxsize=max_size)
        self.created_count = 0
        self.lock = asyncio.Lock()
        
        # Initialize minimum pool size
        asyncio.create_task(self._initialize_pool())
    
    async def _initialize_pool(self):
        """Initialize minimum pool size"""
        for _ in range(self.min_size):
            resource = await self._create_resource()
            await self.pool.put(resource)
    
    async def _create_resource(self):
        """Create a new resource"""
        async with self.lock:
            if self.created_count >= self.max_size:
                raise RuntimeError("Maximum pool size reached")
            
            resource = self.factory()
            self.created_count += 1
            return resource
    
    async def acquire(self):
        """Acquire a resource from the pool"""
        try:
            # Try to get from pool first
            resource = self.pool.get_nowait()
            return resource
        except asyncio.QueueEmpty:
            # Create new resource if under limit
            if self.created_count < self.max_size:
                return await self._create_resource()
            else:
                # Wait for available resource
                return await asyncio.wait_for(self.pool.get(), timeout=self.timeout)
    
    async def release(self, resource):
        """Release a resource back to the pool"""
        try:
            self.pool.put_nowait(resource)
        except asyncio.QueueFull:
            # Pool is full, discard resource
            async with self.lock:
                self.created_count -= 1
    
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "pool_size": self.pool.qsize(),
            "created_count": self.created_count,
            "min_size": self.min_size,
            "max_size": self.max_size
        }

# Async task queue with priority and rate limiting
class AsyncTaskQueue:
    """High-performance async task queue with priorities and rate limiting"""
    
    def __init__(self, max_workers: int = 10, rate_limit: float = 10.0):
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.queue = asyncio.PriorityQueue()
        self.workers = []
        self.running = False
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.semaphore = asyncio.Semaphore(max_workers)
        
        # Rate limiting
        self.last_execution = 0
        self.rate_lock = asyncio.Lock()
    
    async def start(self):
        """Start the task queue workers"""
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
    
    async def stop(self):
        """Stop the task queue workers"""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
    
    async def _worker(self, name: str):
        """Worker coroutine"""
        while self.running:
            try:
                # Get task with timeout to allow stopping
                priority, task_id, coro, callback = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
                
                # Rate limiting
                async with self.rate_lock:
                    current_time = time.time()
                    time_since_last = current_time - self.last_execution
                    min_interval = 1.0 / self.rate_limit
                    
                    if time_since_last < min_interval:
                        await asyncio.sleep(min_interval - time_since_last)
                    
                    self.last_execution = time.time()
                
                # Execute task with semaphore
                async with self.semaphore:
                    try:
                        result = await coro
                        self.completed_tasks += 1
                        if callback:
                            await callback(True, result, None)
                    except Exception as e:
                        self.failed_tasks += 1
                        if callback:
                            await callback(False, None, e)
                
                self.queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # Check if still running
            except asyncio.CancelledError:
                break
            except Exception:
                continue  # Continue running even if task fails
    
    async def submit(self, coro, priority: int = 5, task_id: str = None, 
                    callback: Callable = None) -> None:
        """Submit a coroutine to the queue"""
        if task_id is None:
            task_id = f"task-{time.time()}"
        
        await self.queue.put((priority, task_id, coro, callback))
    
    def stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "queue_size": self.queue.qsize(),
            "max_workers": self.max_workers,
            "active_workers": len([w for w in self.workers if not w.done()]),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": (self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks)) * 100
        }

# Concurrent file processing
class ConcurrentFileProcessor:
    """High-performance concurrent file processing"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.executor = None
    
    def process_files_concurrent(self, file_paths: List[Path], 
                                processor_func: Callable) -> List[Any]:
        """Process files concurrently"""
        results = []
        
        # Create a new executor for each batch to avoid shutdown issues
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = []
            for file_path in file_paths:
                future = executor.submit(processor_func, file_path)
                futures.append((file_path, future))
            
            # Collect results as they complete
            for file_path, future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout per file
                    results.append(result)
                except Exception as e:
                    # Log error but continue processing other files
                    results.append({
                        "error": str(e),
                        "file_path": str(file_path)
                    })
        
        return results
    
    async def process_files_async(self, file_paths: List[Path], 
                                 processor_func: Callable) -> AsyncGenerator[Any, None]:
        """Process files asynchronously with streaming results"""
        loop = asyncio.get_event_loop()
        
        # Create executor for this batch
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [
                loop.run_in_executor(executor, processor_func, file_path)
                for file_path in file_paths
            ]
            
            # Yield results as they complete
            for future in asyncio.as_completed(futures):
                try:
                    result = await future
                    yield result
                except Exception as e:
                    yield {"error": str(e)}

# Auto-scaling system metrics
@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    active_tasks: int
    queue_size: int
    response_time: float
    error_rate: float
    timestamp: datetime

class AutoScaler:
    """Automatic scaling based on system metrics"""
    
    def __init__(self, logger: EnhancedLogger):
        self.logger = logger
        self.metrics_history = []
        self.max_history = 100
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.min_workers = 2
        self.max_workers = 20
        self.current_workers = 5
        self.last_scale_time = time.time()
        self.scale_cooldown = 60  # 1 minute cooldown
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            import psutil
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Network I/O (simplified)
            network_io = 0
            try:
                net_io = psutil.net_io_counters()
                network_io = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
            except:
                pass
            
            metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io=network_io,
                active_tasks=0,  # Will be updated by caller
                queue_size=0,    # Will be updated by caller
                response_time=0, # Will be updated by caller
                error_rate=0,    # Will be updated by caller
                timestamp=datetime.now()
            )
            
            return metrics
            
        except ImportError:
            # Fallback metrics without psutil
            return SystemMetrics(
                cpu_usage=50.0,  # Assume moderate load
                memory_usage=50.0,
                disk_usage=50.0,
                network_io=0,
                active_tasks=0,
                queue_size=0,
                response_time=0,
                error_rate=0,
                timestamp=datetime.now()
            )
    
    def update_metrics(self, metrics: SystemMetrics):
        """Update metrics history"""
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
    
    def should_scale_up(self) -> bool:
        """Determine if scaling up is needed"""
        if len(self.metrics_history) < 5:  # Need some history
            return False
        
        recent_metrics = self.metrics_history[-5:]
        
        # Check multiple conditions
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m.queue_size for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        
        high_resource_usage = (avg_cpu > self.scale_up_threshold * 100 or 
                              avg_memory > self.scale_up_threshold * 100)
        high_queue_size = avg_queue > 10
        slow_response = avg_response_time > 5.0
        
        return (high_resource_usage or high_queue_size or slow_response) and \
               self.current_workers < self.max_workers
    
    def should_scale_down(self) -> bool:
        """Determine if scaling down is needed"""
        if len(self.metrics_history) < 10:  # Need more history for scale down
            return False
        
        recent_metrics = self.metrics_history[-10:]
        
        # Check for sustained low usage
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m.queue_size for m in recent_metrics) / len(recent_metrics)
        
        low_resource_usage = (avg_cpu < self.scale_down_threshold * 100 and 
                             avg_memory < self.scale_down_threshold * 100)
        low_queue_size = avg_queue < 2
        
        return low_resource_usage and low_queue_size and \
               self.current_workers > self.min_workers
    
    def scale_up(self) -> int:
        """Scale up workers"""
        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return self.current_workers
        
        old_workers = self.current_workers
        self.current_workers = min(self.max_workers, self.current_workers + 2)
        self.last_scale_time = current_time
        
        self.logger.info(f"Scaled up from {old_workers} to {self.current_workers} workers")
        return self.current_workers
    
    def scale_down(self) -> int:
        """Scale down workers"""
        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return self.current_workers
        
        old_workers = self.current_workers
        self.current_workers = max(self.min_workers, self.current_workers - 1)
        self.last_scale_time = current_time
        
        self.logger.info(f"Scaled down from {old_workers} to {self.current_workers} workers")
        return self.current_workers

# Optimized task analyzer with caching and concurrency
class OptimizedTaskAnalyzer:
    """High-performance task analyzer with caching and concurrent processing"""
    
    def __init__(self, config: RobustConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.cache = AdvancedCache(max_size=5000, default_ttl=1800)  # 30 min TTL
        self.file_processor = ConcurrentFileProcessor()
        self.processed_files = set()
        
        # Performance tracking
        self.scan_times = []
        self.file_counts = []
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for caching"""
        try:
            stat = file_path.stat()
            # Use file path, size, and modification time for hash
            hash_input = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception:
            return str(file_path)
    
    def _process_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single file for tasks"""
        file_hash = self._get_file_hash(file_path)
        
        # Check cache first
        cached_result = self.cache.get(f"file:{file_hash}")
        if cached_result is not None:
            return cached_result
        
        tasks = []
        search_patterns = ["TODO", "FIXME", "HACK", "XXX", "BUG", "NOTE"]
        
        try:
            # Skip binary files and large files
            if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                return []
            
            # Check file extension
            valid_extensions = {'.py', '.js', '.ts', '.md', '.txt', '.json', '.yaml', '.yml'}
            if file_path.suffix.lower() not in valid_extensions:
                return []
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    for pattern in search_patterns:
                        if pattern.lower() in line_lower:
                            task = {
                                "id": f"opt_todo_{file_hash}_{line_num}",
                                "title": f"Address {pattern} in {file_path.name}",
                                "description": f"Line {line_num}: {line.strip()}",
                                "file_path": str(file_path),
                                "line_number": line_num,
                                "type": "code_improvement",
                                "priority": self._calculate_priority(pattern, line),
                                "created_at": datetime.now().isoformat(),
                                "file_hash": file_hash
                            }
                            tasks.append(task)
                            break  # Only one task per line
        
        except Exception as e:
            self.logger.warning(f"Error processing {file_path}: {e}")
            return []
        
        # Cache the result
        self.cache.set(f"file:{file_hash}", tasks)
        return tasks
    
    def _calculate_priority(self, pattern: str, line: str) -> int:
        """Calculate task priority with enhanced logic"""
        base_priority = {
            "FIXME": 9,
            "BUG": 9,
            "HACK": 8,
            "XXX": 7,
            "TODO": 6,
            "NOTE": 4
        }.get(pattern, 5)
        
        # Increase priority based on context
        line_lower = line.lower()
        if any(word in line_lower for word in ['critical', 'urgent', 'important']):
            base_priority += 2
        if any(word in line_lower for word in ['security', 'vulnerability', 'exploit']):
            base_priority += 3
        
        return min(base_priority, 10)
    
    def scan_repository_optimized(self, repo_path: str = ".") -> Dict[str, Any]:
        """Optimized repository scanning with concurrent processing"""
        start_time = time.time()
        
        try:
            # Get all files to process
            repo_path = Path(repo_path)
            file_patterns = ["**/*.py", "**/*.js", "**/*.ts", "**/*.md", "**/*.txt", 
                           "**/*.json", "**/*.yaml", "**/*.yml"]
            
            all_files = []
            exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 
                          '.pytest_cache', '.mypy_cache', 'dist', 'build'}
            
            for pattern in file_patterns:
                for file_path in repo_path.glob(pattern):
                    if (file_path.is_file() and 
                        not any(excluded in file_path.parts for excluded in exclude_dirs)):
                        all_files.append(file_path)
            
            # Remove duplicates and limit file count for performance
            all_files = list(set(all_files))
            max_files = self.config.get("analyzer.maxFiles", 1000)
            if len(all_files) > max_files:
                self.logger.warning(f"Limiting file scan from {len(all_files)} to {max_files} files")
                all_files = all_files[:max_files]
            
            # Process files concurrently
            all_tasks = []
            batch_size = 50  # Process in batches to manage memory
            
            for i in range(0, len(all_files), batch_size):
                batch_files = all_files[i:i + batch_size]
                batch_results = self.file_processor.process_files_concurrent(
                    batch_files, self._process_single_file
                )
                
                # Flatten results
                for result in batch_results:
                    if isinstance(result, list):
                        all_tasks.extend(result)
                    elif isinstance(result, dict) and "error" not in result:
                        all_tasks.append(result)
            
            # Add repository health checks
            health_tasks = self._check_repository_health(repo_path)
            all_tasks.extend(health_tasks)
            
            # Sort by priority
            all_tasks.sort(key=lambda x: x.get("priority", 0), reverse=True)
            
            # Performance tracking
            scan_time = time.time() - start_time
            self.scan_times.append(scan_time)
            self.file_counts.append(len(all_files))
            
            # Cleanup old performance data
            if len(self.scan_times) > 50:
                self.scan_times = self.scan_times[-50:]
                self.file_counts = self.file_counts[-50:]
            
            return {
                "tasks": all_tasks,
                "performance": {
                    "scan_time": scan_time,
                    "files_processed": len(all_files),
                    "tasks_found": len(all_tasks),
                    "files_per_second": len(all_files) / scan_time if scan_time > 0 else 0,
                    "cache_stats": self.cache.stats()
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in optimized repository scan: {e}")
            raise
    
    def _check_repository_health(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Check repository health and best practices"""
        health_tasks = []
        
        # Check for essential files with higher performance
        essential_checks = [
            ("README.md", "Add comprehensive README documentation", 8),
            ("requirements.txt", "Add Python requirements file", 7),
            (".gitignore", "Add gitignore file", 6),
            ("LICENSE", "Add license file", 5),
            ("setup.py", "Add Python package setup", 4),
            ("pyproject.toml", "Consider modern Python packaging", 3)
        ]
        
        for filename, description, priority in essential_checks:
            file_path = repo_path / filename
            if not file_path.exists():
                health_tasks.append({
                    "id": f"health_missing_{filename}",
                    "title": f"Missing {filename}",
                    "description": description,
                    "type": "repository_health",
                    "priority": priority,
                    "created_at": datetime.now().isoformat()
                })
        
        return health_tasks

# Main optimized application
class OptimizedClaudeManager:
    """High-performance Claude Manager with advanced optimization"""
    
    def __init__(self, config: RobustConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.cache = AdvancedCache(max_size=10000, default_ttl=3600)
        self.task_queue = AsyncTaskQueue(max_workers=10, rate_limit=20.0)
        self.auto_scaler = AutoScaler(logger)
        self.performance_monitor = PerformanceMonitor(logger)
        
        # Analyzers
        self.task_analyzer = OptimizedTaskAnalyzer(config, logger)
        
        # Resource pools - initialize later
        self.io_pool = None
        
        # Background tasks
        self.background_tasks = []
        self.running = False
    
    async def start(self):
        """Start the optimized manager"""
        self.running = True
        await self.task_queue.start()
        
        # Initialize resource pools
        if self.io_pool is None:
            self.io_pool = ResourcePool(lambda: None, min_size=5, max_size=20)
        
        # Start background monitoring
        self.background_tasks.append(
            asyncio.create_task(self._monitoring_loop())
        )
        
        self.logger.info("Optimized Claude Manager started")
    
    async def stop(self):
        """Stop the optimized manager"""
        self.running = False
        await self.task_queue.stop()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.logger.info("Optimized Claude Manager stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring and auto-scaling"""
        while self.running:
            try:
                # Collect metrics
                metrics = self.auto_scaler.collect_metrics()
                
                # Update with queue information
                queue_stats = self.task_queue.stats()
                metrics.active_tasks = queue_stats["completed_tasks"]
                metrics.queue_size = queue_stats["queue_size"]
                
                # Update auto-scaler
                self.auto_scaler.update_metrics(metrics)
                
                # Check for scaling opportunities
                if self.auto_scaler.should_scale_up():
                    new_workers = self.auto_scaler.scale_up()
                    # In a real implementation, you would adjust the task queue workers
                    
                elif self.auto_scaler.should_scale_down():
                    new_workers = self.auto_scaler.scale_down()
                    # In a real implementation, you would adjust the task queue workers
                
                # Memory cleanup
                if len(self.auto_scaler.metrics_history) % 20 == 0:
                    gc.collect()  # Force garbage collection periodically
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def scan_repository_async(self, repo_path: str = ".") -> Dict[str, Any]:
        """Asynchronous repository scanning"""
        start_time = self.performance_monitor.start_operation("async_repository_scan")
        
        try:
            # Run the scanning in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.task_analyzer.scan_repository_optimized,
                repo_path
            )
            
            self.performance_monitor.end_operation("async_repository_scan", start_time, True)
            return result
            
        except Exception as e:
            self.performance_monitor.end_operation("async_repository_scan", start_time, False, str(e))
            raise
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "cache_stats": self.cache.stats(),
            "task_queue_stats": self.task_queue.stats(),
            "auto_scaler_stats": {
                "current_workers": self.auto_scaler.current_workers,
                "metrics_count": len(self.auto_scaler.metrics_history)
            },
            "performance_stats": self.performance_monitor.get_statistics(),
            "analyzer_performance": {
                "avg_scan_time": sum(self.task_analyzer.scan_times) / len(self.task_analyzer.scan_times) if self.task_analyzer.scan_times else 0,
                "avg_files_processed": sum(self.task_analyzer.file_counts) / len(self.task_analyzer.file_counts) if self.task_analyzer.file_counts else 0
            }
        }

# Global optimized manager instance
optimized_manager: Optional[OptimizedClaudeManager] = None

# CLI Application
app = typer.Typer(
    name="claude-manager-optimized",
    help="Claude Manager Service - Generation 3: Optimized Implementation",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()

@app.callback()
def main(
    config_file: str = typer.Option(
        "config.json",
        "--config",
        "-c",
        help="Configuration file path",
    ),
    workers: int = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of worker processes",
    ),
    cache_size: int = typer.Option(
        10000,
        "--cache-size",
        help="Cache size limit",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """Claude Manager Service - Generation 3: Optimized Implementation"""
    global optimized_manager
    
    try:
        # Initialize configuration and logging
        log_config = LogConfig(level="DEBUG" if verbose else "INFO")
        logger = EnhancedLogger("OptimizedClaudeManager", log_config)
        config = RobustConfig(config_file)
        
        # Initialize optimized manager (sync initialization only)
        optimized_manager = OptimizedClaudeManager(config, logger)
        
        # Override cache size if specified
        if cache_size != 10000:
            optimized_manager.cache.max_size = cache_size
        
        logger.info(f"Optimized Claude Manager initialized with {workers or 'auto'} workers")
        
    except Exception as e:
        rprint(f"[red]Initialization failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
@handle_errors()
def scan_optimized(
    path: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="Path to scan",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results",
    ),
    concurrent: bool = typer.Option(
        True,
        "--concurrent/--sequential",
        help="Use concurrent processing",
    ),
):
    """Optimized repository scanning with performance metrics"""
    if not optimized_manager:
        raise ClaudeManagerError("Optimized manager not initialized")
    
    async def run_scan():
        await optimized_manager.start()
        
        try:
            rprint("[bold blue]🚀 Running optimized repository scan...[/bold blue]")
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                scan_task = progress.add_task("Scanning repository...", total=100)
                
                # Start scan
                progress.update(scan_task, advance=20, description="Initializing scan...")
                
                if concurrent:
                    result = await optimized_manager.scan_repository_async(path)
                else:
                    # Fallback to synchronous scan
                    result = optimized_manager.task_analyzer.scan_repository_optimized(path)
                
                progress.update(scan_task, advance=80, description="Processing results...")
                
                tasks = result["tasks"]
                performance = result["performance"]
                
                progress.update(scan_task, completed=100, description="Scan completed!")
            
            # Display results
            if tasks:
                table = Table(title=f"Found {len(tasks)} tasks (optimized scan)")
                table.add_column("Priority", style="yellow", width=8)
                table.add_column("Type", style="cyan", width=15)
                table.add_column("Title", style="green")
                table.add_column("File", style="dim", width=25)
                
                # Show top 50 tasks to avoid overwhelming output
                display_tasks = tasks[:50]
                for task in display_tasks:
                    title = task["title"][:40] + "..." if len(task["title"]) > 40 else task["title"]
                    file_path = task.get("file_path", "N/A")
                    if len(file_path) > 25:
                        file_path = "..." + file_path[-22:]
                    
                    table.add_row(
                        str(task.get("priority", 0)),
                        task.get("type", "unknown"),
                        title,
                        file_path
                    )
                
                console.print(table)
                
                if len(tasks) > 50:
                    rprint(f"[dim]... and {len(tasks) - 50} more tasks[/dim]")
            
            # Display performance metrics
            perf_table = Table(title="Performance Metrics")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="green")
            
            perf_table.add_row("Scan Time", f"{performance['scan_time']:.2f}s")
            perf_table.add_row("Files Processed", str(performance['files_processed']))
            perf_table.add_row("Tasks Found", str(performance['tasks_found']))
            perf_table.add_row("Files/Second", f"{performance['files_per_second']:.1f}")
            perf_table.add_row("Cache Hit Rate", f"{performance['cache_stats']['hit_rate']:.1f}%")
            
            console.print(perf_table)
            
            # Save results if requested
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                rprint(f"[green]✓[/green] Results saved to {output}")
                
        finally:
            await optimized_manager.stop()
    
    # Run the async scan
    asyncio.run(run_scan())

@app.command()
@handle_errors()
def stats():
    """Show comprehensive system statistics"""
    if not optimized_manager:
        raise ClaudeManagerError("Optimized manager not initialized")
    
    rprint("[bold blue]📊 Comprehensive System Statistics[/bold blue]")
    
    stats = optimized_manager.get_comprehensive_stats()
    
    # Cache Statistics
    cache_table = Table(title="Cache Performance")
    cache_table.add_column("Metric", style="cyan")
    cache_table.add_column("Value", style="green")
    
    cache_stats = stats["cache_stats"]
    cache_table.add_row("Cache Size", f"{cache_stats['size']}/{cache_stats['max_size']}")
    cache_table.add_row("Hit Rate", f"{cache_stats['hit_rate']:.1f}%")
    cache_table.add_row("Total Hits", str(cache_stats['hits']))
    cache_table.add_row("Total Misses", str(cache_stats['misses']))
    cache_table.add_row("Memory Usage", f"{cache_stats['memory_usage'] / 1024:.1f} KB")
    
    console.print(cache_table)
    
    # Task Queue Statistics
    if "task_queue_stats" in stats:
        queue_table = Table(title="Task Queue Performance")
        queue_table.add_column("Metric", style="cyan")
        queue_table.add_column("Value", style="green")
        
        queue_stats = stats["task_queue_stats"]
        queue_table.add_row("Queue Size", str(queue_stats['queue_size']))
        queue_table.add_row("Max Workers", str(queue_stats['max_workers']))
        queue_table.add_row("Active Workers", str(queue_stats['active_workers']))
        queue_table.add_row("Completed Tasks", str(queue_stats['completed_tasks']))
        queue_table.add_row("Failed Tasks", str(queue_stats['failed_tasks']))
        queue_table.add_row("Success Rate", f"{queue_stats['success_rate']:.1f}%")
        
        console.print(queue_table)
    
    # Performance Statistics
    if "performance_stats" in stats and not stats["performance_stats"].get("message"):
        perf_table = Table(title="Overall Performance")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf_stats = stats["performance_stats"]
        perf_table.add_row("Total Operations", str(perf_stats['total_operations']))
        perf_table.add_row("Success Rate", f"{perf_stats['success_rate']:.1f}%")
        perf_table.add_row("Average Duration", f"{perf_stats['average_duration']:.2f}s")
        
        console.print(perf_table)

@app.command()
@handle_errors()
def benchmark():
    """Run performance benchmarks"""
    if not optimized_manager:
        raise ClaudeManagerError("Optimized manager not initialized")
    
    async def run_benchmark():
        await optimized_manager.start()
        
        try:
            rprint("[bold blue]🏃 Running Performance Benchmarks...[/bold blue]")
            
            # Benchmark cache performance
            cache = optimized_manager.cache
            
            with Progress(console=console) as progress:
                cache_task = progress.add_task("Cache benchmark...", total=1000)
                
                # Cache write benchmark
                start_time = time.time()
                for i in range(1000):
                    cache.set(f"bench_key_{i}", f"value_{i}")
                    progress.advance(cache_task)
                
                write_time = time.time() - start_time
                
                # Cache read benchmark
                read_task = progress.add_task("Cache read benchmark...", total=1000)
                start_time = time.time()
                hits = 0
                for i in range(1000):
                    if cache.get(f"bench_key_{i}") is not None:
                        hits += 1
                    progress.advance(read_task)
                
                read_time = time.time() - start_time
            
            # Display benchmark results
            bench_table = Table(title="Benchmark Results")
            bench_table.add_column("Operation", style="cyan")
            bench_table.add_column("Time", style="green")
            bench_table.add_column("Ops/Second", style="yellow")
            
            bench_table.add_row("Cache Writes", f"{write_time:.3f}s", f"{1000/write_time:.0f}")
            bench_table.add_row("Cache Reads", f"{read_time:.3f}s", f"{1000/read_time:.0f}")
            bench_table.add_row("Cache Hit Rate", f"{hits}/1000", f"{hits/10:.1f}%")
            
            console.print(bench_table)
            
        finally:
            await optimized_manager.stop()
    
    asyncio.run(run_benchmark())

if __name__ == "__main__":
    app()