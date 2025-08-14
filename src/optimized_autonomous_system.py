#!/usr/bin/env python3
"""
Optimized Autonomous System - Generation 3
High-performance, scalable system with advanced optimization and auto-scaling
"""

import asyncio
import json
import logging
import time
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import collections
import hashlib
import pickle
import weakref

# Configure optimized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"
    TTL = "ttl"
    LFU = "lfu"
    ADAPTIVE = "adaptive"

class ScalingStrategy(Enum):
    """Scaling strategies"""
    CPU_BASED = "cpu_based"
    QUEUE_BASED = "queue_based"
    LATENCY_BASED = "latency_based"
    PREDICTIVE = "predictive"

class LoadBalanceStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    CONSISTENT_HASH = "consistent_hash"

@dataclass
class CacheItem:
    """Cache item with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    size: int = 0

@dataclass
class WorkerStats:
    """Worker statistics"""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    last_task_time: float = 0.0
    active_tasks: int = 0
    max_tasks: int = 10

@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    timestamp: datetime
    throughput: float  # tasks per second
    latency_p50: float
    latency_p95: float
    latency_p99: float
    cpu_usage: float
    memory_usage: float
    cache_hit_rate: float
    error_rate: float
    queue_depth: int

class IntelligentCache:
    """
    High-performance cache with multiple strategies and adaptive behavior
    """
    
    def __init__(self, max_size: int = 10000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, CacheItem] = {}
        self.access_order = collections.OrderedDict()
        self.frequency_counter = collections.Counter()
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                
                # Check TTL expiration
                if item.ttl and time.time() - item.created_at > item.ttl:
                    self._remove_item(key)
                    self.misses += 1
                    return None
                
                # Update access statistics
                item.last_accessed = time.time()
                item.access_count += 1
                
                # Update access order for LRU
                if self.strategy in [CacheStrategy.LRU, CacheStrategy.ADAPTIVE]:
                    self.access_order.move_to_end(key)
                
                # Update frequency for LFU
                if self.strategy in [CacheStrategy.LFU, CacheStrategy.ADAPTIVE]:
                    self.frequency_counter[key] += 1
                
                self.hits += 1
                return item.value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache"""
        with self.lock:
            current_time = time.time()
            
            # Calculate item size
            try:
                item_size = len(pickle.dumps(value))
            except:
                item_size = 1  # Default size if pickle fails
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl,
                size=item_size
            )
            
            # Check if we need to evict items
            while len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_item()
            
            # Add or update item
            if key in self.cache:
                old_item = self.cache[key]
                # Update existing item
                old_item.value = value
                old_item.created_at = current_time
                old_item.last_accessed = current_time
                old_item.ttl = ttl
                old_item.size = item_size
            else:
                self.cache[key] = item
                self.access_order[key] = current_time
                self.frequency_counter[key] = 1
            
            return True
    
    def _evict_item(self):
        """Evict item based on strategy"""
        if not self.cache:
            return
        
        key_to_evict = None
        
        if self.strategy == CacheStrategy.LRU:
            key_to_evict = next(iter(self.access_order))
            
        elif self.strategy == CacheStrategy.LFU:
            key_to_evict = self.frequency_counter.most_common()[-1][0]
            
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired items first, then oldest
            current_time = time.time()
            expired_keys = [
                k for k, item in self.cache.items()
                if item.ttl and current_time - item.created_at > item.ttl
            ]
            
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                key_to_evict = min(self.cache.keys(), 
                                 key=lambda k: self.cache[k].created_at)
                
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Use hit rate to decide between LRU and LFU
            hit_rate = self.hits / max(self.hits + self.misses, 1)
            
            if hit_rate > 0.8:  # High hit rate, use LRU
                key_to_evict = next(iter(self.access_order))
            else:  # Low hit rate, use LFU
                key_to_evict = self.frequency_counter.most_common()[-1][0]
        
        if key_to_evict:
            self._remove_item(key_to_evict)
    
    def _remove_item(self, key: str):
        """Remove item from cache"""
        if key in self.cache:
            del self.cache[key]
            self.access_order.pop(key, None)
            self.frequency_counter.pop(key, None)
            self.evictions += 1
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "strategy": self.strategy.value
            }

class TaskPool:
    """
    High-performance task pool with load balancing and auto-scaling
    """
    
    def __init__(self, initial_workers: int = 4, max_workers: int = 20, 
                 scaling_strategy: ScalingStrategy = ScalingStrategy.QUEUE_BASED):
        self.initial_workers = initial_workers
        self.max_workers = max_workers
        self.scaling_strategy = scaling_strategy
        
        # Worker management
        self.workers: Dict[str, WorkerStats] = {}
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        
        # Load balancing
        self.load_balancer = LoadBalancer(LoadBalanceStrategy.LEAST_CONNECTIONS)
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.task_latencies: List[float] = []
        
        # Scaling parameters
        self.last_scale_time = time.time()
        self.scale_cooldown = 30  # seconds
        self.cpu_threshold_up = 0.8
        self.cpu_threshold_down = 0.3
        self.queue_threshold_up = 50
        self.queue_threshold_down = 10
        
        self.running = False
        
    async def start(self):
        """Start the task pool"""
        self.running = True
        logger.info(f"Starting task pool with {self.initial_workers} workers")
        
        # Start initial workers
        for i in range(self.initial_workers):
            await self._add_worker(f"worker_{i}")
        
        # Start management tasks
        management_tasks = [
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._auto_scaler()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        return management_tasks
    
    async def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """Submit task to pool"""
        task_id = hashlib.md5(f"{task_func.__name__}{time.time()}".encode()).hexdigest()
        
        task_data = {
            "id": task_id,
            "function": task_func,
            "args": args,
            "kwargs": kwargs,
            "submitted_at": time.time()
        }
        
        await self.task_queue.put(task_data)
        return task_id
    
    async def _add_worker(self, worker_id: str):
        """Add a new worker"""
        worker_stats = WorkerStats(worker_id=worker_id)
        self.workers[worker_id] = worker_stats
        
        # Start worker task
        asyncio.create_task(self._worker_loop(worker_id))
        
        logger.info(f"Added worker: {worker_id}")
    
    async def _remove_worker(self, worker_id: str):
        """Remove a worker"""
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info(f"Removed worker: {worker_id}")
    
    async def _worker_loop(self, worker_id: str):
        """Main worker loop"""
        worker_stats = self.workers.get(worker_id)
        if not worker_stats:
            return
        
        while self.running and worker_id in self.workers:
            try:
                # Get task from queue
                try:
                    task_data = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process task
                await self._process_task(worker_id, task_data)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                if worker_stats:
                    worker_stats.tasks_failed += 1
                await asyncio.sleep(1)
    
    async def _process_task(self, worker_id: str, task_data: Dict[str, Any]):
        """Process individual task"""
        worker_stats = self.workers.get(worker_id)
        if not worker_stats:
            return
        
        start_time = time.time()
        worker_stats.active_tasks += 1
        
        try:
            task_func = task_data["function"]
            args = task_data.get("args", ())
            kwargs = task_data.get("kwargs", {})
            
            # Execute task
            if asyncio.iscoroutinefunction(task_func):
                result = await task_func(*args, **kwargs)
            else:
                result = task_func(*args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            worker_stats.tasks_completed += 1
            worker_stats.total_execution_time += execution_time
            worker_stats.avg_execution_time = (
                worker_stats.total_execution_time / worker_stats.tasks_completed
            )
            worker_stats.last_task_time = time.time()
            
            # Track latency
            total_latency = time.time() - task_data["submitted_at"]
            self.task_latencies.append(total_latency)
            
            # Limit latency history
            if len(self.task_latencies) > 1000:
                self.task_latencies = self.task_latencies[-1000:]
            
            # Put result in queue
            await self.result_queue.put({
                "task_id": task_data["id"],
                "result": result,
                "success": True,
                "execution_time": execution_time
            })
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            worker_stats.tasks_failed += 1
            
            await self.result_queue.put({
                "task_id": task_data["id"],
                "error": str(e),
                "success": False,
                "execution_time": time.time() - start_time
            })
        finally:
            worker_stats.active_tasks -= 1
    
    async def _performance_monitor(self):
        """Monitor performance metrics"""
        while self.running:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(1)
    
    async def _collect_performance_metrics(self):
        """Collect current performance metrics"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Calculate throughput
            completed_tasks = sum(w.tasks_completed for w in self.workers.values())
            time_window = 60  # 1 minute window
            
            # Get recent metrics for throughput calculation
            recent_metrics = [
                m for m in self.performance_history
                if (current_time - m.timestamp).total_seconds() < time_window
            ]
            
            if recent_metrics:
                throughput = (completed_tasks - sum(m.throughput for m in recent_metrics)) / time_window
            else:
                throughput = 0
            
            # Calculate latency percentiles
            if self.task_latencies:
                sorted_latencies = sorted(self.task_latencies)
                n = len(sorted_latencies)
                p50 = sorted_latencies[int(n * 0.5)] if n > 0 else 0
                p95 = sorted_latencies[int(n * 0.95)] if n > 0 else 0
                p99 = sorted_latencies[int(n * 0.99)] if n > 0 else 0
            else:
                p50 = p95 = p99 = 0
            
            # System metrics (simplified)
            try:
                import psutil
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
            except ImportError:
                cpu_usage = memory_usage = 0
            
            # Error rate
            total_tasks = sum(w.tasks_completed + w.tasks_failed for w in self.workers.values())
            failed_tasks = sum(w.tasks_failed for w in self.workers.values())
            error_rate = failed_tasks / max(total_tasks, 1)
            
            metrics = PerformanceMetrics(
                timestamp=current_time,
                throughput=throughput,
                latency_p50=p50,
                latency_p95=p95,
                latency_p99=p99,
                cpu_usage=cpu_usage / 100,
                memory_usage=memory_usage / 100,
                cache_hit_rate=0,  # Would be populated by cache
                error_rate=error_rate,
                queue_depth=self.task_queue.qsize()
            )
            
            self.performance_history.append(metrics)
            
            # Limit history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    async def _auto_scaler(self):
        """Auto-scaling based on metrics"""
        while self.running:
            try:
                await self._check_scaling_conditions()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(1)
    
    async def _check_scaling_conditions(self):
        """Check if scaling is needed"""
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return
        
        current_workers = len(self.workers)
        queue_depth = self.task_queue.qsize()
        
        scale_up = False
        scale_down = False
        
        if self.scaling_strategy == ScalingStrategy.QUEUE_BASED:
            if queue_depth > self.queue_threshold_up and current_workers < self.max_workers:
                scale_up = True
            elif queue_depth < self.queue_threshold_down and current_workers > self.initial_workers:
                scale_down = True
        
        elif self.scaling_strategy == ScalingStrategy.CPU_BASED:
            if self.performance_history:
                recent_cpu = self.performance_history[-1].cpu_usage
                if recent_cpu > self.cpu_threshold_up and current_workers < self.max_workers:
                    scale_up = True
                elif recent_cpu < self.cpu_threshold_down and current_workers > self.initial_workers:
                    scale_down = True
        
        elif self.scaling_strategy == ScalingStrategy.LATENCY_BASED:
            if self.performance_history:
                recent_p95 = self.performance_history[-1].latency_p95
                if recent_p95 > 5.0 and current_workers < self.max_workers:  # 5 second threshold
                    scale_up = True
                elif recent_p95 < 1.0 and current_workers > self.initial_workers:
                    scale_down = True
        
        if scale_up:
            await self._scale_up()
        elif scale_down:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up workers"""
        new_worker_id = f"worker_{len(self.workers)}"
        await self._add_worker(new_worker_id)
        self.last_scale_time = time.time()
        logger.info(f"Scaled up to {len(self.workers)} workers")
    
    async def _scale_down(self):
        """Scale down workers"""
        if len(self.workers) > self.initial_workers:
            # Find worker with least activity
            least_active_worker = min(
                self.workers.items(),
                key=lambda x: x[1].tasks_completed
            )[0]
            
            await self._remove_worker(least_active_worker)
            self.last_scale_time = time.time()
            logger.info(f"Scaled down to {len(self.workers)} workers")
    
    async def _metrics_collector(self):
        """Collect and log metrics periodically"""
        while self.running:
            try:
                stats = self.get_stats()
                logger.info(f"Pool stats - Workers: {stats['workers']}, Queue: {stats['queue_depth']}, "
                           f"Throughput: {stats['recent_throughput']:.2f} tasks/s")
                await asyncio.sleep(60)  # Log every minute
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task pool statistics"""
        total_completed = sum(w.tasks_completed for w in self.workers.values())
        total_failed = sum(w.tasks_failed for w in self.workers.values())
        
        recent_throughput = 0
        if self.performance_history:
            recent_throughput = self.performance_history[-1].throughput
        
        return {
            "workers": len(self.workers),
            "queue_depth": self.task_queue.qsize(),
            "tasks_completed": total_completed,
            "tasks_failed": total_failed,
            "recent_throughput": recent_throughput,
            "performance_history_size": len(self.performance_history)
        }

class LoadBalancer:
    """Load balancer for distributing tasks"""
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.current_index = 0
        self.connection_counts: Dict[str, int] = {}
        self.weights: Dict[str, float] = {}
    
    def select_worker(self, workers: Dict[str, WorkerStats]) -> Optional[str]:
        """Select worker based on load balancing strategy"""
        if not workers:
            return None
        
        worker_ids = list(workers.keys())
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            selected = worker_ids[self.current_index % len(worker_ids)]
            self.current_index += 1
            return selected
        
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(worker_ids, key=lambda w: workers[w].active_tasks)
        
        elif self.strategy == LoadBalanceStrategy.WEIGHTED:
            # Implement weighted selection based on performance
            total_weight = sum(self.weights.get(w, 1.0) for w in worker_ids)
            if total_weight > 0:
                import random
                r = random.random() * total_weight
                cumulative = 0
                for worker_id in worker_ids:
                    cumulative += self.weights.get(worker_id, 1.0)
                    if r <= cumulative:
                        return worker_id
            return worker_ids[0]
        
        else:
            return worker_ids[0]

class OptimizedAutonomousSystem:
    """
    Optimized autonomous system with caching, load balancing, and auto-scaling
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Core components
        self.cache = IntelligentCache(
            max_size=self.config.get("cache_size", 10000),
            strategy=CacheStrategy(self.config.get("cache_strategy", "adaptive"))
        )
        
        self.task_pool = TaskPool(
            initial_workers=self.config.get("initial_workers", 4),
            max_workers=self.config.get("max_workers", 20),
            scaling_strategy=ScalingStrategy(self.config.get("scaling_strategy", "queue_based"))
        )
        
        # Performance optimization
        self.connection_pool = None  # Would be implemented for DB connections
        self.request_cache = {}
        self.batch_operations = []
        
        # State management
        self.running = False
        self.start_time = time.time()
        
        logger.info("Optimized Autonomous System initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for optimized system"""
        return {
            "cache_size": 10000,
            "cache_strategy": "adaptive",
            "initial_workers": 4,
            "max_workers": 20,
            "scaling_strategy": "queue_based",
            "enable_caching": True,
            "enable_batch_processing": True,
            "batch_size": 50,
            "batch_timeout": 5.0,
            "connection_pool_size": 10,
            "optimization_level": "high"
        }
    
    @lru_cache(maxsize=1000)
    def _cached_computation(self, input_data: str) -> str:
        """Example cached computation"""
        # Simulate expensive computation
        time.sleep(0.01)
        return f"computed_{hashlib.md5(input_data.encode()).hexdigest()[:8]}"
    
    async def start_system(self):
        """Start the optimized system"""
        self.running = True
        logger.info("Starting Optimized Autonomous System")
        
        # Start task pool
        pool_tasks = await self.task_pool.start()
        
        # Start optimization tasks
        optimization_tasks = [
            asyncio.create_task(self._batch_processor()),
            asyncio.create_task(self._cache_optimizer()),
            asyncio.create_task(self._performance_optimizer())
        ]
        
        all_tasks = pool_tasks + optimization_tasks
        
        try:
            await asyncio.gather(*all_tasks)
        except asyncio.CancelledError:
            logger.info("System tasks cancelled")
        finally:
            self.running = False
    
    async def process_request(self, request_type: str, data: Any) -> Any:
        """Process request with optimizations"""
        # Generate cache key
        cache_key = f"{request_type}_{hashlib.md5(str(data).encode()).hexdigest()}"
        
        # Check cache first
        if self.config.get("enable_caching", True):
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Submit task to pool
        task_id = await self.task_pool.submit_task(self._process_data, request_type, data)
        
        # Wait for result (simplified - would use proper async result handling)
        result = await self._get_task_result(task_id)
        
        # Cache result
        if self.config.get("enable_caching", True) and result:
            self.cache.put(cache_key, result, ttl=300)  # 5 minute TTL
        
        return result
    
    async def _process_data(self, request_type: str, data: Any) -> Any:
        """Process data (placeholder for actual processing)"""
        # Simulate different processing based on type
        if request_type == "analysis":
            await asyncio.sleep(0.1)  # Simulate analysis
            return {"analysis_result": f"analyzed_{data}", "timestamp": time.time()}
        elif request_type == "transformation":
            await asyncio.sleep(0.05)  # Simulate transformation
            return {"transformed_data": f"transformed_{data}"}
        else:
            await asyncio.sleep(0.02)  # Default processing
            return {"processed": True, "data": data}
    
    async def _get_task_result(self, task_id: str) -> Any:
        """Get result from task (simplified implementation)"""
        # In a real implementation, this would properly track results
        # For demo, we'll just wait a bit and return a dummy result
        await asyncio.sleep(0.1)
        return {"task_id": task_id, "completed": True}
    
    async def _batch_processor(self):
        """Process operations in batches for efficiency"""
        if not self.config.get("enable_batch_processing", True):
            return
        
        while self.running:
            try:
                if len(self.batch_operations) >= self.config.get("batch_size", 50):
                    await self._process_batch()
                
                await asyncio.sleep(self.config.get("batch_timeout", 5.0))
                
                # Process remaining operations after timeout
                if self.batch_operations:
                    await self._process_batch()
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self):
        """Process batched operations"""
        if not self.batch_operations:
            return
        
        logger.info(f"Processing batch of {len(self.batch_operations)} operations")
        
        # Process operations in batch (placeholder)
        processed_count = len(self.batch_operations)
        self.batch_operations.clear()
        
        logger.info(f"Processed {processed_count} operations in batch")
    
    async def _cache_optimizer(self):
        """Optimize cache performance"""
        while self.running:
            try:
                stats = self.cache.get_stats()
                
                # Log cache stats
                if stats["hit_rate"] < 0.7:  # Low hit rate
                    logger.warning(f"Low cache hit rate: {stats['hit_rate']:.2f}")
                
                # Adaptive cache size adjustment
                if stats["size"] > stats["max_size"] * 0.9:  # Cache nearly full
                    # Could implement cache size expansion here
                    pass
                
                await asyncio.sleep(60)  # Optimize every minute
                
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
                await asyncio.sleep(5)
    
    async def _performance_optimizer(self):
        """Monitor and optimize performance"""
        while self.running:
            try:
                # Get performance metrics
                pool_stats = self.task_pool.get_stats()
                cache_stats = self.cache.get_stats()
                
                # Performance analysis
                if pool_stats["queue_depth"] > 100:
                    logger.warning("High queue depth detected - consider scaling up")
                
                if cache_stats["hit_rate"] > 0.9:
                    logger.info("Excellent cache performance")
                
                # Could implement automatic optimizations here
                
                await asyncio.sleep(30)  # Optimize every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance optimization error: {e}")
                await asyncio.sleep(5)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            "uptime": time.time() - self.start_time,
            "cache_stats": self.cache.get_stats(),
            "task_pool_stats": self.task_pool.get_stats(),
            "system_status": "running" if self.running else "stopped"
        }
    
    def stop_system(self):
        """Stop the optimized system"""
        self.running = False
        logger.info("Stopping Optimized Autonomous System")

# Example usage and testing
async def test_optimized_system():
    """Test the optimized autonomous system"""
    print("🚀 Testing Optimized Autonomous System - Generation 3")
    print("High-performance with caching, scaling, and optimization")
    print("-" * 60)
    
    system = OptimizedAutonomousSystem()
    
    # Start system in background
    system_task = asyncio.create_task(system.start_system())
    
    # Let system initialize
    await asyncio.sleep(2)
    
    # Test requests
    print("Processing test requests...")
    
    start_time = time.time()
    
    # Submit multiple requests
    tasks = []
    for i in range(20):
        task = asyncio.create_task(
            system.process_request("analysis", f"test_data_{i}")
        )
        tasks.append(task)
    
    # Wait for all requests
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"Processed {len(results)} requests in {end_time - start_time:.2f} seconds")
    
    # Get system metrics
    metrics = system.get_system_metrics()
    print(f"\nSystem Metrics:")
    print(f"  Uptime: {metrics['uptime']:.1f}s")
    print(f"  Cache Hit Rate: {metrics['cache_stats']['hit_rate']:.2f}")
    print(f"  Active Workers: {metrics['task_pool_stats']['workers']}")
    print(f"  Tasks Completed: {metrics['task_pool_stats']['tasks_completed']}")
    print(f"  Queue Depth: {metrics['task_pool_stats']['queue_depth']}")
    
    # Test caching performance
    print(f"\nTesting cache performance...")
    cache_start = time.time()
    
    # Repeat some requests to test caching
    cached_tasks = []
    for i in range(10):
        task = asyncio.create_task(
            system.process_request("analysis", f"test_data_{i % 5}")  # Repeat data
        )
        cached_tasks.append(task)
    
    await asyncio.gather(*cached_tasks)
    cache_end = time.time()
    
    print(f"Cached requests took {cache_end - cache_start:.2f} seconds")
    
    final_cache_stats = system.cache.get_stats()
    print(f"Final cache hit rate: {final_cache_stats['hit_rate']:.2f}")
    
    # Stop system
    system.stop_system()
    system_task.cancel()
    
    try:
        await system_task
    except asyncio.CancelledError:
        pass
    
    print("✅ Optimized system test completed")

if __name__ == "__main__":
    asyncio.run(test_optimized_system())