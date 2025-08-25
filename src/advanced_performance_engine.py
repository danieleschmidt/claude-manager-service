#!/usr/bin/env python3
"""
ADVANCED PERFORMANCE ENGINE - Generation 3
High-performance optimization with caching, async processing, and intelligent resource management
"""

import asyncio
import json
import time
import hashlib
import zlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import queue
import weakref
import gc

from src.logger import get_logger

logger = get_logger(__name__)


class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class CompressionType(Enum):
    """Data compression types"""
    NONE = "none"
    GZIP = "gzip" 
    ZLIB = "zlib"
    BROTLI = "brotli"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None
    compressed: bool = False
    compression_type: Optional[CompressionType] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access information"""
        self.accessed_at = time.time()
        self.access_count += 1


class IntelligentCache:
    """High-performance intelligent caching system"""
    
    def __init__(self, 
                 max_size: int = 10000,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 default_ttl: Optional[float] = 3600,
                 compression_threshold: int = 1024,
                 enable_compression: bool = True):
        
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold
        self.enable_compression = enable_compression
        
        self._cache: Dict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._total_size = 0
        
        # Adaptive strategy metrics
        self._key_frequencies: Dict[str, float] = defaultdict(float)
        self._key_recency_scores: Dict[str, float] = defaultdict(float)
        
        # Background cleanup
        self._cleanup_interval = 60.0
        self._cleanup_task = None
        self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background cleanup task"""
        async def cleanup_task():
            while True:
                try:
                    await asyncio.sleep(self._cleanup_interval)
                    self._cleanup_expired()
                    self._optimize_cache()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_task())
        except RuntimeError:
            # No event loop running
            pass
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._total_size -= entry.size_bytes
                self._miss_count += 1
                return None
            
            # Update access info
            entry.touch()
            self._hit_count += 1
            
            # Move to end for LRU
            if self.strategy in [CacheStrategy.LRU, CacheStrategy.ADAPTIVE]:
                self._cache.move_to_end(key)
            
            # Update frequency for adaptive strategy
            if self.strategy == CacheStrategy.ADAPTIVE:
                self._update_adaptive_metrics(key)
            
            # Decompress if needed
            value = entry.value
            if entry.compressed:
                value = self._decompress(value, entry.compression_type)
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        with self._lock:
            # Calculate size
            size_estimate = self._estimate_size(value)
            
            # Compress if needed
            compressed_value = value
            compressed = False
            compression_type = None
            
            if (self.enable_compression and 
                size_estimate > self.compression_threshold):
                compressed_value, compression_type = self._compress(value)
                compressed = True
                size_estimate = self._estimate_size(compressed_value)
            
            # Remove existing entry
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_size -= old_entry.size_bytes
                del self._cache[key]
            
            # Check if we need to evict
            while (len(self._cache) >= self.max_size or 
                   self._total_size + size_estimate > self.max_size * 100):  # 100 bytes per item estimate
                
                if not self._evict_one():
                    break
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1,
                ttl=ttl or self.default_ttl,
                compressed=compressed,
                compression_type=compression_type,
                size_bytes=size_estimate
            )
            
            self._cache[key] = entry
            self._total_size += size_estimate
            
            # Update adaptive metrics
            if self.strategy == CacheStrategy.ADAPTIVE:
                self._update_adaptive_metrics(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._total_size -= entry.size_bytes
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._total_size = 0
            self._key_frequencies.clear()
            self._key_recency_scores.clear()
    
    def _evict_one(self) -> bool:
        """Evict one cache entry based on strategy"""
        if not self._cache:
            return False
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used (first item in OrderedDict)
            key = next(iter(self._cache))
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired or oldest
            now = time.time()
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        else:  # ADAPTIVE
            key = self._adaptive_eviction()
        
        entry = self._cache[key]
        self._total_size -= entry.size_bytes
        del self._cache[key]
        self._eviction_count += 1
        
        return True
    
    def _adaptive_eviction(self) -> str:
        """Intelligent adaptive eviction strategy"""
        now = time.time()
        
        # Calculate combined score: recency + frequency + size penalty
        best_key = None
        best_score = float('inf')
        
        for key, entry in self._cache.items():
            # Recency score (lower is older)
            recency_score = now - entry.accessed_at
            
            # Frequency score (higher is better, so invert)
            frequency_score = 1.0 / max(entry.access_count, 1)
            
            # Size penalty (larger items more likely to be evicted)
            size_penalty = entry.size_bytes / 1024  # KB
            
            # Adaptive weights based on cache performance
            hit_rate = self._hit_count / max(self._hit_count + self._miss_count, 1)
            
            if hit_rate > 0.8:  # Good hit rate, prefer frequency
                combined_score = frequency_score * 2 + recency_score + size_penalty * 0.5
            else:  # Poor hit rate, prefer recency  
                combined_score = recency_score * 2 + frequency_score + size_penalty * 0.5
            
            if combined_score < best_score:
                best_score = combined_score
                best_key = key
        
        return best_key or next(iter(self._cache))
    
    def _update_adaptive_metrics(self, key: str):
        """Update adaptive strategy metrics"""
        now = time.time()
        
        # Update frequency (exponential decay)
        decay_factor = 0.99
        self._key_frequencies[key] = self._key_frequencies[key] * decay_factor + 1.0
        
        # Update recency score
        self._key_recency_scores[key] = now
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired_keys:
                entry = self._cache[key]
                self._total_size -= entry.size_bytes
                del self._cache[key]
    
    def _optimize_cache(self):
        """Periodic cache optimization"""
        with self._lock:
            # Force garbage collection
            gc.collect()
            
            # Adaptive strategy tuning based on hit rate
            hit_rate = self._hit_count / max(self._hit_count + self._miss_count, 1)
            
            if hit_rate < 0.5 and self.strategy == CacheStrategy.ADAPTIVE:
                # Poor performance, clear frequency metrics to reset
                self._key_frequencies.clear()
                logger.info("Cache performance low, resetting adaptive metrics")
    
    def _compress(self, data: Any) -> Tuple[bytes, CompressionType]:
        """Compress data"""
        try:
            # Convert to JSON string first
            json_data = json.dumps(data, default=str)
            json_bytes = json_data.encode('utf-8')
            
            # Compress with zlib (good balance of speed/compression)
            compressed = zlib.compress(json_bytes, level=6)
            return compressed, CompressionType.ZLIB
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data, CompressionType.NONE
    
    def _decompress(self, data: bytes, compression_type: CompressionType) -> Any:
        """Decompress data"""
        try:
            if compression_type == CompressionType.ZLIB:
                decompressed = zlib.decompress(data)
                json_data = decompressed.decode('utf-8')
                return json.loads(json_data)
            else:
                return data
                
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return data
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            import sys
            return sys.getsizeof(obj)
        except Exception:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj) * 2  # Unicode
            elif isinstance(obj, (dict, list)):
                return len(json.dumps(obj, default=str))
            else:
                return 100  # Default estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / max(total_requests, 1)
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_size_bytes": self._total_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "eviction_count": self._eviction_count,
                "strategy": self.strategy.value,
                "compression_enabled": self.enable_compression,
                "compression_threshold": self.compression_threshold
            }


class AsyncResourcePool:
    """High-performance async resource pool"""
    
    def __init__(self, 
                 resource_factory: Callable,
                 min_size: int = 5,
                 max_size: int = 50,
                 max_idle_time: float = 300.0,
                 health_check: Optional[Callable] = None):
        
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check = health_check
        
        self._available: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._in_use: Set[Any] = set()
        self._resource_creation_times: Dict[Any, float] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._cleanup_task = None
        
    async def initialize(self):
        """Initialize the resource pool"""
        if self._initialized:
            return
        
        # Create initial resources
        for _ in range(self.min_size):
            resource = await self._create_resource()
            await self._available.put(resource)
        
        self._initialized = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def acquire(self) -> Any:
        """Acquire a resource from the pool"""
        if not self._initialized:
            await self.initialize()
        
        # Try to get an available resource
        try:
            resource = self._available.get_nowait()
            
            # Health check
            if self.health_check and not await self.health_check(resource):
                # Resource unhealthy, create a new one
                await self._destroy_resource(resource)
                resource = await self._create_resource()
            
        except asyncio.QueueEmpty:
            # No available resources, create new one if under limit
            async with self._lock:
                total_resources = len(self._in_use) + self._available.qsize()
                if total_resources < self.max_size:
                    resource = await self._create_resource()
                else:
                    # Wait for available resource
                    resource = await self._available.get()
        
        self._in_use.add(resource)
        return resource
    
    async def release(self, resource: Any):
        """Release a resource back to the pool"""
        if resource not in self._in_use:
            logger.warning("Attempting to release resource not in use")
            return
        
        self._in_use.remove(resource)
        
        # Health check before returning to pool
        if self.health_check and not await self.health_check(resource):
            await self._destroy_resource(resource)
            return
        
        # Return to available pool if not full
        try:
            self._available.put_nowait(resource)
        except asyncio.QueueFull:
            # Pool full, destroy resource
            await self._destroy_resource(resource)
    
    async def _create_resource(self) -> Any:
        """Create a new resource"""
        resource = await self.resource_factory()
        self._resource_creation_times[resource] = time.time()
        return resource
    
    async def _destroy_resource(self, resource: Any):
        """Destroy a resource"""
        if hasattr(resource, 'close'):
            if asyncio.iscoroutinefunction(resource.close):
                await resource.close()
            else:
                resource.close()
        
        self._resource_creation_times.pop(resource, None)
    
    async def _cleanup_loop(self):
        """Background cleanup of idle resources"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_resources()
            except Exception as e:
                logger.error(f"Resource pool cleanup error: {e}")
    
    async def _cleanup_idle_resources(self):
        """Clean up idle resources"""
        now = time.time()
        resources_to_cleanup = []
        
        # Check available resources for staleness
        temp_resources = []
        while not self._available.empty():
            try:
                resource = self._available.get_nowait()
                creation_time = self._resource_creation_times.get(resource, now)
                
                if now - creation_time > self.max_idle_time:
                    resources_to_cleanup.append(resource)
                else:
                    temp_resources.append(resource)
                    
            except asyncio.QueueEmpty:
                break
        
        # Put back non-stale resources
        for resource in temp_resources:
            try:
                self._available.put_nowait(resource)
            except asyncio.QueueFull:
                resources_to_cleanup.append(resource)
        
        # Destroy stale resources
        for resource in resources_to_cleanup:
            await self._destroy_resource(resource)
        
        # Ensure minimum pool size
        current_size = len(self._in_use) + self._available.qsize()
        while current_size < self.min_size:
            resource = await self._create_resource()
            try:
                self._available.put_nowait(resource)
                current_size += 1
            except asyncio.QueueFull:
                break
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "total_resources": len(self._in_use) + self._available.qsize(),
            "available": self._available.qsize(),
            "in_use": len(self._in_use),
            "min_size": self.min_size,
            "max_size": self.max_size,
            "max_idle_time": self.max_idle_time
        }
    
    async def close(self):
        """Close the resource pool"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Close all available resources
        while not self._available.empty():
            try:
                resource = self._available.get_nowait()
                await self._destroy_resource(resource)
            except asyncio.QueueEmpty:
                break
        
        # Note: Resources in use should be closed by their users


class ConcurrentTaskProcessor:
    """High-performance concurrent task processing"""
    
    def __init__(self,
                 max_workers: int = None,
                 max_concurrent_tasks: int = 100,
                 enable_process_pool: bool = False,
                 task_timeout: float = 30.0):
        
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_process_pool = enable_process_pool
        self.task_timeout = task_timeout
        
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._process_executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) if enable_process_pool else None
        
        self._task_stats = {
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "total_duration": 0.0
        }
    
    async def process_tasks_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of tasks concurrently"""
        semaphore_tasks = []
        
        for task in tasks:
            semaphore_task = self._process_single_task_with_semaphore(task)
            semaphore_tasks.append(semaphore_task)
        
        # Process all tasks concurrently
        results = await asyncio.gather(*semaphore_tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "task_id": tasks[i].get("id", f"task_{i}"),
                    "status": "error",
                    "error": str(result),
                    "original_task": tasks[i]
                })
                self._task_stats["failed"] += 1
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_task_with_semaphore(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process single task with concurrency control"""
        async with self._semaphore:
            return await self._process_single_task(task)
    
    async def _process_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task"""
        task_id = task.get("id", f"task_{id(task)}")
        start_time = time.time()
        
        try:
            # Determine execution strategy
            task_type = task.get("type", "default")
            
            if task_type == "cpu_intensive" and self._process_executor:
                # Use process pool for CPU-intensive tasks
                result = await self._run_in_process_pool(task)
            elif task_type == "io_bound":
                # Use async processing
                result = await self._run_async_task(task)
            else:
                # Use thread pool for general tasks
                result = await self._run_in_thread_pool(task)
            
            duration = time.time() - start_time
            self._task_stats["completed"] += 1
            self._task_stats["total_duration"] += duration
            
            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "duration": duration,
                "original_task": task
            }
            
        except asyncio.TimeoutError:
            self._task_stats["timeout"] += 1
            return {
                "task_id": task_id,
                "status": "timeout",
                "error": f"Task timed out after {self.task_timeout}s",
                "original_task": task
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self._task_stats["failed"] += 1
            self._task_stats["total_duration"] += duration
            
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "duration": duration,
                "original_task": task
            }
    
    async def _run_in_process_pool(self, task: Dict[str, Any]) -> Any:
        """Run task in process pool"""
        loop = asyncio.get_event_loop()
        
        def cpu_task():
            # Simulate CPU-intensive work
            result = {"message": f"CPU task processed: {task.get('id')}"}
            time.sleep(0.1)  # Simulate work
            return result
        
        future = loop.run_in_executor(self._process_executor, cpu_task)
        return await asyncio.wait_for(future, timeout=self.task_timeout)
    
    async def _run_in_thread_pool(self, task: Dict[str, Any]) -> Any:
        """Run task in thread pool"""
        loop = asyncio.get_event_loop()
        
        def thread_task():
            # Simulate general task processing
            result = {"message": f"Thread task processed: {task.get('id')}"}
            time.sleep(0.05)  # Simulate work
            return result
        
        future = loop.run_in_executor(self._thread_executor, thread_task)
        return await asyncio.wait_for(future, timeout=self.task_timeout)
    
    async def _run_async_task(self, task: Dict[str, Any]) -> Any:
        """Run async I/O bound task"""
        # Simulate async I/O work
        await asyncio.sleep(0.01)
        return {"message": f"Async task processed: {task.get('id')}"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total_tasks = sum(self._task_stats.values()) - self._task_stats["total_duration"]
        avg_duration = self._task_stats["total_duration"] / max(self._task_stats["completed"], 1)
        
        return {
            "total_processed": total_tasks,
            "completed": self._task_stats["completed"],
            "failed": self._task_stats["failed"],
            "timeout": self._task_stats["timeout"],
            "average_duration": avg_duration,
            "success_rate": self._task_stats["completed"] / max(total_tasks, 1),
            "max_workers": self.max_workers,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "process_pool_enabled": self.enable_process_pool
        }
    
    def close(self):
        """Close executors"""
        self._thread_executor.shutdown(wait=True)
        if self._process_executor:
            self._process_executor.shutdown(wait=True)


# Factory functions
def create_performance_cache(strategy: CacheStrategy = CacheStrategy.ADAPTIVE) -> IntelligentCache:
    """Create optimized cache instance"""
    return IntelligentCache(
        max_size=10000,
        strategy=strategy,
        default_ttl=3600.0,
        compression_threshold=1024,
        enable_compression=True
    )


async def create_resource_pool(factory: Callable, **kwargs) -> AsyncResourcePool:
    """Create and initialize resource pool"""
    pool = AsyncResourcePool(factory, **kwargs)
    await pool.initialize()
    return pool


def create_task_processor(**kwargs) -> ConcurrentTaskProcessor:
    """Create concurrent task processor"""
    return ConcurrentTaskProcessor(**kwargs)


# Demo
async def performance_engine_demo():
    """Demonstration of advanced performance engine"""
    logger.info("Starting advanced performance engine demo")
    
    # Test intelligent cache
    logger.info("Testing intelligent cache...")
    cache = create_performance_cache(CacheStrategy.ADAPTIVE)
    
    # Cache operations
    cache.set("test_key", {"data": "test_value", "timestamp": time.time()})
    cached_value = cache.get("test_key")
    logger.info(f"Cached value: {cached_value is not None}")
    
    stats = cache.get_stats()
    logger.info(f"Cache stats: hit_rate={stats['hit_rate']:.2f}, size={stats['size']}")
    
    # Test concurrent task processing
    logger.info("Testing concurrent task processing...")
    processor = create_task_processor(max_workers=8, max_concurrent_tasks=20)
    
    # Create test tasks
    test_tasks = [
        {"id": f"task_{i}", "type": "io_bound" if i % 2 == 0 else "default", "data": f"test_data_{i}"}
        for i in range(50)
    ]
    
    # Process tasks
    start_time = time.time()
    results = await processor.process_tasks_batch(test_tasks)
    processing_time = time.time() - start_time
    
    successful_results = [r for r in results if r["status"] == "completed"]
    logger.info(f"Processed {len(successful_results)}/{len(test_tasks)} tasks in {processing_time:.2f}s")
    
    processor_stats = processor.get_stats()
    logger.info(f"Processor stats: success_rate={processor_stats['success_rate']:.2f}")
    
    # Cleanup
    processor.close()
    
    logger.info("Advanced performance engine demo completed")


if __name__ == "__main__":
    asyncio.run(performance_engine_demo())