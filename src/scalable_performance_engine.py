#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - SCALABLE PERFORMANCE ENGINE
Advanced performance optimization, caching, concurrency, and auto-scaling
"""

import asyncio
import json
import time
import hashlib
import pickle
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import weakref

import aioredis
import structlog
from cachetools import TTLCache, LRUCache
import uvloop


T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"          # Least Recently Used
    TTL = "ttl"          # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    DISTRIBUTED = "distributed"


class PerformanceLevel(Enum):
    """Performance optimization levels"""
    BASIC = "basic"
    OPTIMIZED = "optimized"
    HIGH_PERFORMANCE = "high_performance"
    EXTREME = "extreme"


@dataclass
class CacheConfig:
    """Cache configuration"""
    strategy: CacheStrategy
    max_size: int = 1000
    ttl_seconds: int = 3600
    redis_url: Optional[str] = None
    compression: bool = False
    serialization: str = "pickle"  # pickle, json, msgpack
    
    
@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    operation_name: str
    execution_time: float
    memory_usage: float
    cpu_time: float
    cache_hits: int = 0
    cache_misses: int = 0
    concurrent_operations: int = 0
    throughput: float = 0.0  # operations per second
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ResourceUsage:
    """System resource usage tracking"""
    cpu_percent: float
    memory_mb: float
    disk_io_mb: float
    network_io_mb: float
    open_files: int
    threads: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IntelligentCache(Generic[T]):
    """Multi-strategy intelligent caching system"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = structlog.get_logger("IntelligentCache")
        
        # Local caches
        self.local_cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        
        # Redis connection for distributed caching
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Initialize cache based on strategy
        self._initialize_cache()
        
    def _initialize_cache(self):
        """Initialize cache based on strategy"""
        
        if self.config.strategy == CacheStrategy.LRU:
            self.local_cache = LRUCache(maxsize=self.config.max_size)
        elif self.config.strategy == CacheStrategy.TTL:
            self.local_cache = TTLCache(maxsize=self.config.max_size, ttl=self.config.ttl_seconds)
        elif self.config.strategy == CacheStrategy.DISTRIBUTED:
            # Initialize Redis connection
            asyncio.create_task(self._connect_redis())
    
    async def _connect_redis(self):
        """Connect to Redis for distributed caching"""
        
        if self.config.redis_url:
            try:
                self.redis_client = await aioredis.from_url(self.config.redis_url)
                self.logger.info("Connected to Redis for distributed caching")
            except Exception as e:
                self.logger.error("Failed to connect to Redis", error=str(e))
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        
        # Try local cache first
        if key in self.local_cache:
            self.hit_count += 1
            self.access_times[key] = datetime.now(timezone.utc)
            self.logger.debug("Cache hit (local)", key=key)
            return self.local_cache[key]
        
        # Try distributed cache if available
        if self.redis_client and self.config.strategy == CacheStrategy.DISTRIBUTED:
            try:
                value = await self.redis_client.get(key)
                if value:
                    self.hit_count += 1
                    # Deserialize
                    deserialized_value = self._deserialize(value)
                    # Store in local cache for faster access
                    self.local_cache[key] = deserialized_value
                    self.access_times[key] = datetime.now(timezone.utc)
                    self.logger.debug("Cache hit (distributed)", key=key)
                    return deserialized_value
            except Exception as e:
                self.logger.error("Redis cache get error", key=key, error=str(e))
        
        self.miss_count += 1
        self.logger.debug("Cache miss", key=key)
        return None
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None):
        """Set value in cache"""
        
        try:
            # Store in local cache
            self.local_cache[key] = value
            self.access_times[key] = datetime.now(timezone.utc)
            
            # Store in distributed cache if available
            if self.redis_client and self.config.strategy == CacheStrategy.DISTRIBUTED:
                serialized_value = self._serialize(value)
                effective_ttl = ttl or self.config.ttl_seconds
                await self.redis_client.setex(key, effective_ttl, serialized_value)
            
            self.logger.debug("Cache set", key=key)
            
        except Exception as e:
            self.logger.error("Cache set error", key=key, error=str(e))
    
    async def delete(self, key: str):
        """Delete value from cache"""
        
        # Remove from local cache
        if key in self.local_cache:
            del self.local_cache[key]
        if key in self.access_times:
            del self.access_times[key]
        
        # Remove from distributed cache
        if self.redis_client and self.config.strategy == CacheStrategy.DISTRIBUTED:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                self.logger.error("Redis cache delete error", key=key, error=str(e))
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        
        if self.config.serialization == "pickle":
            return pickle.dumps(value)
        elif self.config.serialization == "json":
            return json.dumps(value).encode('utf-8')
        else:
            return pickle.dumps(value)  # fallback
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        
        if self.config.serialization == "pickle":
            return pickle.loads(data)
        elif self.config.serialization == "json":
            return json.loads(data.decode('utf-8'))
        else:
            return pickle.loads(data)  # fallback
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.local_cache),
            'max_size': self.config.max_size,
            'strategy': self.config.strategy.value
        }


class PerformanceOptimizer:
    """Advanced performance optimization engine"""
    
    def __init__(self):
        self.logger = structlog.get_logger("PerformanceOptimizer")
        self.performance_level = PerformanceLevel.BASIC
        self.metrics: List[PerformanceMetrics] = []
        self.resource_usage: List[ResourceUsage] = []
        self.optimization_strategies: Dict[str, Callable] = {}
        
        # Performance tracking
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.bottlenecks: Dict[str, float] = {}
        
        # Auto-optimization configuration
        self.auto_optimization_enabled = True
        self.optimization_threshold = 2.0  # seconds
        self.optimization_interval = 300   # 5 minutes
        
        # Register built-in optimization strategies
        self._register_optimization_strategies()
    
    def _register_optimization_strategies(self):
        """Register built-in optimization strategies"""
        
        self.optimization_strategies.update({
            'cpu_intensive': self._optimize_cpu_intensive,
            'io_intensive': self._optimize_io_intensive,
            'memory_intensive': self._optimize_memory_intensive,
            'network_intensive': self._optimize_network_intensive,
            'database_intensive': self._optimize_database_intensive
        })
    
    async def optimize_performance(self, operation_name: str, target_function: Callable, 
                                 optimization_level: PerformanceLevel = None) -> Callable:
        """Optimize performance of a function"""
        
        level = optimization_level or self.performance_level
        
        # Analyze function characteristics
        characteristics = await self._analyze_function_characteristics(target_function)
        
        # Apply optimizations based on characteristics and level
        optimized_function = await self._apply_optimizations(
            target_function, characteristics, level
        )
        
        # Wrap with performance monitoring
        monitored_function = self._wrap_with_monitoring(operation_name, optimized_function)
        
        return monitored_function
    
    async def _analyze_function_characteristics(self, func: Callable) -> Dict[str, Any]:
        """Analyze function characteristics to determine optimization strategy"""
        
        # This would use static analysis, profiling, and ML in a full implementation
        # For now, return basic characteristics
        
        characteristics = {
            'cpu_intensive': False,
            'io_intensive': False,
            'memory_intensive': False,
            'network_intensive': False,
            'database_intensive': False,
            'cacheable': True,
            'parallelizable': True,
            'async_compatible': asyncio.iscoroutinefunction(func)
        }
        
        # Simple heuristics based on function name and docstring
        func_name = func.__name__.lower()
        func_doc = (func.__doc__ or "").lower()
        
        if any(keyword in func_name + func_doc for keyword in ['compute', 'calculate', 'process', 'analyze']):
            characteristics['cpu_intensive'] = True
        
        if any(keyword in func_name + func_doc for keyword in ['read', 'write', 'file', 'disk']):
            characteristics['io_intensive'] = True
        
        if any(keyword in func_name + func_doc for keyword in ['network', 'http', 'api', 'request']):
            characteristics['network_intensive'] = True
        
        if any(keyword in func_name + func_doc for keyword in ['database', 'db', 'sql', 'query']):
            characteristics['database_intensive'] = True
        
        return characteristics
    
    async def _apply_optimizations(self, func: Callable, characteristics: Dict[str, Any], 
                                 level: PerformanceLevel) -> Callable:
        """Apply optimizations based on function characteristics"""
        
        optimized_func = func
        
        # Apply caching if function is cacheable
        if characteristics.get('cacheable', False):
            optimized_func = await self._apply_intelligent_caching(optimized_func)
        
        # Apply concurrency optimizations
        if characteristics.get('parallelizable', False):
            if characteristics.get('cpu_intensive', False):
                optimized_func = await self._apply_cpu_parallelization(optimized_func, level)
            elif characteristics.get('io_intensive', False):
                optimized_func = await self._apply_io_parallelization(optimized_func, level)
        
        # Apply memory optimizations
        if characteristics.get('memory_intensive', False):
            optimized_func = await self._apply_memory_optimizations(optimized_func, level)
        
        # Apply network optimizations
        if characteristics.get('network_intensive', False):
            optimized_func = await self._apply_network_optimizations(optimized_func, level)
        
        return optimized_func
    
    async def _apply_intelligent_caching(self, func: Callable) -> Callable:
        """Apply intelligent caching to function"""
        
        cache_config = CacheConfig(
            strategy=CacheStrategy.LRU,
            max_size=1000,
            ttl_seconds=3600
        )
        cache = IntelligentCache(cache_config)
        
        if asyncio.iscoroutinefunction(func):
            async def cached_async_func(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                result = await cache.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await cache.set(cache_key, result)
                return result
            
            return cached_async_func
        else:
            def cached_sync_func(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache (simplified for sync)
                # In real implementation, would use sync cache or async context
                result = func(*args, **kwargs)
                return result
            
            return cached_sync_func
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function arguments"""
        
        # Create deterministic hash of arguments
        key_data = {
            'function': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _apply_cpu_parallelization(self, func: Callable, level: PerformanceLevel) -> Callable:
        """Apply CPU parallelization optimizations"""
        
        # Determine number of workers based on optimization level
        max_workers = {
            PerformanceLevel.BASIC: multiprocessing.cpu_count(),
            PerformanceLevel.OPTIMIZED: multiprocessing.cpu_count() * 2,
            PerformanceLevel.HIGH_PERFORMANCE: multiprocessing.cpu_count() * 4,
            PerformanceLevel.EXTREME: multiprocessing.cpu_count() * 8
        }.get(level, multiprocessing.cpu_count())
        
        if asyncio.iscoroutinefunction(func):
            async def parallelized_async_func(*args, **kwargs):
                # For async functions, use ThreadPoolExecutor
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Convert to sync function for thread execution
                    def sync_wrapper():
                        return asyncio.run(func(*args, **kwargs))
                    
                    result = await loop.run_in_executor(executor, sync_wrapper)
                    return result
            
            return parallelized_async_func
        else:
            def parallelized_sync_func(*args, **kwargs):
                # For CPU-intensive sync functions, use ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result()
            
            return parallelized_sync_func
    
    async def _apply_io_parallelization(self, func: Callable, level: PerformanceLevel) -> Callable:
        """Apply I/O parallelization optimizations"""
        
        max_workers = {
            PerformanceLevel.BASIC: 10,
            PerformanceLevel.OPTIMIZED: 50,
            PerformanceLevel.HIGH_PERFORMANCE: 100,
            PerformanceLevel.EXTREME: 200
        }.get(level, 10)
        
        if asyncio.iscoroutinefunction(func):
            async def io_optimized_async_func(*args, **kwargs):
                # Use asyncio semaphore for concurrency control
                semaphore = asyncio.Semaphore(max_workers)
                async with semaphore:
                    return await func(*args, **kwargs)
            
            return io_optimized_async_func
        else:
            def io_optimized_sync_func(*args, **kwargs):
                # Use ThreadPoolExecutor for I/O-bound operations
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result()
            
            return io_optimized_sync_func
    
    async def _apply_memory_optimizations(self, func: Callable, level: PerformanceLevel) -> Callable:
        """Apply memory optimizations"""
        
        # Implement memory pooling, object reuse, etc.
        # This is a placeholder implementation
        return func
    
    async def _apply_network_optimizations(self, func: Callable, level: PerformanceLevel) -> Callable:
        """Apply network optimizations"""
        
        # Implement connection pooling, request batching, etc.
        # This is a placeholder implementation
        return func
    
    def _wrap_with_monitoring(self, operation_name: str, func: Callable) -> Callable:
        """Wrap function with performance monitoring"""
        
        if asyncio.iscoroutinefunction(func):
            async def monitored_async_func(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record successful execution metrics
                    end_time = time.time()
                    execution_time = end_time - start_time
                    memory_delta = self._get_memory_usage() - start_memory
                    
                    metrics = PerformanceMetrics(
                        operation_name=operation_name,
                        execution_time=execution_time,
                        memory_usage=memory_delta,
                        cpu_time=execution_time,  # Simplified
                        throughput=1.0 / execution_time if execution_time > 0 else 0.0
                    )
                    
                    self.metrics.append(metrics)
                    self.operation_times[operation_name].append(execution_time)
                    
                    # Check for performance bottlenecks
                    await self._check_performance_bottlenecks(operation_name, execution_time)
                    
                    return result
                    
                except Exception as e:
                    # Record error metrics
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    self.logger.error("Operation failed", 
                                    operation=operation_name,
                                    execution_time=execution_time,
                                    error=str(e))
                    raise
            
            return monitored_async_func
        else:
            def monitored_sync_func(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    memory_delta = self._get_memory_usage() - start_memory
                    
                    metrics = PerformanceMetrics(
                        operation_name=operation_name,
                        execution_time=execution_time,
                        memory_usage=memory_delta,
                        cpu_time=execution_time,
                        throughput=1.0 / execution_time if execution_time > 0 else 0.0
                    )
                    
                    self.metrics.append(metrics)
                    self.operation_times[operation_name].append(execution_time)
                    
                    return result
                    
                except Exception as e:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    self.logger.error("Operation failed",
                                    operation=operation_name,
                                    execution_time=execution_time,
                                    error=str(e))
                    raise
            
            return monitored_sync_func
    
    async def _check_performance_bottlenecks(self, operation_name: str, execution_time: float):
        """Check for performance bottlenecks and trigger optimizations"""
        
        if execution_time > self.optimization_threshold:
            self.bottlenecks[operation_name] = execution_time
            
            if self.auto_optimization_enabled:
                await self._trigger_auto_optimization(operation_name)
    
    async def _trigger_auto_optimization(self, operation_name: str):
        """Trigger automatic optimization for slow operations"""
        
        self.logger.warning("Performance bottleneck detected, triggering optimization",
                          operation=operation_name,
                          execution_time=self.bottlenecks[operation_name])
        
        # This would implement automatic optimization strategies
        # For now, just log the event
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    # Optimization strategy implementations
    
    async def _optimize_cpu_intensive(self, func: Callable) -> Callable:
        """Optimize CPU-intensive operations"""
        return await self._apply_cpu_parallelization(func, PerformanceLevel.HIGH_PERFORMANCE)
    
    async def _optimize_io_intensive(self, func: Callable) -> Callable:
        """Optimize I/O-intensive operations"""
        return await self._apply_io_parallelization(func, PerformanceLevel.HIGH_PERFORMANCE)
    
    async def _optimize_memory_intensive(self, func: Callable) -> Callable:
        """Optimize memory-intensive operations"""
        return await self._apply_memory_optimizations(func, PerformanceLevel.HIGH_PERFORMANCE)
    
    async def _optimize_network_intensive(self, func: Callable) -> Callable:
        """Optimize network-intensive operations"""
        return await self._apply_network_optimizations(func, PerformanceLevel.HIGH_PERFORMANCE)
    
    async def _optimize_database_intensive(self, func: Callable) -> Callable:
        """Optimize database-intensive operations"""
        # Combine caching and connection pooling
        cached_func = await self._apply_intelligent_caching(func)
        return await self._apply_network_optimizations(cached_func, PerformanceLevel.HIGH_PERFORMANCE)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not self.metrics:
            return {'status': 'no_data'}
        
        # Calculate aggregate statistics
        total_operations = len(self.metrics)
        avg_execution_time = sum(m.execution_time for m in self.metrics) / total_operations
        max_execution_time = max(m.execution_time for m in self.metrics)
        min_execution_time = min(m.execution_time for m in self.metrics)
        
        # Operation-specific statistics
        operation_stats = {}
        for operation in set(m.operation_name for m in self.metrics):
            op_metrics = [m for m in self.metrics if m.operation_name == operation]
            operation_stats[operation] = {
                'count': len(op_metrics),
                'avg_time': sum(m.execution_time for m in op_metrics) / len(op_metrics),
                'max_time': max(m.execution_time for m in op_metrics),
                'avg_memory': sum(m.memory_usage for m in op_metrics) / len(op_metrics),
                'avg_throughput': sum(m.throughput for m in op_metrics) / len(op_metrics)
            }
        
        return {
            'summary': {
                'total_operations': total_operations,
                'avg_execution_time': avg_execution_time,
                'max_execution_time': max_execution_time,
                'min_execution_time': min_execution_time,
                'performance_level': self.performance_level.value
            },
            'operations': operation_stats,
            'bottlenecks': self.bottlenecks,
            'optimization_enabled': self.auto_optimization_enabled
        }


class AutoScaler:
    """Automatic scaling based on performance metrics"""
    
    def __init__(self, performance_optimizer: PerformanceOptimizer):
        self.logger = structlog.get_logger("AutoScaler")
        self.performance_optimizer = performance_optimizer
        self.scaling_enabled = True
        self.scaling_metrics = {
            'scale_up_threshold': 0.8,    # 80% resource utilization
            'scale_down_threshold': 0.3,  # 30% resource utilization
            'min_instances': 1,
            'max_instances': 10,
            'current_instances': 1
        }
        
        # Scaling history
        self.scaling_history: List[Dict] = []
        
    async def monitor_and_scale(self):
        """Monitor performance and trigger scaling decisions"""
        
        while self.scaling_enabled:
            try:
                # Collect current metrics
                current_metrics = await self._collect_scaling_metrics()
                
                # Make scaling decision
                scaling_decision = await self._make_scaling_decision(current_metrics)
                
                # Execute scaling if needed
                if scaling_decision['action'] != 'none':
                    await self._execute_scaling(scaling_decision)
                
                # Wait before next evaluation
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error("Auto-scaling error", error=str(e))
                await asyncio.sleep(60)
    
    async def _collect_scaling_metrics(self) -> Dict[str, Any]:
        """Collect metrics for scaling decisions"""
        
        # Get performance metrics
        perf_report = self.performance_optimizer.get_performance_report()
        
        # Get system resource metrics
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            return {
                'cpu_utilization': cpu_percent / 100.0,
                'memory_utilization': memory_percent / 100.0,
                'avg_response_time': perf_report.get('summary', {}).get('avg_execution_time', 0),
                'total_operations': perf_report.get('summary', {}).get('total_operations', 0),
                'bottlenecks': len(self.performance_optimizer.bottlenecks)
            }
        except:
            return {
                'cpu_utilization': 0.5,
                'memory_utilization': 0.5,
                'avg_response_time': 1.0,
                'total_operations': 0,
                'bottlenecks': 0
            }
    
    async def _make_scaling_decision(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Make scaling decision based on metrics"""
        
        current_instances = self.scaling_metrics['current_instances']
        max_utilization = max(metrics['cpu_utilization'], metrics['memory_utilization'])
        
        # Scale up conditions
        if (max_utilization > self.scaling_metrics['scale_up_threshold'] and
            current_instances < self.scaling_metrics['max_instances']):
            
            new_instances = min(
                current_instances * 2,  # Double instances
                self.scaling_metrics['max_instances']
            )
            
            return {
                'action': 'scale_up',
                'current_instances': current_instances,
                'target_instances': new_instances,
                'reason': f"High utilization: {max_utilization:.2%}",
                'metrics': metrics
            }
        
        # Scale down conditions
        elif (max_utilization < self.scaling_metrics['scale_down_threshold'] and
              current_instances > self.scaling_metrics['min_instances']):
            
            new_instances = max(
                current_instances // 2,  # Halve instances
                self.scaling_metrics['min_instances']
            )
            
            return {
                'action': 'scale_down',
                'current_instances': current_instances,
                'target_instances': new_instances,
                'reason': f"Low utilization: {max_utilization:.2%}",
                'metrics': metrics
            }
        
        # No scaling needed
        return {
            'action': 'none',
            'current_instances': current_instances,
            'target_instances': current_instances,
            'reason': f"Utilization within range: {max_utilization:.2%}",
            'metrics': metrics
        }
    
    async def _execute_scaling(self, decision: Dict[str, Any]):
        """Execute scaling decision"""
        
        self.logger.info("Executing scaling decision",
                        action=decision['action'],
                        current=decision['current_instances'],
                        target=decision['target_instances'],
                        reason=decision['reason'])
        
        # Update instance count
        self.scaling_metrics['current_instances'] = decision['target_instances']
        
        # Record scaling event
        scaling_event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': decision['action'],
            'from_instances': decision['current_instances'],
            'to_instances': decision['target_instances'],
            'reason': decision['reason'],
            'metrics': decision['metrics']
        }
        
        self.scaling_history.append(scaling_event)
        
        # In a real implementation, this would:
        # - Start/stop containers
        # - Update load balancer configuration
        # - Adjust resource allocation
        # - Notify monitoring systems
        
        # For now, just adjust the performance optimization level
        if decision['action'] == 'scale_up':
            # Increase performance level when scaling up
            if self.performance_optimizer.performance_level == PerformanceLevel.BASIC:
                self.performance_optimizer.performance_level = PerformanceLevel.OPTIMIZED
            elif self.performance_optimizer.performance_level == PerformanceLevel.OPTIMIZED:
                self.performance_optimizer.performance_level = PerformanceLevel.HIGH_PERFORMANCE
        elif decision['action'] == 'scale_down':
            # Decrease performance level when scaling down
            if self.performance_optimizer.performance_level == PerformanceLevel.HIGH_PERFORMANCE:
                self.performance_optimizer.performance_level = PerformanceLevel.OPTIMIZED
            elif self.performance_optimizer.performance_level == PerformanceLevel.OPTIMIZED:
                self.performance_optimizer.performance_level = PerformanceLevel.BASIC


# Performance decorators
def high_performance(cache_strategy: CacheStrategy = CacheStrategy.LRU, 
                    optimization_level: PerformanceLevel = PerformanceLevel.HIGH_PERFORMANCE):
    """Decorator for high-performance optimization"""
    
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Initialize performance optimizer
            optimizer = PerformanceOptimizer()
            optimizer.performance_level = optimization_level
            
            # Optimize function
            optimized_func = await optimizer.optimize_performance(
                func.__name__, func, optimization_level
            )
            
            # Execute optimized function
            return await optimized_func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use basic optimization
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage and testing
async def main():
    """Example usage of the scalable performance engine"""
    
    print("⚡ Initializing Scalable Performance Engine")
    
    # Initialize components
    performance_optimizer = PerformanceOptimizer()
    auto_scaler = AutoScaler(performance_optimizer)
    
    # Configure cache
    cache_config = CacheConfig(
        strategy=CacheStrategy.LRU,
        max_size=1000,
        ttl_seconds=3600
    )
    cache = IntelligentCache(cache_config)
    
    # Test caching
    print("\n🔄 Testing Intelligent Caching")
    await cache.set("test_key", {"data": "test_value", "timestamp": time.time()})
    cached_value = await cache.get("test_key")
    print(f"Cached value: {cached_value}")
    
    cache_stats = cache.get_cache_stats()
    print(f"Cache stats: {cache_stats}")
    
    # Test performance optimization
    print("\n🚀 Testing Performance Optimization")
    
    @high_performance(optimization_level=PerformanceLevel.HIGH_PERFORMANCE)
    async def cpu_intensive_task(n: int) -> int:
        """Simulate CPU-intensive computation"""
        total = 0
        for i in range(n):
            total += i * i
        return total
    
    # Run test operations
    start_time = time.time()
    result = await cpu_intensive_task(100000)
    end_time = time.time()
    
    print(f"CPU-intensive task result: {result}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    
    # Generate performance report
    perf_report = performance_optimizer.get_performance_report()
    print(f"\n📊 Performance Report:")
    print(json.dumps(perf_report, indent=2, default=str))
    
    # Test auto-scaling (in background)
    print("\n📈 Starting Auto-Scaling Monitor")
    scaling_task = asyncio.create_task(auto_scaler.monitor_and_scale())
    
    # Let it run for a short time
    await asyncio.sleep(5)
    
    # Stop auto-scaling
    auto_scaler.scaling_enabled = False
    scaling_task.cancel()
    
    try:
        await scaling_task
    except asyncio.CancelledError:
        pass
    
    print("\n✅ Scalable Performance Engine demonstration complete")


if __name__ == "__main__":
    # Use uvloop for better performance if available
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    
    asyncio.run(main())