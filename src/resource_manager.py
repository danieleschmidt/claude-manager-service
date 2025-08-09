"""
Intelligent Resource Management System for Claude Manager Service Generation 3

This module provides enterprise-grade resource management including:
- Intelligent resource pooling and reuse
- Memory optimization with garbage collection tuning
- CPU usage optimization and thermal management
- Disk I/O optimization with buffering and caching
- Resource monitoring and alerting
- Adaptive resource allocation based on workload
- Resource cleanup and lifecycle management
- Performance profiling and optimization recommendations
"""

import asyncio
import gc
import mmap
import os
import psutil
import resource
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, ContextManager
import logging
import tempfile

from .logger import get_logger
from .performance_monitor import monitor_performance, get_monitor
from .error_handler import ResourceError, with_enhanced_error_handling


logger = get_logger(__name__)


class ResourceType(Enum):
    """Types of managed resources"""
    MEMORY = "memory"
    CPU = "cpu"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    FILE_HANDLES = "file_handles"
    THREADS = "threads"
    PROCESSES = "processes"


class ResourceStatus(Enum):
    """Resource allocation status"""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    BUSY = "busy"
    EXHAUSTED = "exhausted"
    ERROR = "error"


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    resource_type: ResourceType
    allocated: float
    available: float
    utilization: float
    peak_usage: float
    average_usage: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceLimits:
    """Resource limits configuration"""
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_file_handles: Optional[int] = None
    max_threads: Optional[int] = None
    max_disk_usage_mb: Optional[int] = None
    soft_limits: bool = True  # Whether to enforce soft limits


class MemoryPool:
    """Memory pool for efficient allocation and reuse"""
    
    def __init__(self, block_size: int = 4096, max_blocks: int = 1000):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.available_blocks: deque = deque()
        self.allocated_blocks: Set[memoryview] = set()
        self.total_allocated = 0
        self.peak_allocated = 0
        self.lock = threading.Lock()
        
        logger.info(f"MemoryPool initialized: block_size={block_size}, max_blocks={max_blocks}")
    
    def allocate(self, size: int) -> Optional[memoryview]:
        """Allocate memory block"""
        with self.lock:
            if size > self.block_size:
                # Large allocation, create new block
                try:
                    data = bytearray(size)
                    block = memoryview(data)
                    self.allocated_blocks.add(block)
                    self.total_allocated += size
                    self.peak_allocated = max(self.peak_allocated, self.total_allocated)
                    return block
                except MemoryError:
                    logger.error(f"Failed to allocate {size} bytes")
                    return None
            
            # Try to reuse existing block
            if self.available_blocks:
                block = self.available_blocks.popleft()
                self.allocated_blocks.add(block)
                return block
            
            # Create new block if under limit
            if len(self.allocated_blocks) < self.max_blocks:
                try:
                    data = bytearray(self.block_size)
                    block = memoryview(data)
                    self.allocated_blocks.add(block)
                    self.total_allocated += self.block_size
                    self.peak_allocated = max(self.peak_allocated, self.total_allocated)
                    return block
                except MemoryError:
                    logger.error(f"Failed to allocate memory block of {self.block_size} bytes")
                    return None
            
            return None  # Pool exhausted
    
    def deallocate(self, block: memoryview):
        """Deallocate memory block"""
        with self.lock:
            if block in self.allocated_blocks:
                self.allocated_blocks.remove(block)
                
                if len(block) == self.block_size:
                    # Standard block, return to pool
                    self.available_blocks.append(block)
                else:
                    # Large block, let it be garbage collected
                    self.total_allocated -= len(block)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        with self.lock:
            return {
                'block_size': self.block_size,
                'max_blocks': self.max_blocks,
                'available_blocks': len(self.available_blocks),
                'allocated_blocks': len(self.allocated_blocks),
                'total_allocated': self.total_allocated,
                'peak_allocated': self.peak_allocated,
                'utilization': len(self.allocated_blocks) / self.max_blocks
            }


class DiskIOManager:
    """Disk I/O optimization manager"""
    
    def __init__(self, cache_size_mb: int = 64, buffer_size: int = 65536):
        self.cache_size_mb = cache_size_mb
        self.buffer_size = buffer_size
        self.file_cache: Dict[str, bytes] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.max_cache_size = cache_size_mb * 1024 * 1024
        self.current_cache_size = 0
        self.lock = threading.RLock()
        
        # I/O statistics
        self.read_count = 0
        self.write_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.bytes_read = 0
        self.bytes_written = 0
        
        logger.info(f"DiskIOManager initialized: cache={cache_size_mb}MB, buffer={buffer_size}")
    
    @contextmanager
    def buffered_read(self, file_path: str):
        """Context manager for buffered file reading"""
        try:
            with open(file_path, 'rb', buffering=self.buffer_size) as f:
                yield f
                self.read_count += 1
        except IOError as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    @contextmanager
    def buffered_write(self, file_path: str):
        """Context manager for buffered file writing"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb', buffering=self.buffer_size) as f:
                yield f
                self.write_count += 1
        except IOError as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise
    
    def read_cached(self, file_path: str) -> Optional[bytes]:
        """Read file with caching"""
        with self.lock:
            # Check cache first
            if file_path in self.file_cache:
                self.cache_access_times[file_path] = time.time()
                self.cache_hits += 1
                return self.file_cache[file_path]
            
            # Read from disk
            try:
                with self.buffered_read(file_path) as f:
                    data = f.read()
                    self.bytes_read += len(data)
                    
                    # Cache if it fits
                    if len(data) + self.current_cache_size <= self.max_cache_size:
                        self._add_to_cache(file_path, data)
                    elif len(data) < self.max_cache_size // 2:
                        # Make room in cache
                        self._evict_lru_cache(len(data))
                        self._add_to_cache(file_path, data)
                    
                    self.cache_misses += 1
                    return data
            except IOError:
                return None
    
    def write_with_sync(self, file_path: str, data: bytes, sync: bool = True):
        """Write file with optional sync"""
        try:
            with self.buffered_write(file_path) as f:
                f.write(data)
                if sync:
                    f.flush()
                    os.fsync(f.fileno())
                self.bytes_written += len(data)
        except IOError as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise
    
    def _add_to_cache(self, file_path: str, data: bytes):
        """Add file to cache"""
        self.file_cache[file_path] = data
        self.cache_access_times[file_path] = time.time()
        self.current_cache_size += len(data)
    
    def _evict_lru_cache(self, required_space: int):
        """Evict least recently used items from cache"""
        # Sort by access time
        items = sorted(self.cache_access_times.items(), key=lambda x: x[1])
        
        for file_path, _ in items:
            if self.current_cache_size + required_space <= self.max_cache_size:
                break
            
            data = self.file_cache.pop(file_path, None)
            if data:
                self.current_cache_size -= len(data)
            self.cache_access_times.pop(file_path, None)
    
    def clear_cache(self):
        """Clear file cache"""
        with self.lock:
            self.file_cache.clear()
            self.cache_access_times.clear()
            self.current_cache_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk I/O statistics"""
        with self.lock:
            cache_hit_ratio = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            
            return {
                'cache_size_mb': self.cache_size_mb,
                'buffer_size': self.buffer_size,
                'current_cache_size': self.current_cache_size,
                'cache_utilization': self.current_cache_size / self.max_cache_size,
                'cached_files': len(self.file_cache),
                'read_count': self.read_count,
                'write_count': self.write_count,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_ratio': cache_hit_ratio,
                'bytes_read': self.bytes_read,
                'bytes_written': self.bytes_written
            }


class CPUManager:
    """CPU usage optimization manager"""
    
    def __init__(self):
        self.cpu_affinity_enabled = hasattr(psutil.Process(), 'cpu_affinity')
        self.process = psutil.Process()
        self.cpu_count = psutil.cpu_count()
        self.cpu_usage_history: deque = deque(maxlen=100)
        self.thermal_throttling_threshold = 85.0  # Temperature in Celsius
        
        logger.info(f"CPUManager initialized: cores={self.cpu_count}, affinity={self.cpu_affinity_enabled}")
    
    def set_cpu_affinity(self, cpu_list: List[int]) -> bool:
        """Set CPU affinity for the current process"""
        if not self.cpu_affinity_enabled:
            return False
        
        try:
            valid_cpus = [cpu for cpu in cpu_list if 0 <= cpu < self.cpu_count]
            if valid_cpus:
                self.process.cpu_affinity(valid_cpus)
                logger.info(f"Set CPU affinity to cores: {valid_cpus}")
                return True
        except Exception as e:
            logger.error(f"Failed to set CPU affinity: {e}")
        
        return False
    
    def set_process_priority(self, priority: int) -> bool:
        """Set process priority (nice value on Unix)"""
        try:
            if hasattr(os, 'nice'):
                os.nice(priority)
                logger.info(f"Set process priority to {priority}")
                return True
        except Exception as e:
            logger.error(f"Failed to set process priority: {e}")
        
        return False
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize CPU settings for specific workload type"""
        if workload_type == "cpu_intensive":
            # Use all available cores
            if self.cpu_affinity_enabled:
                self.set_cpu_affinity(list(range(self.cpu_count)))
            self.set_process_priority(-10)  # Higher priority
            
        elif workload_type == "io_intensive":
            # Use subset of cores to leave room for other processes
            if self.cpu_affinity_enabled and self.cpu_count > 2:
                cores_to_use = max(1, self.cpu_count // 2)
                self.set_cpu_affinity(list(range(cores_to_use)))
            self.set_process_priority(0)  # Normal priority
            
        elif workload_type == "balanced":
            # Use most cores but leave one for system
            if self.cpu_affinity_enabled and self.cpu_count > 1:
                self.set_cpu_affinity(list(range(max(1, self.cpu_count - 1))))
            self.set_process_priority(0)
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        usage = self.process.cpu_percent()
        self.cpu_usage_history.append((time.time(), usage))
        return usage
    
    def get_thermal_status(self) -> Optional[Dict[str, float]]:
        """Get thermal status if available"""
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Return first available temperature sensor
                    for sensor_name, sensors in temps.items():
                        if sensors:
                            return {
                                'sensor': sensor_name,
                                'current': sensors[0].current,
                                'high': sensors[0].high,
                                'critical': sensors[0].critical
                            }
        except Exception as e:
            logger.debug(f"Could not get thermal status: {e}")
        
        return None
    
    def is_thermal_throttling(self) -> bool:
        """Check if CPU is thermal throttling"""
        thermal = self.get_thermal_status()
        if thermal:
            return thermal['current'] > self.thermal_throttling_threshold
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CPU management statistics"""
        current_usage = self.get_cpu_usage()
        thermal = self.get_thermal_status()
        
        affinity = None
        if self.cpu_affinity_enabled:
            try:
                affinity = self.process.cpu_affinity()
            except Exception:
                pass
        
        return {
            'cpu_count': self.cpu_count,
            'current_usage': current_usage,
            'cpu_affinity': affinity,
            'thermal_status': thermal,
            'is_thermal_throttling': self.is_thermal_throttling(),
            'usage_history_count': len(self.cpu_usage_history)
        }


class GCOptimizer:
    """Garbage collection optimization"""
    
    def __init__(self):
        self.gc_stats = {
            'collections': defaultdict(int),
            'total_time': 0.0,
            'objects_collected': 0
        }
        self.original_thresholds = gc.get_threshold()
        self.tuning_enabled = False
        
        logger.info("GCOptimizer initialized")
    
    def enable_gc_optimization(self):
        """Enable garbage collection optimization"""
        if not self.tuning_enabled:
            # Set more aggressive thresholds for better performance
            gc.set_threshold(1000, 15, 15)  # Increased gen0 threshold
            self.tuning_enabled = True
            logger.info("GC optimization enabled")
    
    def disable_gc_optimization(self):
        """Disable garbage collection optimization"""
        if self.tuning_enabled:
            gc.set_threshold(*self.original_thresholds)
            self.tuning_enabled = False
            logger.info("GC optimization disabled")
    
    def force_collection(self, generation: Optional[int] = None) -> int:
        """Force garbage collection"""
        start_time = time.time()
        
        if generation is not None:
            collected = gc.collect(generation)
        else:
            collected = gc.collect()
        
        collection_time = time.time() - start_time
        self.gc_stats['total_time'] += collection_time
        self.gc_stats['objects_collected'] += collected
        
        logger.debug(f"GC collected {collected} objects in {collection_time:.4f}s")
        return collected
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information"""
        # Get GC stats
        gc_counts = gc.get_count()
        gc_stats_list = gc.get_stats()
        
        # Get process memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'gc_counts': gc_counts,
            'gc_stats': gc_stats_list,
            'gc_thresholds': gc.get_threshold(),
            'gc_objects': len(gc.get_objects()),
            'tuning_enabled': self.tuning_enabled
        }
    
    @contextmanager
    def gc_disabled(self):
        """Context manager to temporarily disable GC"""
        was_enabled = gc.isenabled()
        try:
            gc.disable()
            yield
        finally:
            if was_enabled:
                gc.enable()


class ResourceManager:
    """Central resource management system"""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        
        # Component managers
        self.memory_pool = MemoryPool()
        self.disk_io_manager = DiskIOManager()
        self.cpu_manager = CPUManager()
        self.gc_optimizer = GCOptimizer()
        
        # Resource tracking
        self.resource_metrics: Dict[ResourceType, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.resource_status: Dict[ResourceType, ResourceStatus] = {}
        self.resource_alerts: List[Dict[str, Any]] = []
        
        # Monitoring
        self.monitoring_enabled = True
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_thresholds = {
            ResourceType.MEMORY: 85.0,
            ResourceType.CPU: 90.0,
            ResourceType.DISK_IO: 80.0
        }
        
        logger.info("ResourceManager initialized")
    
    async def start(self):
        """Start resource monitoring"""
        if self.monitoring_enabled and not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.gc_optimizer.enable_gc_optimization()
            logger.info("Resource monitoring started")
    
    async def stop(self):
        """Stop resource monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        self.gc_optimizer.disable_gc_optimization()
        logger.info("Resource monitoring stopped")
    
    @asynccontextmanager
    async def allocate_memory(self, size: int):
        """Context manager for memory allocation"""
        block = self.memory_pool.allocate(size)
        if block is None:
            raise ResourceError(f"Failed to allocate {size} bytes", "memory_allocation")
        
        try:
            yield block
        finally:
            self.memory_pool.deallocate(block)
    
    @asynccontextmanager
    async def managed_file_operation(self, file_path: str, mode: str = 'r'):
        """Context manager for managed file operations"""
        if 'r' in mode:
            try:
                with self.disk_io_manager.buffered_read(file_path) as f:
                    yield f
            except IOError as e:
                raise ResourceError(f"File read error: {e}", "file_operation")
        else:
            try:
                with self.disk_io_manager.buffered_write(file_path) as f:
                    yield f
            except IOError as e:
                raise ResourceError(f"File write error: {e}", "file_operation")
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize resources for specific workload"""
        self.cpu_manager.optimize_for_workload(workload_type)
        
        if workload_type == "memory_intensive":
            # Increase memory pool size
            self.memory_pool.max_blocks = min(2000, self.memory_pool.max_blocks * 2)
            # More aggressive GC
            self.gc_optimizer.enable_gc_optimization()
            
        elif workload_type == "io_intensive":
            # Increase disk cache
            self.disk_io_manager.cache_size_mb = min(128, self.disk_io_manager.cache_size_mb * 2)
            self.disk_io_manager.max_cache_size = self.disk_io_manager.cache_size_mb * 1024 * 1024
        
        logger.info(f"Optimized resources for {workload_type} workload")
    
    async def _monitoring_loop(self):
        """Background resource monitoring loop"""
        logger.debug("Started resource monitoring loop")
        
        while True:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_metrics(self):
        """Collect resource metrics"""
        timestamp = datetime.now()
        
        # Memory metrics
        memory_info = self.gc_optimizer.get_memory_info()
        memory_percent = psutil.virtual_memory().percent
        
        memory_metrics = ResourceMetrics(
            resource_type=ResourceType.MEMORY,
            allocated=memory_info['rss_mb'],
            available=psutil.virtual_memory().available / 1024 / 1024,
            utilization=memory_percent,
            peak_usage=self.memory_pool.peak_allocated,
            average_usage=memory_percent,
            timestamp=timestamp
        )
        self.resource_metrics[ResourceType.MEMORY].append(memory_metrics)
        
        # CPU metrics
        cpu_usage = self.cpu_manager.get_cpu_usage()
        cpu_metrics = ResourceMetrics(
            resource_type=ResourceType.CPU,
            allocated=cpu_usage,
            available=100.0 - cpu_usage,
            utilization=cpu_usage,
            peak_usage=cpu_usage,
            average_usage=cpu_usage,
            timestamp=timestamp
        )
        self.resource_metrics[ResourceType.CPU].append(cpu_metrics)
        
        # Disk I/O metrics
        disk_stats = self.disk_io_manager.get_stats()
        disk_metrics = ResourceMetrics(
            resource_type=ResourceType.DISK_IO,
            allocated=disk_stats['current_cache_size'] / 1024 / 1024,
            available=disk_stats['cache_size_mb'] - (disk_stats['current_cache_size'] / 1024 / 1024),
            utilization=disk_stats['cache_utilization'] * 100,
            peak_usage=disk_stats['cache_utilization'] * 100,
            average_usage=disk_stats['cache_utilization'] * 100,
            timestamp=timestamp
        )
        self.resource_metrics[ResourceType.DISK_IO].append(disk_metrics)
    
    async def _check_alerts(self):
        """Check for resource alerts"""
        for resource_type, threshold in self.alert_thresholds.items():
            if resource_type in self.resource_metrics:
                recent_metrics = list(self.resource_metrics[resource_type])[-5:]  # Last 5 readings
                if recent_metrics:
                    avg_utilization = sum(m.utilization for m in recent_metrics) / len(recent_metrics)
                    
                    if avg_utilization > threshold:
                        alert = {
                            'timestamp': datetime.now().isoformat(),
                            'resource_type': resource_type.value,
                            'utilization': avg_utilization,
                            'threshold': threshold,
                            'message': f"{resource_type.value} utilization ({avg_utilization:.1f}%) exceeds threshold ({threshold}%)"
                        }
                        
                        self.resource_alerts.append(alert)
                        logger.warning(alert['message'])
                        
                        # Trigger optimization actions
                        await self._handle_resource_alert(resource_type, avg_utilization)
    
    async def _handle_resource_alert(self, resource_type: ResourceType, utilization: float):
        """Handle resource alerts with automatic optimization"""
        if resource_type == ResourceType.MEMORY and utilization > 90:
            # Force garbage collection
            collected = self.gc_optimizer.force_collection()
            logger.info(f"Emergency GC collected {collected} objects")
            
            # Clear disk cache to free memory
            self.disk_io_manager.clear_cache()
            logger.info("Cleared disk cache to free memory")
            
        elif resource_type == ResourceType.CPU and utilization > 95:
            # Check for thermal throttling
            if self.cpu_manager.is_thermal_throttling():
                logger.warning("CPU thermal throttling detected - reducing workload")
                # This could trigger scaling down or load shedding
        
        elif resource_type == ResourceType.DISK_IO and utilization > 85:
            # Reduce cache size
            self.disk_io_manager.cache_size_mb = max(16, self.disk_io_manager.cache_size_mb // 2)
            self.disk_io_manager.max_cache_size = self.disk_io_manager.cache_size_mb * 1024 * 1024
            logger.info(f"Reduced disk cache size to {self.disk_io_manager.cache_size_mb}MB")
    
    @monitor_performance(track_memory=True, custom_name="resource_cleanup")
    async def cleanup_resources(self):
        """Clean up resources to free memory and improve performance"""
        start_time = time.time()
        
        # Force garbage collection
        collected_objects = self.gc_optimizer.force_collection()
        
        # Clear disk cache
        cache_cleared = len(self.disk_io_manager.file_cache)
        self.disk_io_manager.clear_cache()
        
        # Clear old metrics
        for metrics_queue in self.resource_metrics.values():
            if len(metrics_queue) > 500:
                # Keep only recent half
                keep_count = len(metrics_queue) // 2
                for _ in range(len(metrics_queue) - keep_count):
                    metrics_queue.popleft()
        
        # Clear old alerts
        if len(self.resource_alerts) > 100:
            self.resource_alerts = self.resource_alerts[-50:]
        
        cleanup_time = time.time() - start_time
        logger.info(f"Resource cleanup completed in {cleanup_time:.2f}s: "
                   f"{collected_objects} objects collected, {cache_cleared} cache entries cleared")
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'memory_pool': self.memory_pool.get_stats(),
            'disk_io': self.disk_io_manager.get_stats(),
            'cpu': self.cpu_manager.get_stats(),
            'gc_info': self.gc_optimizer.get_memory_info(),
            'resource_limits': {
                'max_memory_mb': self.limits.max_memory_mb,
                'max_cpu_percent': self.limits.max_cpu_percent,
                'max_file_handles': self.limits.max_file_handles,
                'soft_limits': self.limits.soft_limits
            },
            'recent_alerts': self.resource_alerts[-10:],  # Last 10 alerts
            'monitoring_enabled': self.monitoring_enabled
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get resource optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        memory_stats = self.memory_pool.get_stats()
        if memory_stats['utilization'] > 0.8:
            recommendations.append("Consider increasing memory pool size or optimizing memory usage")
        
        # CPU recommendations
        cpu_stats = self.cpu_manager.get_stats()
        if cpu_stats.get('is_thermal_throttling', False):
            recommendations.append("CPU thermal throttling detected - improve cooling or reduce load")
        
        # Disk I/O recommendations
        disk_stats = self.disk_io_manager.get_stats()
        if disk_stats['cache_hit_ratio'] < 0.5:
            recommendations.append("Low disk cache hit ratio - consider increasing cache size")
        
        # GC recommendations
        gc_info = self.gc_optimizer.get_memory_info()
        if gc_info['gc_objects'] > 100000:
            recommendations.append("High number of GC objects - consider optimizing object lifecycle")
        
        return recommendations


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None
_manager_lock = asyncio.Lock()


async def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance"""
    global _resource_manager
    
    if _resource_manager is None:
        async with _manager_lock:
            if _resource_manager is None:
                _resource_manager = ResourceManager()
                await _resource_manager.start()
    
    return _resource_manager


# Convenience functions
@with_enhanced_error_handling("resource_optimization")
async def optimize_for_workload(workload_type: str):
    """Optimize resources for specific workload type"""
    manager = await get_resource_manager()
    manager.optimize_for_workload(workload_type)


async def cleanup_system_resources():
    """Clean up system resources"""
    manager = await get_resource_manager()
    await manager.cleanup_resources()


async def get_resource_stats() -> Dict[str, Any]:
    """Get comprehensive resource statistics"""
    manager = await get_resource_manager()
    return await manager.get_comprehensive_stats()


@asynccontextmanager
async def managed_memory(size: int):
    """Context manager for managed memory allocation"""
    manager = await get_resource_manager()
    async with manager.allocate_memory(size) as block:
        yield block


@asynccontextmanager
async def managed_file(file_path: str, mode: str = 'r'):
    """Context manager for managed file operations"""
    manager = await get_resource_manager()
    async with manager.managed_file_operation(file_path, mode) as f:
        yield f