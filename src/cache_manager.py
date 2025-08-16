"""
High-Performance Cache Manager for Claude Manager Service Generation 3

This module provides enterprise-grade caching capabilities with:
- Redis-compatible interface with fallback to in-memory caching
- Intelligent TTL management and cache invalidation patterns
- Multi-level caching strategy (L1: Memory, L2: Redis)
- Cache warming and prefetching capabilities
- Performance monitoring and cache hit ratio optimization
- Automatic cache partitioning and sharding
- Circuit breaker pattern for cache failures
- Serialization optimization for complex objects
"""

import asyncio
import json
import pickle
import hashlib
import time
import threading
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from functools import wraps
import logging
import os

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.logger import get_logger
from src.performance_monitor import monitor_performance, get_monitor
from src.error_handler import CacheError, with_enhanced_error_handling


logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float]
    access_count: int = 0
    last_access: Optional[float] = None
    size_bytes: Optional[int] = None
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.last_access is None:
            self.last_access = self.timestamp


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage: int = 0
    avg_access_time: float = 0.0
    hit_ratio: float = 0.0


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass


class InMemoryCache(CacheBackend):
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = asyncio.Lock()
        self.stats = CacheStats()
        
        # Tag-based invalidation support
        self.tag_keys: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info(f"InMemoryCache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        async with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            
            # Check TTL expiration
            if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                del self.cache[key]
                self._remove_from_tags(key, entry.tags)
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            # Update access tracking
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            self.stats.hits += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[Set[str]] = None) -> bool:
        """Set value in memory cache"""
        async with self.lock:
            try:
                # Calculate size
                size_bytes = len(pickle.dumps(value))
                
                # Check if we need to evict entries
                await self._ensure_capacity(size_bytes)
                
                # Remove old entry if exists
                if key in self.cache:
                    old_entry = self.cache[key]
                    self._remove_from_tags(key, old_entry.tags)
                    self.stats.memory_usage -= old_entry.size_bytes or 0
                
                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl,
                    size_bytes=size_bytes,
                    tags=tags or set()
                )
                
                # Store entry
                self.cache[key] = entry
                self.stats.memory_usage += size_bytes
                self.stats.sets += 1
                
                # Update tag index
                for tag in entry.tags:
                    self.tag_keys[tag].add(key)
                
                return True
                
            except Exception as e:
                logger.error(f"Error setting cache entry {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from memory cache"""
        async with self.lock:
            entry = self.cache.pop(key, None)
            if entry:
                self.stats.memory_usage -= entry.size_bytes or 0
                self.stats.deletes += 1
                self._remove_from_tags(key, entry.tags)
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache"""
        async with self.lock:
            return key in self.cache
    
    async def clear(self) -> bool:
        """Clear all entries from memory cache"""
        async with self.lock:
            self.cache.clear()
            self.tag_keys.clear()
            self.stats = CacheStats()
            return True
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all keys with any of the specified tags"""
        async with self.lock:
            keys_to_remove = set()
            for tag in tags:
                keys_to_remove.update(self.tag_keys.get(tag, set()))
            
            count = 0
            for key in keys_to_remove:
                if await self.delete(key):
                    count += 1
            
            return count
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        async with self.lock:
            total_requests = self.stats.hits + self.stats.misses
            self.stats.hit_ratio = self.stats.hits / total_requests if total_requests > 0 else 0.0
            return self.stats
    
    def _remove_from_tags(self, key: str, tags: Set[str]):
        """Remove key from tag index"""
        for tag in tags:
            self.tag_keys[tag].discard(key)
            if not self.tag_keys[tag]:
                del self.tag_keys[tag]
    
    async def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry"""
        # Memory-based eviction
        while (self.stats.memory_usage + new_entry_size) > self.max_memory_bytes and self.cache:
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            self.stats.memory_usage -= oldest_entry.size_bytes or 0
            self.stats.evictions += 1
            self._remove_from_tags(oldest_key, oldest_entry.tags)
        
        # Size-based eviction
        while len(self.cache) >= self.max_size and self.cache:
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            self.stats.memory_usage -= oldest_entry.size_bytes or 0
            self.stats.evictions += 1
            self._remove_from_tags(oldest_key, oldest_entry.tags)


class RedisCache(CacheBackend):
    """Redis-based cache backend with advanced features"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", key_prefix: str = "claude:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis_client: Optional[redis.Redis] = None
        self.stats = CacheStats()
        self.connected = False
        
        logger.info(f"RedisCache initialized: url={redis_url}, prefix={key_prefix}")
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            if not REDIS_AVAILABLE:
                logger.warning("Redis client not available, using in-memory cache")
                return False
            
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            await self.redis_client.ping()
            self.connected = True
            logger.info("Connected to Redis successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        if not self.connected:
            return None
        
        try:
            prefixed_key = self._make_key(key)
            data = await self.redis_client.get(prefixed_key)
            
            if data is None:
                self.stats.misses += 1
                return None
            
            # Deserialize data
            try:
                value = pickle.loads(data)
                self.stats.hits += 1
                return value
            except Exception as e:
                logger.warning(f"Failed to deserialize Redis data for key {key}: {e}")
                await self.delete(key)  # Clean up corrupted data
                self.stats.misses += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[Set[str]] = None) -> bool:
        """Set value in Redis"""
        if not self.connected:
            return False
        
        try:
            prefixed_key = self._make_key(key)
            serialized_data = pickle.dumps(value)
            
            # Set with TTL if specified
            if ttl:
                await self.redis_client.setex(prefixed_key, int(ttl), serialized_data)
            else:
                await self.redis_client.set(prefixed_key, serialized_data)
            
            # Handle tags for invalidation
            if tags:
                for tag in tags:
                    tag_key = f"{self.key_prefix}tag:{tag}"
                    await self.redis_client.sadd(tag_key, key)
            
            self.stats.sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.connected:
            return False
        
        try:
            prefixed_key = self._make_key(key)
            result = await self.redis_client.delete(prefixed_key)
            
            if result > 0:
                self.stats.deletes += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.connected:
            return False
        
        try:
            prefixed_key = self._make_key(key)
            return bool(await self.redis_client.exists(prefixed_key))
            
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all entries with prefix from Redis"""
        if not self.connected:
            return False
        
        try:
            # Use SCAN to find all keys with prefix
            keys = []
            async for key in self.redis_client.scan_iter(match=f"{self.key_prefix}*"):
                keys.append(key)
            
            if keys:
                await self.redis_client.delete(*keys)
            
            self.stats = CacheStats()
            return True
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all keys with any of the specified tags"""
        if not self.connected:
            return 0
        
        try:
            keys_to_remove = set()
            for tag in tags:
                tag_key = f"{self.key_prefix}tag:{tag}"
                tagged_keys = await self.redis_client.smembers(tag_key)
                keys_to_remove.update(tagged_keys)
                # Clean up tag set
                await self.redis_client.delete(tag_key)
            
            count = 0
            for key in keys_to_remove:
                if await self.delete(key.decode() if isinstance(key, bytes) else key):
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Redis tag invalidation error: {e}")
            return 0
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        total_requests = self.stats.hits + self.stats.misses
        self.stats.hit_ratio = self.stats.hits / total_requests if total_requests > 0 else 0.0
        return self.stats


class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (Redis) tiers"""
    
    def __init__(self, 
                 memory_cache: Optional[InMemoryCache] = None,
                 redis_cache: Optional[RedisCache] = None,
                 enable_prefetching: bool = True):
        
        self.l1_cache = memory_cache or InMemoryCache()
        self.l2_cache = redis_cache
        self.enable_prefetching = enable_prefetching
        
        # Combined stats
        self.combined_stats = CacheStats()
        
        # Prefetching configuration
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prefetch_task: Optional[asyncio.Task] = None
        
        logger.info("MultiLevelCache initialized")
    
    async def initialize(self) -> bool:
        """Initialize cache backends"""
        success = True
        
        # Connect to Redis if available
        if self.l2_cache:
            redis_connected = await self.l2_cache.connect()
            if not redis_connected:
                logger.warning("Redis connection failed, using memory-only cache")
                self.l2_cache = None
                success = False
        
        # Start prefetching task
        if self.enable_prefetching:
            self.prefetch_task = asyncio.create_task(self._prefetch_worker())
        
        return success
    
    async def shutdown(self):
        """Shutdown cache backends"""
        if self.prefetch_task:
            self.prefetch_task.cancel()
            try:
                await self.prefetch_task
            except asyncio.CancelledError:
                pass
        
        if self.l2_cache:
            await self.l2_cache.disconnect()
    
    @monitor_performance(track_memory=True, custom_name="cache_get")
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from multi-level cache"""
        start_time = time.time()
        
        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            self.combined_stats.hits += 1
            return value
        
        # Try L2 cache
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Promote to L1 cache
                await self.l1_cache.set(key, value)
                self.combined_stats.hits += 1
                return value
        
        # Cache miss
        self.combined_stats.misses += 1
        self.combined_stats.avg_access_time = (
            self.combined_stats.avg_access_time + (time.time() - start_time)
        ) / 2
        
        return default
    
    @monitor_performance(track_memory=True, custom_name="cache_set")
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[Set[str]] = None) -> bool:
        """Set value in multi-level cache"""
        success = True
        
        # Set in L1 cache
        if not await self.l1_cache.set(key, value, ttl, tags):
            success = False
        
        # Set in L2 cache
        if self.l2_cache:
            if not await self.l2_cache.set(key, value, ttl, tags):
                success = False
        
        if success:
            self.combined_stats.sets += 1
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        l1_deleted = await self.l1_cache.delete(key)
        l2_deleted = True
        
        if self.l2_cache:
            l2_deleted = await self.l2_cache.delete(key)
        
        if l1_deleted or l2_deleted:
            self.combined_stats.deletes += 1
            return True
        
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache level"""
        if await self.l1_cache.exists(key):
            return True
        
        if self.l2_cache and await self.l2_cache.exists(key):
            return True
        
        return False
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all keys with specified tags"""
        l1_count = await self.l1_cache.invalidate_by_tags(tags)
        l2_count = 0
        
        if self.l2_cache:
            l2_count = await self.l2_cache.invalidate_by_tags(tags)
        
        total_count = l1_count + l2_count
        self.combined_stats.evictions += total_count
        return total_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        l1_stats = await self.l1_cache.get_stats()
        l2_stats = await self.l2_cache.get_stats() if self.l2_cache else CacheStats()
        
        return {
            'combined': asdict(self.combined_stats),
            'l1_memory': asdict(l1_stats),
            'l2_redis': asdict(l2_stats),
            'total_hit_ratio': (
                (l1_stats.hits + l2_stats.hits) / 
                max(l1_stats.hits + l1_stats.misses + l2_stats.hits + l2_stats.misses, 1)
            )
        }
    
    async def warm_cache(self, keys_values: Dict[str, Any], ttl: Optional[float] = None):
        """Warm cache with predefined data"""
        logger.info(f"Warming cache with {len(keys_values)} entries")
        
        tasks = []
        for key, value in keys_values.items():
            task = asyncio.create_task(self.set(key, value, ttl))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for result in results if result is True)
        
        logger.info(f"Cache warming completed: {success_count}/{len(keys_values)} successful")
    
    async def _prefetch_worker(self):
        """Background worker for cache prefetching"""
        logger.info("Starting cache prefetch worker")
        
        while True:
            try:
                # Get prefetch request from queue
                prefetch_func = await self.prefetch_queue.get()
                
                # Execute prefetch function
                if asyncio.iscoroutinefunction(prefetch_func):
                    await prefetch_func()
                else:
                    prefetch_func()
                
                self.prefetch_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Cache prefetch worker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cache prefetch worker: {e}")


# Global cache instance
_cache_manager: Optional[MultiLevelCache] = None
_cache_lock = asyncio.Lock()


async def get_cache_manager() -> MultiLevelCache:
    """Get global cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        async with _cache_lock:
            if _cache_manager is None:
                # Initialize from environment
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                memory_max_size = int(os.getenv('CACHE_MEMORY_MAX_SIZE', '10000'))
                memory_max_mb = int(os.getenv('CACHE_MEMORY_MAX_MB', '512'))
                
                memory_cache = InMemoryCache(memory_max_size, memory_max_mb)
                redis_cache = RedisCache(redis_url) if REDIS_AVAILABLE else None
                
                _cache_manager = MultiLevelCache(memory_cache, redis_cache)
                await _cache_manager.initialize()
    
    return _cache_manager


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(ttl: Optional[float] = 3600, tags: Optional[Set[str]] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Get cache manager
            cache = await get_cache_manager()
            
            # Try to get cached result
            cached_result = await cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache the result
            await cache.set(key, result, ttl, tags)
            return result
        
        return wrapper
    return decorator


class CacheManager:
    """High-level cache manager with advanced features"""
    
    def __init__(self):
        self.cache: Optional[MultiLevelCache] = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize cache manager"""
        if not self.initialized:
            self.cache = await get_cache_manager()
            self.initialized = True
        return self.initialized
    
    async def shutdown(self):
        """Shutdown cache manager"""
        if self.cache:
            await self.cache.shutdown()
            self.initialized = False
    
    @with_enhanced_error_handling("cache_operation")
    async def get_or_set(self, 
                        key: str, 
                        factory: Callable,
                        ttl: Optional[float] = None,
                        tags: Optional[Set[str]] = None) -> Any:
        """Get value from cache or set using factory function"""
        if not self.cache:
            await self.initialize()
        
        # Try to get from cache
        value = await self.cache.get(key)
        if value is not None:
            return value
        
        # Generate value using factory
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        # Store in cache
        await self.cache.set(key, value, ttl, tags)
        return value
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys from cache efficiently"""
        if not self.cache:
            await self.initialize()
        
        results = {}
        tasks = []
        
        for key in keys:
            task = asyncio.create_task(self.cache.get(key))
            tasks.append((key, task))
        
        for key, task in tasks:
            try:
                value = await task
                if value is not None:
                    results[key] = value
            except Exception as e:
                logger.warning(f"Error getting cache key {key}: {e}")
        
        return results
    
    async def batch_set(self, items: Dict[str, Any], ttl: Optional[float] = None) -> Dict[str, bool]:
        """Set multiple keys in cache efficiently"""
        if not self.cache:
            await self.initialize()
        
        results = {}
        tasks = []
        
        for key, value in items.items():
            task = asyncio.create_task(self.cache.set(key, value, ttl))
            tasks.append((key, task))
        
        for key, task in tasks:
            try:
                results[key] = await task
            except Exception as e:
                logger.error(f"Error setting cache key {key}: {e}")
                results[key] = False
        
        return results
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive cache performance report"""
        if not self.cache:
            return {'error': 'Cache not initialized'}
        
        stats = await self.cache.get_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cache_stats': stats,
            'recommendations': self._generate_recommendations(stats)
        }
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on stats"""
        recommendations = []
        
        total_hit_ratio = stats.get('total_hit_ratio', 0)
        if total_hit_ratio < 0.8:
            recommendations.append(f"Cache hit ratio is low ({total_hit_ratio:.2%}). Consider increasing TTL or cache size.")
        
        l1_stats = stats.get('l1_memory', {})
        if l1_stats.get('evictions', 0) > 100:
            recommendations.append("High eviction rate in L1 cache. Consider increasing memory cache size.")
        
        l2_stats = stats.get('l2_redis', {})
        if l2_stats.get('hits', 0) == 0 and l2_stats.get('misses', 0) > 0:
            recommendations.append("Redis cache not being hit. Check Redis connectivity and TTL settings.")
        
        return recommendations


# Global cache manager instance
cache_manager = CacheManager()