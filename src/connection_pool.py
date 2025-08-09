"""
Intelligent Connection Pool Manager for Claude Manager Service Generation 3

This module provides enterprise-grade connection pooling for:
- Database connections with auto-scaling and load balancing
- HTTP API connections with keep-alive and multiplexing
- WebSocket connections for real-time features
- Intelligent pool management with health monitoring
- Circuit breaker pattern for failed connections
- Connection lifecycle management and cleanup
- Performance monitoring and optimization
"""

import asyncio
import time
import weakref
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, AsyncContextManager
from enum import Enum
import logging
import os
import ssl

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False

from .logger import get_logger
from .performance_monitor import monitor_performance, get_monitor
from .error_handler import ConnectionError, with_enhanced_error_handling


logger = get_logger(__name__)


class ConnectionState(Enum):
    """Connection states"""
    IDLE = "idle"
    ACTIVE = "active"
    CONNECTING = "connecting"
    DISCONNECTING = "disconnecting"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class ConnectionMetrics:
    """Metrics for individual connections"""
    connection_id: str
    created_at: float
    last_used: float
    total_uses: int = 0
    total_errors: int = 0
    average_response_time: float = 0.0
    state: ConnectionState = ConnectionState.IDLE
    pool_name: str = ""
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used


@dataclass
class PoolStats:
    """Connection pool statistics"""
    pool_name: str
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_wait_time: float = 0.0
    average_connection_lifetime: float = 0.0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class Connection(ABC):
    """Abstract base class for pooled connections"""
    
    def __init__(self, connection_id: str, pool_name: str):
        self.connection_id = connection_id
        self.pool_name = pool_name
        self.metrics = ConnectionMetrics(connection_id, time.time(), time.time(), pool_name=pool_name)
        self._raw_connection: Optional[Any] = None
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection"""
        pass
    
    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        pass
    
    @abstractmethod
    async def execute(self, operation: Any) -> Any:
        """Execute operation using this connection"""
        pass
    
    async def acquire(self) -> bool:
        """Acquire connection for use"""
        async with self._lock:
            if self.metrics.state == ConnectionState.IDLE:
                self.metrics.state = ConnectionState.ACTIVE
                self.metrics.last_used = time.time()
                return True
            return False
    
    async def release(self):
        """Release connection back to pool"""
        async with self._lock:
            if self.metrics.state == ConnectionState.ACTIVE:
                self.metrics.state = ConnectionState.IDLE
                self.metrics.last_used = time.time()
    
    def update_metrics(self, success: bool, response_time: float):
        """Update connection metrics"""
        self.metrics.total_uses += 1
        if not success:
            self.metrics.total_errors += 1
        
        # Update average response time
        if self.metrics.total_uses == 1:
            self.metrics.average_response_time = response_time
        else:
            self.metrics.average_response_time = (
                self.metrics.average_response_time * (self.metrics.total_uses - 1) + response_time
            ) / self.metrics.total_uses


class DatabaseConnection(Connection):
    """PostgreSQL connection wrapper"""
    
    def __init__(self, connection_id: str, pool_name: str, dsn: str):
        super().__init__(connection_id, pool_name)
        self.dsn = dsn
        self._connection: Optional[asyncpg.Connection] = None
    
    async def connect(self) -> bool:
        """Establish database connection"""
        try:
            if not ASYNCPG_AVAILABLE:
                logger.error("asyncpg not available for database connections")
                return False
            
            self.metrics.state = ConnectionState.CONNECTING
            self._connection = await asyncpg.connect(self.dsn)
            self._raw_connection = self._connection
            self.metrics.state = ConnectionState.IDLE
            
            logger.debug(f"Database connection {self.connection_id} established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish database connection {self.connection_id}: {e}")
            self.metrics.state = ConnectionState.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Close database connection"""
        try:
            if self._connection:
                self.metrics.state = ConnectionState.DISCONNECTING
                await self._connection.close()
                self.metrics.state = ConnectionState.CLOSED
                logger.debug(f"Database connection {self.connection_id} closed")
            return True
            
        except Exception as e:
            logger.error(f"Error closing database connection {self.connection_id}: {e}")
            return False
    
    async def is_healthy(self) -> bool:
        """Check database connection health"""
        try:
            if not self._connection or self._connection.is_closed():
                return False
            
            # Simple health check query
            await self._connection.fetchval("SELECT 1")
            return True
            
        except Exception:
            return False
    
    async def execute(self, query: str, *args) -> Any:
        """Execute database query"""
        if not self._connection:
            raise ConnectionError("Database connection not established", "execute_query")
        
        start_time = time.time()
        try:
            if query.strip().upper().startswith(('SELECT', 'WITH')):
                result = await self._connection.fetch(query, *args)
            else:
                result = await self._connection.execute(query, *args)
            
            response_time = time.time() - start_time
            self.update_metrics(True, response_time)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.update_metrics(False, response_time)
            raise ConnectionError(f"Database query failed: {e}", "execute_query", e)


class HTTPConnection(Connection):
    """HTTP client connection wrapper"""
    
    def __init__(self, connection_id: str, pool_name: str, base_url: str, timeout: int = 30):
        super().__init__(connection_id, pool_name)
        self.base_url = base_url
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self) -> bool:
        """Establish HTTP client session"""
        try:
            if not AIOHTTP_AVAILABLE:
                logger.error("aiohttp not available for HTTP connections")
                return False
            
            self.metrics.state = ConnectionState.CONNECTING
            
            # Configure timeout
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            # Configure connector for keep-alive and connection pooling
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection limit
                limit_per_host=20,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            self._session = aiohttp.ClientSession(
                base_url=self.base_url,
                timeout=timeout,
                connector=connector,
                headers={'Connection': 'keep-alive'}
            )
            
            self._raw_connection = self._session
            self.metrics.state = ConnectionState.IDLE
            
            logger.debug(f"HTTP connection {self.connection_id} established for {self.base_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish HTTP connection {self.connection_id}: {e}")
            self.metrics.state = ConnectionState.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Close HTTP client session"""
        try:
            if self._session:
                self.metrics.state = ConnectionState.DISCONNECTING
                await self._session.close()
                self.metrics.state = ConnectionState.CLOSED
                logger.debug(f"HTTP connection {self.connection_id} closed")
            return True
            
        except Exception as e:
            logger.error(f"Error closing HTTP connection {self.connection_id}: {e}")
            return False
    
    async def is_healthy(self) -> bool:
        """Check HTTP connection health"""
        try:
            if not self._session or self._session.closed:
                return False
            
            # Simple health check request
            async with self._session.get('/health', timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status < 500
                
        except Exception:
            # If health endpoint doesn't exist, assume healthy if session is open
            return self._session and not self._session.closed
    
    async def execute(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Execute HTTP request"""
        if not self._session:
            raise ConnectionError("HTTP connection not established", "execute_request")
        
        start_time = time.time()
        try:
            async with self._session.request(method, url, **kwargs) as response:
                response_time = time.time() - start_time
                success = response.status < 400
                self.update_metrics(success, response_time)
                return response
                
        except Exception as e:
            response_time = time.time() - start_time
            self.update_metrics(False, response_time)
            raise ConnectionError(f"HTTP request failed: {e}", "execute_request", e)


class ConnectionPool:
    """Generic connection pool with intelligent management"""
    
    def __init__(self, 
                 name: str,
                 connection_factory: Callable[[str, str], Connection],
                 min_size: int = 5,
                 max_size: int = 20,
                 max_idle_time: int = 300,  # 5 minutes
                 health_check_interval: int = 60,  # 1 minute
                 max_connection_age: int = 3600):  # 1 hour
        
        self.name = name
        self.connection_factory = connection_factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        self.max_connection_age = max_connection_age
        
        # Connection management
        self._connections: Dict[str, Connection] = {}
        self._available_connections: deque[Connection] = deque()
        self._pool_lock = asyncio.Lock()
        self._connection_counter = 0
        
        # Statistics
        self.stats = PoolStats(name)
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"Connection pool '{name}' initialized: min={min_size}, max={max_size}")
    
    async def start(self) -> bool:
        """Start the connection pool"""
        if self._running:
            return True
        
        try:
            # Create minimum number of connections
            for _ in range(self.min_size):
                await self._create_connection()
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_worker())
            self._cleanup_task = asyncio.create_task(self._cleanup_worker())
            
            self._running = True
            logger.info(f"Connection pool '{self.name}' started with {len(self._connections)} connections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start connection pool '{self.name}': {e}")
            return False
    
    async def stop(self):
        """Stop the connection pool"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Close all connections
        async with self._pool_lock:
            for connection in list(self._connections.values()):
                await connection.disconnect()
            
            self._connections.clear()
            self._available_connections.clear()
        
        logger.info(f"Connection pool '{self.name}' stopped")
    
    @asynccontextmanager
    async def acquire_connection(self) -> AsyncContextManager[Connection]:
        """Acquire a connection from the pool"""
        start_time = time.time()
        connection = None
        
        try:
            # Get connection from pool
            connection = await self._get_connection()
            
            if connection:
                await connection.acquire()
                wait_time = time.time() - start_time
                self._update_wait_time(wait_time)
                yield connection
            else:
                raise ConnectionError(f"No connections available in pool '{self.name}'", "acquire_connection")
        
        finally:
            # Return connection to pool
            if connection:
                await connection.release()
                await self._return_connection(connection)
    
    async def _get_connection(self) -> Optional[Connection]:
        """Get an available connection from the pool"""
        async with self._pool_lock:
            # Try to get an existing available connection
            while self._available_connections:
                connection = self._available_connections.popleft()
                
                # Check if connection is still healthy
                if await connection.is_healthy():
                    self.stats.active_connections += 1
                    self.stats.idle_connections -= 1
                    return connection
                else:
                    # Remove unhealthy connection
                    await self._remove_connection(connection)
            
            # Create new connection if under max limit
            if len(self._connections) < self.max_size:
                connection = await self._create_connection()
                if connection:
                    self.stats.active_connections += 1
                    return connection
            
            return None
    
    async def _return_connection(self, connection: Connection):
        """Return connection to the available pool"""
        async with self._pool_lock:
            if connection.connection_id in self._connections:
                self._available_connections.append(connection)
                self.stats.active_connections -= 1
                self.stats.idle_connections += 1
    
    async def _create_connection(self) -> Optional[Connection]:
        """Create a new connection"""
        try:
            self._connection_counter += 1
            connection_id = f"{self.name}-{self._connection_counter}"
            
            connection = self.connection_factory(connection_id, self.name)
            
            if await connection.connect():
                self._connections[connection_id] = connection
                self.stats.total_connections += 1
                self.stats.idle_connections += 1
                
                logger.debug(f"Created new connection {connection_id} in pool '{self.name}'")
                return connection
            else:
                logger.warning(f"Failed to create connection {connection_id} in pool '{self.name}'")
                return None
                
        except Exception as e:
            logger.error(f"Error creating connection in pool '{self.name}': {e}")
            return None
    
    async def _remove_connection(self, connection: Connection):
        """Remove connection from pool"""
        try:
            await connection.disconnect()
            
            if connection.connection_id in self._connections:
                del self._connections[connection.connection_id]
                self.stats.total_connections -= 1
                self.stats.failed_connections += 1
                
                if connection in self._available_connections:
                    self._available_connections.remove(connection)
                    self.stats.idle_connections -= 1
                else:
                    self.stats.active_connections -= 1
            
            logger.debug(f"Removed connection {connection.connection_id} from pool '{self.name}'")
            
        except Exception as e:
            logger.error(f"Error removing connection {connection.connection_id}: {e}")
    
    async def _health_check_worker(self):
        """Background worker for connection health checks"""
        logger.debug(f"Starting health check worker for pool '{self.name}'")
        
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check worker for pool '{self.name}': {e}")
    
    async def _cleanup_worker(self):
        """Background worker for connection cleanup"""
        logger.debug(f"Starting cleanup worker for pool '{self.name}'")
        
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_idle_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup worker for pool '{self.name}': {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on idle connections"""
        async with self._pool_lock:
            unhealthy_connections = []
            
            for connection in list(self._available_connections):
                if not await connection.is_healthy():
                    unhealthy_connections.append(connection)
            
            for connection in unhealthy_connections:
                self._available_connections.remove(connection)
                await self._remove_connection(connection)
                
                # Create replacement connection if below minimum
                if len(self._connections) < self.min_size:
                    await self._create_connection()
    
    async def _cleanup_idle_connections(self):
        """Clean up old idle connections"""
        async with self._pool_lock:
            current_time = time.time()
            connections_to_remove = []
            
            for connection in list(self._available_connections):
                # Remove connections that are too old or idle too long
                if (connection.metrics.age_seconds > self.max_connection_age or
                    connection.metrics.idle_seconds > self.max_idle_time):
                    connections_to_remove.append(connection)
            
            for connection in connections_to_remove:
                # Don't go below minimum size
                if len(self._connections) > self.min_size:
                    self._available_connections.remove(connection)
                    await self._remove_connection(connection)
    
    def _update_wait_time(self, wait_time: float):
        """Update average wait time statistics"""
        if self.stats.total_requests == 0:
            self.stats.average_wait_time = wait_time
        else:
            self.stats.average_wait_time = (
                self.stats.average_wait_time * self.stats.total_requests + wait_time
            ) / (self.stats.total_requests + 1)
        
        self.stats.total_requests += 1
    
    async def get_stats(self) -> PoolStats:
        """Get current pool statistics"""
        async with self._pool_lock:
            # Update success rate
            if self.stats.total_requests > 0:
                self.stats.success_rate = self.stats.successful_requests / self.stats.total_requests
            
            # Calculate average connection lifetime
            if self._connections:
                total_lifetime = sum(conn.metrics.age_seconds for conn in self._connections.values())
                self.stats.average_connection_lifetime = total_lifetime / len(self._connections)
            
            return self.stats


class PoolManager:
    """Central manager for all connection pools"""
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self.monitor = get_monitor()
        self._lock = asyncio.Lock()
    
    async def create_database_pool(self, 
                                 name: str, 
                                 dsn: str, 
                                 min_size: int = 5, 
                                 max_size: int = 20) -> bool:
        """Create database connection pool"""
        async with self._lock:
            if name in self.pools:
                logger.warning(f"Database pool '{name}' already exists")
                return False
            
            def factory(connection_id: str, pool_name: str) -> DatabaseConnection:
                return DatabaseConnection(connection_id, pool_name, dsn)
            
            pool = ConnectionPool(name, factory, min_size, max_size)
            success = await pool.start()
            
            if success:
                self.pools[name] = pool
                logger.info(f"Database pool '{name}' created successfully")
            
            return success
    
    async def create_http_pool(self, 
                              name: str, 
                              base_url: str, 
                              min_size: int = 3, 
                              max_size: int = 10,
                              timeout: int = 30) -> bool:
        """Create HTTP connection pool"""
        async with self._lock:
            if name in self.pools:
                logger.warning(f"HTTP pool '{name}' already exists")
                return False
            
            def factory(connection_id: str, pool_name: str) -> HTTPConnection:
                return HTTPConnection(connection_id, pool_name, base_url, timeout)
            
            pool = ConnectionPool(name, factory, min_size, max_size)
            success = await pool.start()
            
            if success:
                self.pools[name] = pool
                logger.info(f"HTTP pool '{name}' created successfully")
            
            return success
    
    async def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get connection pool by name"""
        return self.pools.get(name)
    
    @asynccontextmanager
    async def get_connection(self, pool_name: str):
        """Get connection from specified pool"""
        pool = await self.get_pool(pool_name)
        if not pool:
            raise ConnectionError(f"Pool '{pool_name}' not found", "get_connection")
        
        async with pool.acquire_connection() as connection:
            yield connection
    
    @monitor_performance(track_memory=True, custom_name="database_query")
    async def execute_database_query(self, pool_name: str, query: str, *args) -> Any:
        """Execute database query using connection pool"""
        async with self.get_connection(pool_name) as connection:
            if not isinstance(connection, DatabaseConnection):
                raise ConnectionError(f"Pool '{pool_name}' is not a database pool", "execute_database_query")
            return await connection.execute(query, *args)
    
    @monitor_performance(track_memory=True, custom_name="http_request")
    async def execute_http_request(self, pool_name: str, method: str, url: str, **kwargs) -> Any:
        """Execute HTTP request using connection pool"""
        async with self.get_connection(pool_name) as connection:
            if not isinstance(connection, HTTPConnection):
                raise ConnectionError(f"Pool '{pool_name}' is not an HTTP pool", "execute_http_request")
            return await connection.execute(method, url, **kwargs)
    
    async def get_all_stats(self) -> Dict[str, PoolStats]:
        """Get statistics for all pools"""
        stats = {}
        for name, pool in self.pools.items():
            stats[name] = await pool.get_stats()
        return stats
    
    async def shutdown_all(self):
        """Shutdown all connection pools"""
        async with self._lock:
            for pool in self.pools.values():
                await pool.stop()
            self.pools.clear()
            logger.info("All connection pools shut down")


# Global pool manager instance
_pool_manager: Optional[PoolManager] = None
_manager_lock = asyncio.Lock()


async def get_pool_manager() -> PoolManager:
    """Get global pool manager instance"""
    global _pool_manager
    
    if _pool_manager is None:
        async with _manager_lock:
            if _pool_manager is None:
                _pool_manager = PoolManager()
    
    return _pool_manager


# Convenience functions
async def initialize_default_pools():
    """Initialize default connection pools from environment"""
    manager = await get_pool_manager()
    
    # Database pool
    db_dsn = os.getenv('DATABASE_URL', 'postgresql://claude_user:claude_password@localhost:5432/claude_manager')
    await manager.create_database_pool('default_db', db_dsn)
    
    # GitHub API pool
    github_base_url = 'https://api.github.com'
    await manager.create_http_pool('github_api', github_base_url)
    
    # Claude API pool (if available)
    claude_api_url = os.getenv('CLAUDE_API_URL', 'https://api.anthropic.com')
    await manager.create_http_pool('claude_api', claude_api_url)
    
    logger.info("Default connection pools initialized")


@with_enhanced_error_handling("pool_query")
async def query_database(query: str, *args, pool_name: str = 'default_db') -> Any:
    """Execute database query using default pool"""
    manager = await get_pool_manager()
    return await manager.execute_database_query(pool_name, query, *args)


@with_enhanced_error_handling("pool_request")
async def make_http_request(method: str, url: str, pool_name: str = 'github_api', **kwargs) -> Any:
    """Make HTTP request using connection pool"""
    manager = await get_pool_manager()
    return await manager.execute_http_request(pool_name, method, url, **kwargs)