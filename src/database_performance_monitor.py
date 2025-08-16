"""
Database-backed Performance Monitor for Claude Manager Service

This module provides a database-backed implementation of the PerformanceMonitor
that replaces JSON file-based storage with SQLite database operations.

Features:
- Database persistence for performance metrics
- Async operations for better performance
- Enhanced querying and aggregation capabilities
- Real-time metrics collection with database storage
- Performance trend analysis with SQL
"""

import time
import psutil
import threading
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from functools import wraps
import statistics

from src.logger import get_logger
from src.services.database_service import get_database_service
from src.error_handler import DatabaseError


logger = get_logger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation execution"""
    function_name: str
    module_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_delta: Optional[float] = None
    args_count: int = 0
    kwargs_count: int = 0


@dataclass
class AggregateMetrics:
    """Aggregated metrics for a function over time"""
    function_name: str
    module_name: str
    total_calls: int
    successful_calls: int
    failed_calls: int
    success_rate: float
    total_duration: float
    average_duration: float
    min_duration: float
    max_duration: float
    median_duration: float
    p95_duration: float
    p99_duration: float
    last_called: datetime
    first_called: datetime
    error_types: Dict[str, int]
    avg_memory_usage: Optional[float] = None


class DatabasePerformanceMonitor:
    """
    Database-backed performance monitoring system
    
    This class provides the same interface as the original PerformanceMonitor
    but uses a SQLite database for persistence instead of JSON files.
    
    Features:
    - Database persistence with better querying
    - Real-time metrics collection
    - Enhanced aggregation with SQL
    - Memory usage tracking
    - Performance trend analysis
    """
    
    _instance: Optional['DatabasePerformanceMonitor'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.logger = get_logger(__name__)
        
        # Configuration
        self.enabled = True
        self.memory_tracking_enabled = True
        self.retention_days = 30
        self.alert_thresholds = {
            'api_call_failure_rate': 0.1,
            'memory_usage_mb': 512,
            'function_duration_seconds': 10.0
        }
        
        # Database service
        self._db_service = None
        
        # In-memory cache for recent metrics (last hour)
        self._recent_metrics = defaultdict(list)
        self._metrics_lock = threading.Lock()
        
        # System info
        self._process = psutil.Process()
        self._start_time = time.time()
        
        self._initialized = True
        self.logger.info("Database performance monitor initialized")
    
    async def _get_db_service(self):
        """Get database service instance (lazy initialization)"""
        if self._db_service is None:
            self._db_service = await get_database_service()
        return self._db_service
    
    def monitor_function(self, function_name: str = None, 
                        track_memory: bool = True, 
                        alert_on_failure: bool = True):
        """
        Decorator to monitor function performance with database storage
        
        Args:
            function_name: Custom name for the function (uses actual name if None)
            track_memory: Whether to track memory usage
            alert_on_failure: Whether to log alerts on failure
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            nonlocal function_name
            if function_name is None:
                function_name = func.__name__
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                
                return await self._monitor_execution(
                    func, function_name, track_memory, alert_on_failure, True, args, kwargs
                )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # For sync functions, we'll use asyncio.run to handle the monitoring
                async def run_monitoring():
                    return await self._monitor_execution(
                        func, function_name, track_memory, alert_on_failure, False, args, kwargs
                    )
                
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If event loop is running, create a task
                        task = loop.create_task(run_monitoring())
                        return task
                    else:
                        return loop.run_until_complete(run_monitoring())
                except RuntimeError:
                    # No event loop, create one
                    return asyncio.run(run_monitoring())
            
            # Return appropriate wrapper based on whether function is async
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _monitor_execution(self, func: Callable, function_name: str, 
                               track_memory: bool, alert_on_failure: bool,
                               is_async: bool, args: tuple, kwargs: dict):
        """Monitor function execution and store metrics in database"""
        
        # Pre-execution setup
        start_time = time.time()
        memory_before = None
        
        if track_memory and self.memory_tracking_enabled:
            try:
                memory_before = self._process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                memory_before = None
        
        error_message = None
        success = True
        result = None
        
        try:
            # Execute function
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
        except Exception as e:
            success = False
            error_message = str(e)
            if alert_on_failure:
                self.logger.error(f"Function {function_name} failed: {error_message}")
            raise
        
        finally:
            # Post-execution metrics
            end_time = time.time()
            duration = end_time - start_time
            memory_after = None
            memory_delta = None
            
            if track_memory and self.memory_tracking_enabled and memory_before:
                try:
                    memory_after = self._process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = memory_after - memory_before
                except Exception:
                    pass
            
            # Create metrics object
            metrics = OperationMetrics(
                function_name=function_name,
                module_name=func.__module__ if hasattr(func, '__module__') else 'unknown',
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                error_message=error_message,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_delta,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            # Store metrics (async)
            asyncio.create_task(self._store_metrics(metrics))
            
            # Add to in-memory cache
            with self._metrics_lock:
                self._recent_metrics[function_name].append(metrics)
                # Keep only last hour of metrics in memory
                cutoff_time = time.time() - 3600
                self._recent_metrics[function_name] = [
                    m for m in self._recent_metrics[function_name] 
                    if m.start_time > cutoff_time
                ]
        
        return result
    
    async def _store_metrics(self, metrics: OperationMetrics) -> None:
        """Store metrics in database"""
        try:
            db_service = await self._get_db_service()
            await db_service.save_performance_metric(asdict(metrics))
        except Exception as e:
            self.logger.error(f"Failed to store performance metrics: {e}")
    
    async def get_function_metrics(self, function_name: str = None, 
                                 hours_back: int = 24) -> List[OperationMetrics]:
        """
        Get performance metrics for a specific function
        
        Args:
            function_name: Function name to filter (None for all)
            hours_back: Hours to look back
            
        Returns:
            List of operation metrics
        """
        try:
            db_service = await self._get_db_service()
            raw_metrics = await db_service.get_performance_metrics(function_name, hours_back)
            
            # Convert to OperationMetrics objects
            metrics = []
            for raw_metric in raw_metrics:
                metrics.append(OperationMetrics(
                    function_name=raw_metric['function_name'],
                    module_name=raw_metric['module_name'],
                    start_time=raw_metric['start_time'],
                    end_time=raw_metric['end_time'],
                    duration=raw_metric['duration'],
                    success=bool(raw_metric['success']),
                    error_message=raw_metric.get('error_message'),
                    memory_before=raw_metric.get('memory_before'),
                    memory_after=raw_metric.get('memory_after'),
                    memory_delta=raw_metric.get('memory_delta'),
                    args_count=raw_metric.get('args_count', 0),
                    kwargs_count=raw_metric.get('kwargs_count', 0)
                ))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get function metrics: {e}")
            return []
    
    async def get_aggregate_metrics(self, function_name: str = None, 
                                  hours_back: int = 24) -> List[AggregateMetrics]:
        """
        Get aggregated performance metrics
        
        Args:
            function_name: Function name to filter (None for all)
            hours_back: Hours to look back
            
        Returns:
            List of aggregate metrics
        """
        try:
            db_service = await self._get_db_service()
            conn = await db_service.get_connection()
            
            try:
                # Build query
                query = """
                    SELECT 
                        function_name,
                        module_name,
                        COUNT(*) as total_calls,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls,
                        AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                        SUM(duration) as total_duration,
                        AVG(duration) as average_duration,
                        MIN(duration) as min_duration,
                        MAX(duration) as max_duration,
                        MAX(created_at) as last_called,
                        MIN(created_at) as first_called,
                        AVG(memory_after) as avg_memory_usage
                    FROM performance_metrics
                    WHERE created_at > datetime('now', '-{} hours')
                """.format(hours_back)
                
                params = []
                if function_name:
                    query += " AND function_name = ?"
                    params.append(function_name)
                
                query += " GROUP BY function_name, module_name ORDER BY total_calls DESC"
                
                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    aggregates = []
                    for row in rows:
                        # Get individual durations for percentile calculation
                        duration_query = """
                            SELECT duration FROM performance_metrics
                            WHERE function_name = ? AND module_name = ?
                            AND created_at > datetime('now', '-{} hours')
                            ORDER BY duration
                        """.format(hours_back)
                        
                        async with conn.execute(duration_query, (row[0], row[1])) as duration_cursor:
                            durations = [d[0] for d in await duration_cursor.fetchall()]
                        
                        # Calculate percentiles
                        median_duration = statistics.median(durations) if durations else 0
                        p95_duration = statistics.quantiles(durations, n=20)[18] if len(durations) > 1 else (durations[0] if durations else 0)
                        p99_duration = statistics.quantiles(durations, n=100)[98] if len(durations) > 1 else (durations[0] if durations else 0)
                        
                        # Get error types
                        error_query = """
                            SELECT error_message, COUNT(*) FROM performance_metrics
                            WHERE function_name = ? AND module_name = ? AND success = 0
                            AND created_at > datetime('now', '-{} hours')
                            GROUP BY error_message
                        """.format(hours_back)
                        
                        async with conn.execute(error_query, (row[0], row[1])) as error_cursor:
                            error_rows = await error_cursor.fetchall()
                            error_types = {error: count for error, count in error_rows}
                        
                        aggregate = AggregateMetrics(
                            function_name=row[0],
                            module_name=row[1],
                            total_calls=row[2],
                            successful_calls=row[3],
                            failed_calls=row[4],
                            success_rate=row[5] * 100,  # Convert to percentage
                            total_duration=row[6],
                            average_duration=row[7],
                            min_duration=row[8],
                            max_duration=row[9],
                            median_duration=median_duration,
                            p95_duration=p95_duration,
                            p99_duration=p99_duration,
                            last_called=datetime.fromisoformat(row[10]),
                            first_called=datetime.fromisoformat(row[11]),
                            error_types=error_types,
                            avg_memory_usage=row[12]
                        )
                        
                        aggregates.append(aggregate)
                    
                    return aggregates
                    
            finally:
                await db_service.return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Failed to get aggregate metrics: {e}")
            return []
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current system and performance metrics
        
        Returns:
            Dictionary with current metrics
        """
        try:
            # System metrics
            memory_info = self._process.memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024
            uptime_seconds = time.time() - self._start_time
            
            # Recent performance metrics (from in-memory cache)
            with self._metrics_lock:
                total_operations = sum(len(metrics) for metrics in self._recent_metrics.values())
                successful_operations = sum(
                    len([m for m in metrics if m.success]) 
                    for metrics in self._recent_metrics.values()
                )
                
                success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 100
            
            # Database statistics
            db_service = await self._get_db_service()
            db_stats = await db_service.get_database_statistics()
            
            return {
                "memory_usage_mb": round(memory_usage_mb, 2),
                "peak_memory_mb": round(memory_usage_mb, 2),  # Current as peak for now
                "uptime_seconds": round(uptime_seconds, 2),
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "success_rate": round(success_rate, 2),
                "last_operation_time": datetime.now().isoformat(),
                "operation_metrics": {
                    "functions_monitored": len(self._recent_metrics),
                    "recent_hour_operations": total_operations
                },
                "database_metrics": db_stats,
                "alert_thresholds": self.alert_thresholds,
                "monitoring_enabled": self.enabled
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get current metrics: {e}")
            return {
                "error": str(e),
                "monitoring_enabled": self.enabled
            }
    
    async def cleanup_old_metrics(self, days: int = None) -> int:
        """
        Clean up old performance metrics
        
        Args:
            days: Days to retain (uses retention_days if None)
            
        Returns:
            Number of metrics removed
        """
        if days is None:
            days = self.retention_days
        
        try:
            db_service = await self._get_db_service()
            cleanup_stats = await db_service.cleanup_old_data(days)
            
            metrics_removed = cleanup_stats.get('performance_metrics', 0)
            
            if metrics_removed > 0:
                self.logger.info(f"Cleaned up {metrics_removed} old performance metrics (older than {days} days)")
            
            return metrics_removed
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old metrics: {e}")
            return 0
    
    def set_alert_threshold(self, metric_name: str, threshold: float) -> None:
        """Set alert threshold for a metric"""
        self.alert_thresholds[metric_name] = threshold
        self.logger.info(f"Updated alert threshold for {metric_name}: {threshold}")
    
    def enable_monitoring(self) -> None:
        """Enable performance monitoring"""
        self.enabled = True
        self.logger.info("Performance monitoring enabled")
    
    def disable_monitoring(self) -> None:
        """Disable performance monitoring"""
        self.enabled = False
        self.logger.info("Performance monitoring disabled")


# Global instance
_database_performance_monitor: Optional[DatabasePerformanceMonitor] = None


def get_database_performance_monitor() -> DatabasePerformanceMonitor:
    """
    Get global DatabasePerformanceMonitor instance
    
    Returns:
        DatabasePerformanceMonitor: Global performance monitor instance
    """
    global _database_performance_monitor
    if _database_performance_monitor is None:
        _database_performance_monitor = DatabasePerformanceMonitor()
    return _database_performance_monitor


# Convenience decorators
def monitor_performance(function_name: str = None, track_memory: bool = True):
    """Convenience decorator for monitoring function performance"""
    monitor = get_database_performance_monitor()
    return monitor.monitor_function(function_name, track_memory)


def monitor_api_call(function_name: str = None):
    """Convenience decorator for monitoring API calls"""
    monitor = get_database_performance_monitor()
    return monitor.monitor_function(function_name, track_memory=True, alert_on_failure=True)


# Example usage and testing
async def example_database_performance_monitor():
    """Example of using database performance monitor"""
    try:
        # Get performance monitor
        monitor = get_database_performance_monitor()
        
        # Example monitored function
        @monitor.monitor_function("example_function")
        async def example_function(delay: float = 1.0):
            await asyncio.sleep(delay)
            return f"Completed after {delay} seconds"
        
        # Execute monitored function
        result = await example_function(0.5)
        
        # Get metrics
        current_metrics = await monitor.get_current_metrics()
        function_metrics = await monitor.get_function_metrics("example_function")
        aggregate_metrics = await monitor.get_aggregate_metrics("example_function")
        
        logger.info(f"Database performance monitor example completed")
        logger.info(f"Function result: {result}")
        logger.info(f"Current metrics: {current_metrics}")
        logger.info(f"Function metrics count: {len(function_metrics)}")
        logger.info(f"Aggregate metrics count: {len(aggregate_metrics)}")
        
    except Exception as e:
        logger.error(f"Database performance monitor example failed: {e}")
        raise


if __name__ == "__main__":
    # Test database performance monitor
    asyncio.run(example_database_performance_monitor())