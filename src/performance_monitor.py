"""
Performance Monitoring Module for Claude Manager Service

This module provides comprehensive performance tracking including:
- Function execution times and statistics
- API call success/failure rates
- Memory usage tracking
- Detailed operation metrics
- Performance trend analysis
- Configurable alerting thresholds
"""

import json
import os
import time
import psutil
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from functools import wraps
import statistics

from .logger import get_logger


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


class PerformanceMonitor:
    """
    Centralized performance monitoring system
    
    Features:
    - Real-time metrics collection
    - Configurable retention periods
    - Performance trend analysis
    - Memory usage tracking
    - Automated alerting for performance degradation
    - JSON-based persistence
    """
    
    _instance: Optional['PerformanceMonitor'] = None
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
        
        # Configuration (configurable via environment variables)
        try:
            from .config_env import get_env_config
            config = get_env_config()
            self.max_operations_in_memory = config.perf_max_operations
            self.retention_days = config.perf_retention_days
            self.alert_threshold_duration = config.perf_alert_duration
            self.alert_threshold_error_rate = config.perf_alert_error_rate
        except ImportError:
            # Fallback to direct environment variable access
            self.max_operations_in_memory = int(os.getenv('PERF_MAX_OPERATIONS', '10000'))
            self.retention_days = int(os.getenv('PERF_RETENTION_DAYS', '30'))
            self.alert_threshold_duration = float(os.getenv('PERF_ALERT_DURATION', '30.0'))  # seconds
            self.alert_threshold_error_rate = float(os.getenv('PERF_ALERT_ERROR_RATE', '0.1'))  # 10%
        self.data_dir = Path(__file__).parent.parent / 'performance_data'
        self.data_dir.mkdir(exist_ok=True)
        
        # In-memory storage
        self.operations: deque = deque(maxlen=self.max_operations_in_memory)
        self.function_stats: Dict[str, List[OperationMetrics]] = defaultdict(list)
        self.api_call_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'success': 0, 'failure': 0})
        
        # Alerting state
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(hours=1)
        
        # Load existing data
        self._load_historical_data()
        
        # Start background cleanup task
        self._start_cleanup_task()
        
        self._initialized = True
        self.logger.info("Performance monitor initialized successfully")
    
    def _start_cleanup_task(self):
        """Start background thread for periodic data cleanup"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._cleanup_old_data()
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.logger.debug("Started background cleanup task")
    
    def _cleanup_old_data(self):
        """Remove old performance data based on retention policy"""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        cutoff_timestamp = cutoff_time.timestamp()
        
        # Clean in-memory operations
        initial_count = len(self.operations)
        self.operations = deque(
            [op for op in self.operations if op.start_time >= cutoff_timestamp],
            maxlen=self.max_operations_in_memory
        )
        
        # Clean function stats
        for func_name in list(self.function_stats.keys()):
            self.function_stats[func_name] = [
                op for op in self.function_stats[func_name] 
                if op.start_time >= cutoff_timestamp
            ]
            if not self.function_stats[func_name]:
                del self.function_stats[func_name]
        
        cleaned_count = initial_count - len(self.operations)
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old performance records")
    
    def record_operation(self, metrics: OperationMetrics):
        """Record metrics for a completed operation"""
        try:
            with self._lock:
                self.operations.append(metrics)
                self.function_stats[f"{metrics.module_name}.{metrics.function_name}"].append(metrics)
                
                # Update API call stats if this looks like an API operation
                if any(api_term in metrics.function_name.lower() for api_term in ['api', 'request', 'call', 'fetch']):
                    api_key = f"{metrics.module_name}.{metrics.function_name}"
                    if metrics.success:
                        self.api_call_stats[api_key]['success'] += 1
                    else:
                        self.api_call_stats[api_key]['failure'] += 1
                
                # Check for performance alerts
                self._check_performance_alerts(metrics)
                
        except Exception as e:
            self.logger.error(f"Error recording operation metrics: {e}")
    
    def _check_performance_alerts(self, metrics: OperationMetrics):
        """Check if operation metrics trigger any alerts"""
        func_key = f"{metrics.module_name}.{metrics.function_name}"
        
        # Check duration alert
        if metrics.duration > self.alert_threshold_duration:
            self._maybe_send_alert(
                func_key,
                f"Slow operation detected: {func_key} took {metrics.duration:.2f}s"
            )
        
        # Check error rate alert
        if not metrics.success:
            recent_ops = self.function_stats[func_key][-10:]  # Last 10 operations
            if len(recent_ops) >= 5:  # Only alert if we have enough data
                error_rate = sum(1 for op in recent_ops if not op.success) / len(recent_ops)
                if error_rate >= self.alert_threshold_error_rate:
                    self._maybe_send_alert(
                        func_key,
                        f"High error rate detected: {func_key} has {error_rate:.1%} errors in recent operations"
                    )
    
    def _maybe_send_alert(self, key: str, message: str):
        """Send alert if cooldown period has passed"""
        now = datetime.now()
        last_alert = self.last_alert_times.get(key)
        
        if last_alert is None or (now - last_alert) > self.alert_cooldown:
            self.logger.warning(f"PERFORMANCE ALERT: {message}")
            self.last_alert_times[key] = now
    
    def get_function_metrics(self, function_name: str, module_name: str = None) -> Optional[AggregateMetrics]:
        """Get aggregated metrics for a specific function"""
        try:
            # Find matching operations
            if module_name:
                key = f"{module_name}.{function_name}"
                operations = self.function_stats.get(key, [])
            else:
                # Search across all modules
                operations = []
                for key, ops in self.function_stats.items():
                    if key.endswith(f".{function_name}"):
                        operations.extend(ops)
            
            if not operations:
                return None
            
            # Calculate aggregated metrics
            successful_ops = [op for op in operations if op.success]
            failed_ops = [op for op in operations if not op.success]
            durations = [op.duration for op in operations]
            
            # Calculate duration statistics
            durations.sort()
            
            # Error type analysis
            error_types = defaultdict(int)
            for op in failed_ops:
                if op.error_message:
                    error_type = type(op.error_message).__name__ if hasattr(op.error_message, '__class__') else 'Unknown'
                    error_types[error_type] += 1
            
            # Memory usage statistics
            memory_usages = [op.memory_delta for op in operations if op.memory_delta is not None]
            avg_memory_usage = statistics.mean(memory_usages) if memory_usages else None
            
            return AggregateMetrics(
                function_name=function_name,
                module_name=module_name or "mixed",
                total_calls=len(operations),
                successful_calls=len(successful_ops),
                failed_calls=len(failed_ops),
                success_rate=len(successful_ops) / len(operations) if operations else 0,
                total_duration=sum(durations),
                average_duration=statistics.mean(durations),
                min_duration=min(durations),
                max_duration=max(durations),
                median_duration=statistics.median(durations),
                p95_duration=self._percentile(durations, 95),
                p99_duration=self._percentile(durations, 99),
                last_called=datetime.fromtimestamp(max(op.end_time for op in operations)),
                first_called=datetime.fromtimestamp(min(op.start_time for op in operations)),
                error_types=dict(error_types),
                avg_memory_usage=avg_memory_usage
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating function metrics: {e}")
            return None
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value from sorted data"""
        if not data:
            return 0.0
        n = len(data)
        k = (n - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f == n - 1:
            return data[f]
        return data[f] * (1 - c) + data[f + 1] * c
    
    def get_api_call_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of API call statistics"""
        summary = {}
        for api_key, stats in self.api_call_stats.items():
            total_calls = stats['success'] + stats['failure']
            success_rate = stats['success'] / total_calls if total_calls > 0 else 0
            
            summary[api_key] = {
                'total_calls': total_calls,
                'successful_calls': stats['success'],
                'failed_calls': stats['failure'],
                'success_rate': success_rate,
                'error_rate': 1 - success_rate
            }
        
        return summary
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        recent_operations = [op for op in self.operations if op.start_time >= cutoff_time]
        
        if not recent_operations:
            return {"message": f"No operations recorded in the last {hours} hours"}
        
        # Overall statistics
        total_operations = len(recent_operations)
        successful_operations = sum(1 for op in recent_operations if op.success)
        failed_operations = total_operations - successful_operations
        
        # Duration statistics
        durations = [op.duration for op in recent_operations]
        durations.sort()
        
        # Function breakdown
        function_breakdown = defaultdict(lambda: {'count': 0, 'success': 0, 'total_duration': 0})
        for op in recent_operations:
            key = f"{op.module_name}.{op.function_name}"
            function_breakdown[key]['count'] += 1
            if op.success:
                function_breakdown[key]['success'] += 1
            function_breakdown[key]['total_duration'] += op.duration
        
        # Convert to regular dict and add calculated fields
        function_breakdown = {
            func: {
                **stats,
                'success_rate': stats['success'] / stats['count'],
                'avg_duration': stats['total_duration'] / stats['count']
            }
            for func, stats in function_breakdown.items()
        }
        
        # Memory usage statistics
        memory_ops = [op for op in recent_operations if op.memory_delta is not None]
        memory_stats = None
        if memory_ops:
            memory_deltas = [op.memory_delta for op in memory_ops]
            memory_stats = {
                'avg_memory_delta': statistics.mean(memory_deltas),
                'max_memory_delta': max(memory_deltas),
                'min_memory_delta': min(memory_deltas),
                'total_operations_with_memory_tracking': len(memory_ops)
            }
        
        return {
            'report_period_hours': hours,
            'generated_at': datetime.now().isoformat(),
            'overall_stats': {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'failed_operations': failed_operations,
                'success_rate': successful_operations / total_operations,
                'total_duration': sum(durations),
                'average_duration': statistics.mean(durations),
                'median_duration': statistics.median(durations),
                'p95_duration': self._percentile(durations, 95),
                'p99_duration': self._percentile(durations, 99)
            },
            'function_breakdown': function_breakdown,
            'api_call_summary': self.get_api_call_summary(),
            'memory_stats': memory_stats,
            'slowest_operations': [
                {
                    'function': f"{op.module_name}.{op.function_name}",
                    'duration': op.duration,
                    'timestamp': datetime.fromtimestamp(op.start_time).isoformat(),
                    'success': op.success
                }
                for op in sorted(recent_operations, key=lambda x: x.duration, reverse=True)[:10]
            ]
        }
    
    def save_metrics(self, filename: str = None):
        """Save current metrics to disk"""
        if filename is None:
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.data_dir / filename
        
        try:
            data = {
                'saved_at': datetime.now().isoformat(),
                'operations': [asdict(op) for op in self.operations],
                'api_call_stats': dict(self.api_call_stats),
                'configuration': {
                    'max_operations_in_memory': self.max_operations_in_memory,
                    'retention_days': self.retention_days,
                    'alert_threshold_duration': self.alert_threshold_duration,
                    'alert_threshold_error_rate': self.alert_threshold_error_rate
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Performance metrics saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            return None
    
    def _load_historical_data(self):
        """Load the most recent historical data on startup"""
        try:
            # Find the most recent metrics file
            metrics_files = sorted(self.data_dir.glob("performance_metrics_*.json"))
            if not metrics_files:
                self.logger.info("No historical performance data found")
                return
            
            latest_file = metrics_files[-1]
            self.logger.info(f"Loading historical performance data from {latest_file}")
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Load operations
            for op_data in data.get('operations', []):
                try:
                    metrics = OperationMetrics(**op_data)
                    self.operations.append(metrics)
                    self.function_stats[f"{metrics.module_name}.{metrics.function_name}"].append(metrics)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid operation data: {e}")
            
            # Load API call stats
            for api_key, stats in data.get('api_call_stats', {}).items():
                self.api_call_stats[api_key].update(stats)
            
            self.logger.info(f"Loaded {len(self.operations)} historical operations")
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")


# Global instance
_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _monitor


def monitor_performance(
    track_memory: bool = False,
    custom_name: str = None
) -> Callable:
    """
    Enhanced decorator for performance monitoring with detailed metrics collection
    
    Args:
        track_memory: Whether to track memory usage during execution
        custom_name: Custom name for the operation (defaults to function name)
    
    Example:
        @monitor_performance(track_memory=True)
        def expensive_operation():
            # This will be monitored for execution time and memory usage
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            
            # Prepare metrics collection
            operation_name = custom_name or func.__name__
            module_name = func.__module__
            start_time = time.time()
            
            # Memory tracking
            memory_before = None
            memory_after = None
            memory_delta = None
            
            if track_memory:
                try:
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                except Exception:
                    pass
            
            # Execute function
            success = True
            error_message = None
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                # Finalize metrics collection
                end_time = time.time()
                duration = end_time - start_time
                
                if track_memory and memory_before is not None:
                    try:
                        process = psutil.Process()
                        memory_after = process.memory_info().rss / 1024 / 1024  # MB
                        memory_delta = memory_after - memory_before
                    except Exception:
                        pass
                
                # Record metrics
                metrics = OperationMetrics(
                    function_name=operation_name,
                    module_name=module_name,
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
                
                monitor.record_operation(metrics)
        
        return wrapper
    return decorator


def monitor_api_call(api_name: str = None) -> Callable:
    """
    Specialized decorator for API call monitoring
    
    Args:
        api_name: Custom name for the API operation
    
    Example:
        @monitor_api_call("github_create_issue")
        def create_github_issue():
            # API call monitoring with specific naming
            pass
    """
    return monitor_performance(track_memory=False, custom_name=api_name)


# Backwards compatibility with existing log_performance decorator
def enhanced_log_performance(func: Callable) -> Callable:
    """Enhanced version of log_performance that includes monitoring"""
    from logger import log_performance
    
    # Apply both decorators
    monitored_func = monitor_performance(track_memory=True)(func)
    logged_func = log_performance(monitored_func)
    
    return logged_func