#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - COMPREHENSIVE MONITORING & OBSERVABILITY
Advanced monitoring, metrics, tracing, and alerting system
"""

import asyncio
import json
import time
import psutil
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging

import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
import aiohttp


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Alert definition"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    threshold: float
    metric_name: str
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    duration: int = 300  # seconds
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    
@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profiling data"""
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    call_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MetricsCollector:
    """Advanced metrics collection and aggregation"""
    
    def __init__(self, enable_prometheus: bool = True):
        self.logger = structlog.get_logger("MetricsCollector")
        self.enable_prometheus = enable_prometheus
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_configs: Dict[str, Dict] = {}
        
        # Prometheus metrics
        self.prometheus_registry = CollectorRegistry()
        self.prometheus_metrics: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_profiles: Dict[str, List[PerformanceProfile]] = defaultdict(list)
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0},
            'process_count': 0,
            'open_files': 0
        }
        
        # Start background collection
        self._collection_task = None
        self.collection_interval = 10  # seconds
        
        if enable_prometheus:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        
        # System metrics
        self.prometheus_metrics['cpu_usage'] = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['memory_usage'] = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['disk_usage'] = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.prometheus_registry
        )
        
        # Application metrics
        self.prometheus_metrics['request_count'] = Counter(
            'app_requests_total',
            'Total application requests',
            ['method', 'endpoint', 'status'],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['request_duration'] = Histogram(
            'app_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['task_duration'] = Histogram(
            'sdlc_task_duration_seconds',
            'SDLC task duration in seconds',
            ['task_type', 'executor'],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['task_count'] = Counter(
            'sdlc_tasks_total',
            'Total SDLC tasks executed',
            ['task_type', 'status'],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['error_count'] = Counter(
            'app_errors_total',
            'Total application errors',
            ['error_type', 'component'],
            registry=self.prometheus_registry
        )
    
    async def start_collection(self):
        """Start background metrics collection"""
        
        if self._collection_task is None:
            self._collection_task = asyncio.create_task(self._collect_system_metrics())
            self.logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop background metrics collection"""
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
            self.logger.info("Metrics collection stopped")
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                await self.record_metric('cpu_usage', cpu_percent, {'type': 'system'})
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                await self.record_metric('memory_usage', memory_percent, {'type': 'system'})
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                await self.record_metric('disk_usage', disk_percent, {'type': 'system'})
                
                # Network I/O
                net_io = psutil.net_io_counters()
                await self.record_metric('network_bytes_sent', net_io.bytes_sent, {'type': 'network'})
                await self.record_metric('network_bytes_recv', net_io.bytes_recv, {'type': 'network'})
                
                # Process metrics
                process_count = len(psutil.pids())
                await self.record_metric('process_count', process_count, {'type': 'system'})
                
                # Update Prometheus metrics
                if self.enable_prometheus:
                    self.prometheus_metrics['cpu_usage'].set(cpu_percent)
                    self.prometheus_metrics['memory_usage'].set(memory_percent)
                    self.prometheus_metrics['disk_usage'].set(disk_percent)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error("Error collecting system metrics", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    async def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        
        timestamp = datetime.now(timezone.utc)
        metric_point = MetricPoint(
            timestamp=timestamp,
            value=value,
            labels=labels or {}
        )
        
        self.metrics[name].append(metric_point)
        
        # Log significant metrics
        if name in ['cpu_usage', 'memory_usage'] and value > 80:
            self.logger.warning("High resource usage detected",
                              metric=name,
                              value=value,
                              threshold=80)
    
    def record_performance_profile(self, profile: PerformanceProfile):
        """Record performance profiling data"""
        
        self.performance_profiles[profile.function_name].append(profile)
        
        # Keep only recent profiles (last 1000)
        if len(self.performance_profiles[profile.function_name]) > 1000:
            self.performance_profiles[profile.function_name] = \
                self.performance_profiles[profile.function_name][-1000:]
    
    def get_metric_statistics(self, name: str, time_range: timedelta = None) -> Dict[str, Any]:
        """Get statistics for a metric"""
        
        if name not in self.metrics:
            return {}
        
        # Filter by time range if specified
        points = list(self.metrics[name])
        if time_range:
            cutoff_time = datetime.now(timezone.utc) - time_range
            points = [p for p in points if p.timestamp > cutoff_time]
        
        if not points:
            return {}
        
        values = [p.value for p in points]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else None,
            'first_timestamp': points[0].timestamp.isoformat(),
            'last_timestamp': points[-1].timestamp.isoformat()
        }
    
    def get_performance_summary(self, function_name: str = None) -> Dict[str, Any]:
        """Get performance summary"""
        
        if function_name:
            profiles = self.performance_profiles.get(function_name, [])
            if not profiles:
                return {}
            
            execution_times = [p.execution_time for p in profiles]
            memory_usages = [p.memory_usage for p in profiles]
            
            return {
                'function_name': function_name,
                'call_count': len(profiles),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'max_execution_time': max(execution_times),
                'avg_memory_usage': sum(memory_usages) / len(memory_usages),
                'max_memory_usage': max(memory_usages)
            }
        else:
            # Return summary for all functions
            summary = {}
            for func_name in self.performance_profiles:
                summary[func_name] = self.get_performance_summary(func_name)
            return summary


class AlertManager:
    """Advanced alerting system with multiple notification channels"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.logger = structlog.get_logger("AlertManager")
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, datetime] = {}
        self.notification_channels: List[Callable] = []
        self.alert_history: List[Dict] = []
        
        # Alert evaluation task
        self._evaluation_task = None
        self.evaluation_interval = 30  # seconds
    
    def register_alert(self, alert: Alert):
        """Register a new alert"""
        
        self.alerts[alert.id] = alert
        self.logger.info("Alert registered", alert_id=alert.id, metric=alert.metric_name)
    
    def add_notification_channel(self, channel: Callable):
        """Add notification channel (email, Slack, webhook, etc.)"""
        
        self.notification_channels.append(channel)
    
    async def start_monitoring(self):
        """Start alert monitoring"""
        
        if self._evaluation_task is None:
            self._evaluation_task = asyncio.create_task(self._evaluate_alerts())
            self.logger.info("Alert monitoring started")
    
    async def stop_monitoring(self):
        """Stop alert monitoring"""
        
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
            self._evaluation_task = None
            self.logger.info("Alert monitoring stopped")
    
    async def _evaluate_alerts(self):
        """Evaluate alerts periodically"""
        
        while True:
            try:
                for alert_id, alert in self.alerts.items():
                    await self._evaluate_single_alert(alert)
                
                await asyncio.sleep(self.evaluation_interval)
                
            except Exception as e:
                self.logger.error("Error evaluating alerts", error=str(e))
                await asyncio.sleep(self.evaluation_interval)
    
    async def _evaluate_single_alert(self, alert: Alert):
        """Evaluate a single alert"""
        
        # Get recent metric values
        time_range = timedelta(seconds=alert.duration)
        stats = self.metrics_collector.get_metric_statistics(alert.metric_name, time_range)
        
        if not stats:
            return  # No data available
        
        # Get the latest value for evaluation
        latest_value = stats.get('latest')
        if latest_value is None:
            return
        
        # Evaluate condition
        triggered = self._evaluate_condition(latest_value, alert.threshold, alert.condition)
        
        if triggered:
            await self._trigger_alert(alert, latest_value)
        else:
            # Clear alert if it was active
            if alert.id in self.active_alerts:
                await self._clear_alert(alert)
    
    def _evaluate_condition(self, value: float, threshold: float, condition: str) -> bool:
        """Evaluate alert condition"""
        
        if condition == "gt":
            return value > threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return value == threshold
        else:
            return False
    
    async def _trigger_alert(self, alert: Alert, current_value: float):
        """Trigger an alert"""
        
        # Check if alert is already active (to avoid spam)
        if alert.id in self.active_alerts:
            return
        
        self.active_alerts[alert.id] = datetime.now(timezone.utc)
        alert.last_triggered = datetime.now(timezone.utc)
        alert.trigger_count += 1
        
        # Create alert event
        alert_event = {
            'alert_id': alert.id,
            'title': alert.title,
            'description': alert.description,
            'severity': alert.severity.value,
            'metric_name': alert.metric_name,
            'current_value': current_value,
            'threshold': alert.threshold,
            'condition': alert.condition,
            'triggered_at': alert.last_triggered.isoformat()
        }
        
        # Add to history
        self.alert_history.append(alert_event)
        
        # Log alert
        self.logger.critical("Alert triggered",
                           alert_id=alert.id,
                           severity=alert.severity.value,
                           current_value=current_value,
                           threshold=alert.threshold)
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                await channel(alert_event)
            except Exception as e:
                self.logger.error("Failed to send alert notification",
                                channel=str(channel),
                                error=str(e))
    
    async def _clear_alert(self, alert: Alert):
        """Clear an active alert"""
        
        if alert.id in self.active_alerts:
            del self.active_alerts[alert.id]
            
            self.logger.info("Alert cleared", alert_id=alert.id)
            
            # Create clear event
            clear_event = {
                'alert_id': alert.id,
                'title': f"CLEARED: {alert.title}",
                'description': f"Alert {alert.id} has been cleared",
                'severity': 'info',
                'cleared_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Send clear notifications
            for channel in self.notification_channels:
                try:
                    await channel(clear_event)
                except Exception as e:
                    self.logger.error("Failed to send clear notification",
                                    channel=str(channel),
                                    error=str(e))


class DistributedTracing:
    """Distributed tracing for request/task correlation"""
    
    def __init__(self):
        self.logger = structlog.get_logger("DistributedTracing")
        self.active_traces: Dict[str, Dict] = {}
        self.completed_traces: List[Dict] = []
        self.trace_id_counter = 0
        self.trace_lock = threading.Lock()
    
    def start_trace(self, operation: str, parent_trace_id: str = None) -> str:
        """Start a new trace"""
        
        with self.trace_lock:
            self.trace_id_counter += 1
            trace_id = f"trace_{self.trace_id_counter}_{int(time.time())}"
        
        trace_data = {
            'trace_id': trace_id,
            'parent_trace_id': parent_trace_id,
            'operation': operation,
            'start_time': datetime.now(timezone.utc),
            'end_time': None,
            'duration': None,
            'spans': [],
            'tags': {},
            'status': 'active'
        }
        
        self.active_traces[trace_id] = trace_data
        
        self.logger.debug("Trace started",
                         trace_id=trace_id,
                         operation=operation,
                         parent=parent_trace_id)
        
        return trace_id
    
    def add_span(self, trace_id: str, span_name: str, tags: Dict[str, str] = None):
        """Add a span to a trace"""
        
        if trace_id not in self.active_traces:
            return
        
        span = {
            'span_name': span_name,
            'start_time': datetime.now(timezone.utc),
            'end_time': None,
            'duration': None,
            'tags': tags or {}
        }
        
        self.active_traces[trace_id]['spans'].append(span)
    
    def finish_span(self, trace_id: str, span_name: str):
        """Finish a span in a trace"""
        
        if trace_id not in self.active_traces:
            return
        
        trace = self.active_traces[trace_id]
        for span in reversed(trace['spans']):
            if span['span_name'] == span_name and span['end_time'] is None:
                span['end_time'] = datetime.now(timezone.utc)
                span['duration'] = (span['end_time'] - span['start_time']).total_seconds()
                break
    
    def add_trace_tag(self, trace_id: str, key: str, value: str):
        """Add a tag to a trace"""
        
        if trace_id in self.active_traces:
            self.active_traces[trace_id]['tags'][key] = value
    
    def finish_trace(self, trace_id: str, status: str = 'completed'):
        """Finish a trace"""
        
        if trace_id not in self.active_traces:
            return
        
        trace = self.active_traces[trace_id]
        trace['end_time'] = datetime.now(timezone.utc)
        trace['duration'] = (trace['end_time'] - trace['start_time']).total_seconds()
        trace['status'] = status
        
        # Move to completed traces
        self.completed_traces.append(trace)
        del self.active_traces[trace_id]
        
        # Keep only recent traces (last 1000)
        if len(self.completed_traces) > 1000:
            self.completed_traces = self.completed_traces[-1000:]
        
        self.logger.debug("Trace finished",
                         trace_id=trace_id,
                         duration=trace['duration'],
                         status=status)
    
    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get trace data"""
        
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        
        for trace in self.completed_traces:
            if trace['trace_id'] == trace_id:
                return trace
        
        return None
    
    def get_trace_summary(self, time_range: timedelta = None) -> Dict[str, Any]:
        """Get trace summary statistics"""
        
        traces = self.completed_traces.copy()
        
        if time_range:
            cutoff_time = datetime.now(timezone.utc) - time_range
            traces = [t for t in traces if t['start_time'] > cutoff_time]
        
        if not traces:
            return {}
        
        durations = [t['duration'] for t in traces if t['duration']]
        operations = [t['operation'] for t in traces]
        
        return {
            'total_traces': len(traces),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'operations': {op: operations.count(op) for op in set(operations)}
        }


def performance_monitor(function_name: str = None):
    """Decorator for performance monitoring"""
    
    def decorator(func):
        from functools import wraps
        import tracemalloc
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get function name
            name = function_name or f"{func.__module__}.{func.__qualname__}"
            
            # Start monitoring
            start_time = time.time()
            tracemalloc.start()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                # Record performance data
                end_time = time.time()
                execution_time = end_time - start_time
                
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Create performance profile
                profile = PerformanceProfile(
                    function_name=name,
                    execution_time=execution_time,
                    memory_usage=peak / 1024 / 1024,  # MB
                    cpu_usage=0.0,  # Would need more sophisticated CPU monitoring
                    call_count=1
                )
                
                # This would be injected or retrieved from global context
                # metrics_collector.record_performance_profile(profile)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar implementation for sync functions
            name = function_name or f"{func.__module__}.{func.__qualname__}"
            
            start_time = time.time()
            tracemalloc.start()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                profile = PerformanceProfile(
                    function_name=name,
                    execution_time=execution_time,
                    memory_usage=peak / 1024 / 1024,
                    cpu_usage=0.0,
                    call_count=1
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Notification channel implementations
async def email_notification_channel(alert_event: Dict[str, Any]):
    """Email notification channel (placeholder)"""
    print(f"📧 EMAIL ALERT: {alert_event['title']} - {alert_event['description']}")


async def slack_notification_channel(alert_event: Dict[str, Any]):
    """Slack notification channel (placeholder)"""
    print(f"💬 SLACK ALERT: {alert_event['title']} - {alert_event['description']}")


async def webhook_notification_channel(alert_event: Dict[str, Any]):
    """Webhook notification channel"""
    webhook_url = "https://example.com/webhook"  # Configure as needed
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=alert_event) as response:
                if response.status == 200:
                    print(f"🔗 Webhook notification sent successfully")
                else:
                    print(f"❌ Webhook notification failed: {response.status}")
    except Exception as e:
        print(f"❌ Webhook notification error: {e}")


# Example usage and comprehensive monitoring setup
async def main():
    """Example comprehensive monitoring setup"""
    
    print("📊 Setting up Comprehensive Monitoring System")
    
    # Initialize components
    metrics_collector = MetricsCollector(enable_prometheus=True)
    alert_manager = AlertManager(metrics_collector)
    distributed_tracing = DistributedTracing()
    
    # Start metrics collection
    await metrics_collector.start_collection()
    
    # Setup alerts
    cpu_alert = Alert(
        id="high_cpu_usage",
        title="High CPU Usage",
        description="CPU usage is above 80% for 5 minutes",
        severity=AlertSeverity.HIGH,
        threshold=80.0,
        metric_name="cpu_usage",
        condition="gt",
        duration=300
    )
    
    memory_alert = Alert(
        id="high_memory_usage",
        title="High Memory Usage", 
        description="Memory usage is above 85% for 3 minutes",
        severity=AlertSeverity.CRITICAL,
        threshold=85.0,
        metric_name="memory_usage",
        condition="gt",
        duration=180
    )
    
    alert_manager.register_alert(cpu_alert)
    alert_manager.register_alert(memory_alert)
    
    # Add notification channels
    alert_manager.add_notification_channel(email_notification_channel)
    alert_manager.add_notification_channel(slack_notification_channel)
    alert_manager.add_notification_channel(webhook_notification_channel)
    
    # Start alert monitoring
    await alert_manager.start_monitoring()
    
    # Start Prometheus metrics server
    if metrics_collector.enable_prometheus:
        start_http_server(8000, registry=metrics_collector.prometheus_registry)
        print("🎯 Prometheus metrics server started on port 8000")
    
    # Simulate some activity
    print("🔄 Simulating system activity...")
    
    # Start a trace
    trace_id = distributed_tracing.start_trace("test_operation")
    distributed_tracing.add_span(trace_id, "data_processing")
    
    # Simulate work
    await asyncio.sleep(2)
    
    # Record some custom metrics
    await metrics_collector.record_metric("request_count", 150, {"endpoint": "/api/test"})
    await metrics_collector.record_metric("response_time", 0.25, {"endpoint": "/api/test"})
    
    # Finish trace
    distributed_tracing.finish_span(trace_id, "data_processing")
    distributed_tracing.add_trace_tag(trace_id, "user_id", "test_user")
    distributed_tracing.finish_trace(trace_id, "completed")
    
    print("✅ Monitoring system demonstration complete")
    
    # Show some statistics
    await asyncio.sleep(5)  # Let metrics collect
    
    cpu_stats = metrics_collector.get_metric_statistics("cpu_usage")
    print(f"📈 CPU Usage Stats: {cpu_stats}")
    
    trace_summary = distributed_tracing.get_trace_summary()
    print(f"🔍 Trace Summary: {trace_summary}")
    
    # Clean shutdown
    await alert_manager.stop_monitoring()
    await metrics_collector.stop_collection()
    
    print("🛑 Monitoring system shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())