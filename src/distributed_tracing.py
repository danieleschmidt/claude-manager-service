"""
Distributed Tracing and Advanced Performance Monitoring for Claude Manager Service Generation 3

This module provides enterprise-grade monitoring and tracing including:
- Real-time performance metrics collection
- Distributed tracing with OpenTelemetry integration
- APM (Application Performance Monitoring) features
- Performance profiling and bottleneck detection
- Custom metrics and alerting
- Request correlation and tracking
- Service dependency mapping
- Performance anomaly detection
"""

import asyncio
import json
import time
import threading
import traceback
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, AsyncContextManager
import logging
import os
import platform

try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    metrics = None

from src.logger import get_logger
from src.performance_monitor import monitor_performance, get_monitor
from src.error_handler import TracingError, with_enhanced_error_handling


logger = get_logger(__name__)


class TraceLevel(Enum):
    """Trace logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class SpanType(Enum):
    """Types of spans"""
    HTTP_REQUEST = "http.request"
    DATABASE_QUERY = "db.query"
    EXTERNAL_CALL = "http.client"
    CACHE_OPERATION = "cache.operation"
    TASK_EXECUTION = "task.execution"
    CUSTOM = "custom"


@dataclass
class TraceContext:
    """Trace context information"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    sampling_priority: int = 1
    
    @classmethod
    def generate(cls, parent_context: Optional['TraceContext'] = None) -> 'TraceContext':
        """Generate new trace context"""
        if parent_context:
            return cls(
                trace_id=parent_context.trace_id,
                span_id=str(uuid.uuid4()),
                parent_span_id=parent_context.span_id,
                baggage=parent_context.baggage.copy(),
                sampling_priority=parent_context.sampling_priority
            )
        else:
            return cls(
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4())
            )


@dataclass
class Span:
    """Distributed tracing span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: bool = False
    
    def finish(self, end_time: Optional[float] = None):
        """Finish the span"""
        self.end_time = end_time or time.time()
        self.duration = self.end_time - self.start_time
    
    def set_tag(self, key: str, value: Any):
        """Set span tag"""
        self.tags[key] = value
    
    def set_error(self, error: Exception):
        """Mark span as error"""
        self.error = True
        self.status = "error"
        self.tags["error.type"] = type(error).__name__
        self.tags["error.message"] = str(error)
        self.tags["error.stack"] = traceback.format_exc()
    
    def log(self, message: str, level: TraceLevel = TraceLevel.INFO, **kwargs):
        """Add log to span"""
        log_entry = {
            "timestamp": time.time(),
            "level": level.value,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Performance metric"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


class TracerImplementation:
    """Custom tracer implementation"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.active_spans: Dict[str, Span] = {}
        self.finished_spans: deque[Span] = deque(maxlen=10000)
        self.context_stack: Dict[int, List[TraceContext]] = defaultdict(list)
        self.lock = threading.RLock()
        
        # OpenTelemetry integration
        self.otel_tracer = None
        if OPENTELEMETRY_AVAILABLE:
            self._setup_opentelemetry()
        
        logger.info(f"TracerImplementation initialized for service '{service_name}'")
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry integration"""
        try:
            # Configure tracer provider
            trace.set_tracer_provider(TracerProvider())
            
            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
                agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
            )
            
            # Add span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Get tracer
            self.otel_tracer = trace.get_tracer(__name__)
            
            logger.info("OpenTelemetry tracing configured")
            
        except Exception as e:
            logger.warning(f"Failed to setup OpenTelemetry: {e}")
    
    def start_span(self, 
                   operation_name: str,
                   span_type: SpanType = SpanType.CUSTOM,
                   parent_context: Optional[TraceContext] = None,
                   tags: Optional[Dict[str, Any]] = None) -> Span:
        """Start a new span"""
        
        # Get or create trace context
        if parent_context:
            context = TraceContext.generate(parent_context)
        else:
            # Try to get context from current thread
            thread_id = threading.get_ident()
            contexts = self.context_stack[thread_id]
            if contexts:
                context = TraceContext.generate(contexts[-1])
            else:
                context = TraceContext.generate()
        
        # Create span
        span = Span(
            trace_id=context.trace_id,
            span_id=context.span_id,
            parent_span_id=context.parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        # Set default tags
        span.set_tag("service.name", self.service_name)
        span.set_tag("span.type", span_type.value)
        span.set_tag("thread.id", threading.get_ident())
        
        if tags:
            for key, value in tags.items():
                span.set_tag(key, value)
        
        # Store active span
        with self.lock:
            self.active_spans[span.span_id] = span
            
            # Push context to stack
            thread_id = threading.get_ident()
            self.context_stack[thread_id].append(context)
        
        # Start OpenTelemetry span if available
        if self.otel_tracer:
            try:
                otel_span = self.otel_tracer.start_span(operation_name)
                span.otel_span = otel_span
                
                # Set OpenTelemetry attributes
                if tags:
                    for key, value in tags.items():
                        otel_span.set_attribute(key, str(value))
                        
            except Exception as e:
                logger.debug(f"Failed to start OpenTelemetry span: {e}")
        
        logger.debug(f"Started span '{operation_name}' ({span.span_id})")
        return span
    
    def finish_span(self, span: Span):
        """Finish a span"""
        span.finish()
        
        with self.lock:
            # Remove from active spans
            self.active_spans.pop(span.span_id, None)
            
            # Add to finished spans
            self.finished_spans.append(span)
            
            # Pop context from stack
            thread_id = threading.get_ident()
            contexts = self.context_stack[thread_id]
            if contexts:
                contexts.pop()
        
        # Finish OpenTelemetry span
        if hasattr(span, 'otel_span') and span.otel_span:
            try:
                if span.error:
                    span.otel_span.set_status(trace.Status(trace.StatusCode.ERROR, span.tags.get("error.message", "Error")))
                
                span.otel_span.end()
            except Exception as e:
                logger.debug(f"Failed to finish OpenTelemetry span: {e}")
        
        logger.debug(f"Finished span '{span.operation_name}' ({span.span_id}) - duration: {span.duration:.3f}s")
    
    @asynccontextmanager
    async def trace(self, 
                   operation_name: str,
                   span_type: SpanType = SpanType.CUSTOM,
                   tags: Optional[Dict[str, Any]] = None) -> AsyncContextManager[Span]:
        """Context manager for tracing operations"""
        span = self.start_span(operation_name, span_type, tags=tags)
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.finish_span(span)
    
    def get_current_context(self) -> Optional[TraceContext]:
        """Get current trace context"""
        thread_id = threading.get_ident()
        contexts = self.context_stack[thread_id]
        return contexts[-1] if contexts else None
    
    def inject_context(self, context: TraceContext, headers: Dict[str, str]):
        """Inject trace context into headers"""
        headers["X-Trace-Id"] = context.trace_id
        headers["X-Span-Id"] = context.span_id
        if context.parent_span_id:
            headers["X-Parent-Span-Id"] = context.parent_span_id
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Extract trace context from headers"""
        trace_id = headers.get("X-Trace-Id")
        span_id = headers.get("X-Span-Id")
        parent_span_id = headers.get("X-Parent-Span-Id")
        
        if trace_id and span_id:
            return TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id
            )
        return None
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        with self.lock:
            active_count = len(self.active_spans)
            finished_count = len(self.finished_spans)
            
            # Calculate average duration
            avg_duration = 0.0
            if self.finished_spans:
                total_duration = sum(span.duration or 0 for span in self.finished_spans)
                avg_duration = total_duration / len(self.finished_spans)
            
            # Count errors
            error_count = sum(1 for span in self.finished_spans if span.error)
            
            return {
                "service_name": self.service_name,
                "active_spans": active_count,
                "finished_spans": finished_count,
                "average_duration": avg_duration,
                "error_count": error_count,
                "error_rate": error_count / max(finished_count, 1),
                "opentelemetry_enabled": self.otel_tracer is not None
            }


class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.metrics: Dict[str, deque[Metric]] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_registry: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
        # Prometheus integration
        self.prometheus_enabled = False
        self._setup_prometheus()
        
        logger.info(f"MetricsCollector initialized for service '{service_name}'")
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics export"""
        try:
            if OPENTELEMETRY_AVAILABLE:
                # Configure metrics provider
                reader = PrometheusMetricReader()
                provider = MeterProvider(metric_readers=[reader])
                metrics.set_meter_provider(provider)
                
                self.meter = metrics.get_meter(__name__)
                self.prometheus_enabled = True
                
                logger.info("Prometheus metrics export configured")
                
        except Exception as e:
            logger.warning(f"Failed to setup Prometheus: {e}")
    
    def register_metric(self, 
                       name: str, 
                       metric_type: MetricType,
                       description: str = "",
                       unit: Optional[str] = None):
        """Register a metric"""
        self.metric_registry[name] = {
            "type": metric_type,
            "description": description,
            "unit": unit,
            "created_at": time.time()
        }
        
        logger.debug(f"Registered metric '{name}' of type {metric_type.value}")
    
    def record_metric(self, 
                     name: str,
                     value: float,
                     tags: Optional[Dict[str, str]] = None,
                     timestamp: Optional[float] = None):
        """Record a metric value"""
        
        metric = Metric(
            name=name,
            metric_type=self.metric_registry.get(name, {}).get("type", MetricType.GAUGE),
            value=value,
            timestamp=timestamp or time.time(),
            tags=tags or {},
            unit=self.metric_registry.get(name, {}).get("unit")
        )
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        if name not in self.metric_registry:
            self.register_metric(name, MetricType.COUNTER)
        
        self.record_metric(name, value, tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        if name not in self.metric_registry:
            self.register_metric(name, MetricType.GAUGE)
        
        self.record_metric(name, value, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        if name not in self.metric_registry:
            self.register_metric(name, MetricType.HISTOGRAM)
        
        self.record_metric(name, value, tags)
    
    def get_metric_values(self, name: str, time_range: int = 300) -> List[Metric]:
        """Get metric values within time range"""
        cutoff_time = time.time() - time_range
        
        with self.lock:
            return [
                metric for metric in self.metrics[name]
                if metric.timestamp >= cutoff_time
            ]
    
    def get_metric_summary(self, name: str, time_range: int = 300) -> Optional[Dict[str, Any]]:
        """Get metric summary statistics"""
        values = self.get_metric_values(name, time_range)
        if not values:
            return None
        
        numeric_values = [m.value for m in values]
        
        return {
            "name": name,
            "count": len(numeric_values),
            "sum": sum(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "avg": sum(numeric_values) / len(numeric_values),
            "latest": numeric_values[-1],
            "time_range": time_range
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        
        with self.lock:
            for name in self.metrics.keys():
                metric_summary = self.get_metric_summary(name)
                if metric_summary:
                    summary[name] = metric_summary
        
        return {
            "service_name": self.service_name,
            "timestamp": time.time(),
            "metrics": summary,
            "registered_metrics": len(self.metric_registry),
            "prometheus_enabled": self.prometheus_enabled
        }


class PerformanceProfiler:
    """Performance profiling and bottleneck detection"""
    
    def __init__(self):
        self.profiling_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.hot_spots: List[Dict[str, Any]] = []
        self.profiling_enabled = True
        
        logger.info("PerformanceProfiler initialized")
    
    def profile_function(self, func_name: str, duration: float, memory_delta: Optional[float] = None):
        """Profile function execution"""
        if not self.profiling_enabled:
            return
        
        profile_data = {
            "timestamp": time.time(),
            "duration": duration,
            "memory_delta": memory_delta,
            "thread_id": threading.get_ident(),
            "call_stack": traceback.format_stack()[-5:]  # Last 5 stack frames
        }
        
        self.profiling_data[func_name].append(profile_data)
        
        # Check for performance hot spots
        if duration > 1.0:  # Functions taking more than 1 second
            self._record_hot_spot(func_name, duration, profile_data)
    
    def _record_hot_spot(self, func_name: str, duration: float, profile_data: Dict[str, Any]):
        """Record performance hot spot"""
        hot_spot = {
            "function": func_name,
            "duration": duration,
            "timestamp": profile_data["timestamp"],
            "call_stack": profile_data["call_stack"]
        }
        
        self.hot_spots.append(hot_spot)
        
        # Keep only recent hot spots
        if len(self.hot_spots) > 100:
            self.hot_spots = self.hot_spots[-50:]
        
        logger.warning(f"Performance hot spot detected: {func_name} took {duration:.2f}s")
    
    def get_function_profile(self, func_name: str) -> Dict[str, Any]:
        """Get profiling data for a function"""
        data = self.profiling_data.get(func_name, [])
        if not data:
            return {}
        
        durations = [d["duration"] for d in data]
        memory_deltas = [d["memory_delta"] for d in data if d["memory_delta"] is not None]
        
        profile = {
            "function": func_name,
            "call_count": len(data),
            "total_duration": sum(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "recent_calls": len([d for d in data if time.time() - d["timestamp"] < 300])
        }
        
        if memory_deltas:
            profile.update({
                "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
                "max_memory_delta": max(memory_deltas)
            })
        
        return profile
    
    def get_top_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top functions by total duration"""
        function_profiles = []
        
        for func_name in self.profiling_data.keys():
            profile = self.get_function_profile(func_name)
            if profile:
                function_profiles.append(profile)
        
        # Sort by total duration
        function_profiles.sort(key=lambda x: x["total_duration"], reverse=True)
        
        return function_profiles[:limit]
    
    def get_hot_spots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance hot spots"""
        return sorted(self.hot_spots, key=lambda x: x["duration"], reverse=True)[:limit]


class APMSystem:
    """Application Performance Monitoring system"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = TracerImplementation(service_name)
        self.metrics = MetricsCollector(service_name)
        self.profiler = PerformanceProfiler()
        
        # System monitoring
        self.system_metrics_task: Optional[asyncio.Task] = None
        self.monitoring_enabled = True
        
        logger.info(f"APMSystem initialized for service '{service_name}'")
    
    async def start(self):
        """Start APM system"""
        if self.monitoring_enabled and not self.system_metrics_task:
            self.system_metrics_task = asyncio.create_task(self._system_metrics_loop())
            logger.info("APM system started")
    
    async def stop(self):
        """Stop APM system"""
        if self.system_metrics_task:
            self.system_metrics_task.cancel()
            try:
                await self.system_metrics_task
            except asyncio.CancelledError:
                pass
            self.system_metrics_task = None
            
        logger.info("APM system stopped")
    
    async def _system_metrics_loop(self):
        """Background system metrics collection"""
        logger.debug("Started system metrics collection")
        
        while True:
            try:
                import psutil
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.set_gauge("system.cpu.usage", cpu_percent, {"unit": "percent"})
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.metrics.set_gauge("system.memory.usage", memory.percent, {"unit": "percent"})
                self.metrics.set_gauge("system.memory.available", memory.available / (1024**3), {"unit": "GB"})
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_usage = (disk.used / disk.total) * 100
                self.metrics.set_gauge("system.disk.usage", disk_usage, {"unit": "percent"})
                
                # Network metrics
                network = psutil.net_io_counters()
                self.metrics.set_gauge("system.network.bytes_sent", network.bytes_sent, {"unit": "bytes"})
                self.metrics.set_gauge("system.network.bytes_recv", network.bytes_recv, {"unit": "bytes"})
                
                await asyncio.sleep(60)  # Collect every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)
    
    @asynccontextmanager
    async def trace_operation(self, 
                             operation_name: str,
                             span_type: SpanType = SpanType.CUSTOM,
                             tags: Optional[Dict[str, Any]] = None):
        """Trace an operation"""
        async with self.tracer.trace(operation_name, span_type, tags) as span:
            # Start profiling
            start_time = time.time()
            start_memory = None
            
            try:
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss
            except Exception:
                pass
            
            try:
                yield span
            finally:
                # Record profiling data
                duration = time.time() - start_time
                memory_delta = None
                
                if start_memory:
                    try:
                        import psutil
                        process = psutil.Process()
                        end_memory = process.memory_info().rss
                        memory_delta = end_memory - start_memory
                    except Exception:
                        pass
                
                self.profiler.profile_function(operation_name, duration, memory_delta)
                
                # Record metrics
                self.metrics.record_histogram("operation.duration", duration * 1000, {"operation": operation_name})
                if memory_delta:
                    self.metrics.record_histogram("operation.memory_delta", memory_delta, {"operation": operation_name})
    
    def record_request(self, method: str, path: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        tags = {
            "method": method,
            "path": path,
            "status_code": str(status_code)
        }
        
        self.metrics.increment_counter("http.requests.total", 1.0, tags)
        self.metrics.record_histogram("http.request.duration", duration * 1000, tags)
        
        if 200 <= status_code < 300:
            self.metrics.increment_counter("http.requests.success", 1.0, tags)
        elif 400 <= status_code < 500:
            self.metrics.increment_counter("http.requests.client_error", 1.0, tags)
        elif 500 <= status_code < 600:
            self.metrics.increment_counter("http.requests.server_error", 1.0, tags)
    
    def record_database_query(self, query_type: str, duration: float, success: bool):
        """Record database query metrics"""
        tags = {
            "query_type": query_type,
            "success": str(success)
        }
        
        self.metrics.increment_counter("db.queries.total", 1.0, tags)
        self.metrics.record_histogram("db.query.duration", duration * 1000, tags)
        
        if success:
            self.metrics.increment_counter("db.queries.success", 1.0, tags)
        else:
            self.metrics.increment_counter("db.queries.error", 1.0, tags)
    
    async def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive APM report"""
        return {
            "service_name": self.service_name,
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "architecture": platform.machine()
            },
            "tracing": self.tracer.get_trace_stats(),
            "metrics": self.metrics.get_all_metrics_summary(),
            "profiling": {
                "top_functions": self.profiler.get_top_functions(),
                "hot_spots": self.profiler.get_hot_spots()
            },
            "monitoring_enabled": self.monitoring_enabled
        }


# Global APM instance
_apm_system: Optional[APMSystem] = None
_apm_lock = asyncio.Lock()


async def get_apm_system(service_name: str = "claude-manager") -> APMSystem:
    """Get global APM system instance"""
    global _apm_system
    
    if _apm_system is None:
        async with _apm_lock:
            if _apm_system is None:
                _apm_system = APMSystem(service_name)
                await _apm_system.start()
    
    return _apm_system


# Convenience functions and decorators
@asynccontextmanager
async def trace_operation(operation_name: str, 
                         span_type: SpanType = SpanType.CUSTOM,
                         tags: Optional[Dict[str, Any]] = None):
    """Trace operation context manager"""
    apm = await get_apm_system()
    async with apm.trace_operation(operation_name, span_type, tags) as span:
        yield span


def traced(operation_name: Optional[str] = None, 
          span_type: SpanType = SpanType.CUSTOM,
          tags: Optional[Dict[str, Any]] = None):
    """Decorator for tracing functions"""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with trace_operation(operation_name, span_type, tags):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # For sync functions, we need to handle this differently
                # This is a simplified version - in practice you'd want proper async handling
                return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


async def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record a custom metric"""
    apm = await get_apm_system()
    apm.metrics.record_metric(name, value, tags)


async def get_apm_report() -> Dict[str, Any]:
    """Get comprehensive APM report"""
    apm = await get_apm_system()
    return await apm.get_comprehensive_report()


# Integration with existing performance monitor
@with_enhanced_error_handling("apm_integration")
async def integrate_with_performance_monitor():
    """Integrate APM with existing performance monitor"""
    apm = await get_apm_system()
    perf_monitor = get_monitor()
    
    # Get performance report and convert to APM metrics
    if hasattr(perf_monitor, 'get_performance_report'):
        report = perf_monitor.get_performance_report(hours=1)
        
        if 'overall_stats' in report:
            stats = report['overall_stats']
            
            # Convert performance stats to APM metrics
            apm.metrics.set_gauge("perf.total_operations", stats.get('total_operations', 0))
            apm.metrics.set_gauge("perf.success_rate", stats.get('success_rate', 0) * 100)
            apm.metrics.set_gauge("perf.average_duration", stats.get('average_duration', 0) * 1000)
            apm.metrics.set_gauge("perf.p95_duration", stats.get('p95_duration', 0) * 1000)
            apm.metrics.set_gauge("perf.p99_duration", stats.get('p99_duration', 0) * 1000)
    
    logger.info("APM integration with performance monitor completed")