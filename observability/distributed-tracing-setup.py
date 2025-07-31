"""
Distributed Tracing Setup for Claude Code Manager
Implements OpenTelemetry instrumentation for comprehensive observability
"""

import os
from typing import Dict, Any, Optional
import logging
from contextlib import contextmanager

from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.asyncio import AsyncIOInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.propagators.composite import CompositeHTTPPropagator
from opentelemetry.resource import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.semconv.trace import SpanAttributes


class AdvancedTracingSetup:
    """Advanced distributed tracing configuration for production environments"""
    
    def __init__(self, service_name: str = "claude-code-manager"):
        self.service_name = service_name
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.version = os.getenv("SERVICE_VERSION", "1.0.0")
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            "jaeger_endpoint": os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
            "otlp_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            "sampling_rate": float(os.getenv("TRACE_SAMPLING_RATE", "0.1")),
            "enable_logging": os.getenv("ENABLE_TRACE_LOGGING", "false").lower() == "true",
            "export_batch_size": int(os.getenv("TRACE_BATCH_SIZE", "512")),
            "export_timeout": int(os.getenv("TRACE_EXPORT_TIMEOUT", "30")),
        }
    
    def initialize(self) -> None:
        """Initialize complete observability stack"""
        try:
            self._setup_resource()
            self._setup_tracing()
            self._setup_metrics()
            self._setup_propagators()
            self._instrument_libraries()
            self.logger.info(f"Observability stack initialized for {self.service_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize observability: {e}")
            raise
    
    def _setup_resource(self) -> Resource:
        """Configure service resource with comprehensive metadata"""
        self.resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.SERVICE_VERSION: self.version,
            ResourceAttributes.SERVICE_NAMESPACE: "terragon-labs",
            ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("HOSTNAME", "unknown"),
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.environment,
            ResourceAttributes.CONTAINER_NAME: os.getenv("CONTAINER_NAME", ""),
            ResourceAttributes.K8S_POD_NAME: os.getenv("K8S_POD_NAME", ""),
            ResourceAttributes.K8S_NAMESPACE_NAME: os.getenv("K8S_NAMESPACE", ""),
            ResourceAttributes.HOST_NAME: os.getenv("HOST_NAME", ""),
            ResourceAttributes.CLOUD_PROVIDER: os.getenv("CLOUD_PROVIDER", ""),
            ResourceAttributes.CLOUD_REGION: os.getenv("CLOUD_REGION", ""),
            "application.team": "terragon-labs",
            "application.owner": "platform-team",
            "cost.center": "engineering",
        })
        return self.resource
    
    def _setup_tracing(self) -> None:
        """Configure distributed tracing with multiple exporters"""
        # Configure sampling strategy
        sampler = ParentBased(
            root=TraceIdRatioBased(self.config["sampling_rate"])
        )
        
        # Initialize tracer provider
        tracer_provider = TracerProvider(
            resource=self.resource,
            sampler=sampler
        )
        
        # Add span processors with different exporters
        self._add_span_processors(tracer_provider)
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Get tracer for application use
        self.tracer = trace.get_tracer(__name__)
    
    def _add_span_processors(self, tracer_provider: TracerProvider) -> None:
        """Add multiple span processors for different backends"""
        
        # Jaeger exporter for trace visualization
        if self.config["jaeger_endpoint"]:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
                collector_endpoint=self.config["jaeger_endpoint"],
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    jaeger_exporter,
                    max_export_batch_size=self.config["export_batch_size"],
                    export_timeout_millis=self.config["export_timeout"] * 1000,
                )
            )
        
        # OTLP exporter for external systems
        if self.config["otlp_endpoint"]:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config["otlp_endpoint"],
                headers={"x-api-key": os.getenv("OTEL_API_KEY", "")},
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        
        # Console exporter for development
        if self.config["enable_logging"]:
            from opentelemetry.exporter.console import ConsoleSpanExporter
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(
                SimpleSpanProcessor(console_exporter)
            )
    
    def _setup_metrics(self) -> None:
        """Configure metrics collection and export"""
        # Prometheus metrics reader
        prometheus_reader = PrometheusMetricReader(port=8889)
        
        # OTLP metrics reader
        otlp_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=self.config["otlp_endpoint"],
                headers={"x-api-key": os.getenv("OTEL_API_KEY", "")},
            ),
            export_interval_millis=15000,  # 15 seconds
        )
        
        # Initialize meter provider
        meter_provider = MeterProvider(
            resource=self.resource,
            metric_readers=[prometheus_reader, otlp_reader],
        )
        
        # Set global meter provider
        metrics.set_meter_provider(meter_provider)
        
        # Get meter for application use
        self.meter = metrics.get_meter(__name__)
        
        # Create custom metrics
        self._create_custom_metrics()
    
    def _create_custom_metrics(self) -> None:
        """Create application-specific metrics"""
        # Counters
        self.request_counter = self.meter.create_counter(
            name="http_requests_total",
            description="Total HTTP requests",
            unit="1",
        )
        
        self.task_counter = self.meter.create_counter(
            name="tasks_processed_total",
            description="Total tasks processed",
            unit="1",
        )
        
        self.error_counter = self.meter.create_counter(
            name="errors_total",
            description="Total errors encountered",
            unit="1",
        )
        
        # Histograms
        self.request_duration = self.meter.create_histogram(
            name="http_request_duration_seconds",
            description="HTTP request duration",
            unit="s",
        )
        
        self.task_duration = self.meter.create_histogram(
            name="task_processing_duration_seconds",
            description="Task processing duration",
            unit="s",
        )
        
        # Gauges
        self.active_connections = self.meter.create_up_down_counter(
            name="active_connections",
            description="Number of active connections",
            unit="1",
        )
        
        self.queue_size = self.meter.create_up_down_counter(
            name="task_queue_size",
            description="Current task queue size",
            unit="1",
        )
    
    def _setup_propagators(self) -> None:
        """Configure trace context propagation"""
        set_global_textmap(
            CompositeHTTPPropagator([
                JaegerPropagator(),
                B3MultiFormat(),
            ])
        )
    
    def _instrument_libraries(self) -> None:
        """Auto-instrument common libraries"""
        # Flask instrumentation
        FlaskInstrumentor().instrument()
        
        # HTTP requests instrumentation
        RequestsInstrumentor().instrument()
        
        # AsyncIO instrumentation
        AsyncIOInstrumentor().instrument()
        
        # AIOHTTP client instrumentation
        AioHttpClientInstrumentor().instrument()
        
        # Database instrumentation
        try:
            Psycopg2Instrumentor().instrument()
        except ImportError:
            self.logger.warning("psycopg2 not available, skipping database instrumentation")
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for manual tracing"""
        with self.tracer.start_as_current_span(operation_name) as span:
            try:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                yield span
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
    
    def add_baggage(self, key: str, value: str) -> None:
        """Add baggage to current trace context"""
        baggage.set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage from current trace context"""
        return baggage.get_baggage(key)
    
    def create_child_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> trace.Span:
        """Create a child span of current span"""
        span = self.tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        return span
    
    def record_custom_metric(self, metric_name: str, value: float, attributes: Optional[Dict[str, Any]] = None):
        """Record a custom metric value"""
        metric_attributes = attributes or {}
        
        if metric_name == "request_count":
            self.request_counter.add(1, metric_attributes)
        elif metric_name == "request_duration":
            self.request_duration.record(value, metric_attributes)
        elif metric_name == "task_count":
            self.task_counter.add(1, metric_attributes)
        elif metric_name == "task_duration":
            self.task_duration.record(value, metric_attributes)
        elif metric_name == "error_count":
            self.error_counter.add(1, metric_attributes)
    
    def shutdown(self) -> None:
        """Gracefully shutdown tracing"""
        try:
            # Shutdown tracer provider
            if hasattr(trace.get_tracer_provider(), 'shutdown'):
                trace.get_tracer_provider().shutdown()
            
            # Shutdown meter provider
            if hasattr(metrics.get_meter_provider(), 'shutdown'):
                metrics.get_meter_provider().shutdown()
                
            self.logger.info("Observability stack shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Decorator for automatic tracing
def traced_function(operation_name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to automatically trace function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            span_attributes = attributes or {}
            
            # Add function metadata
            span_attributes.update({
                "function.name": func.__name__,
                "function.module": func.__module__,
            })
            
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name) as span:
                try:
                    for key, value in span_attributes.items():
                        span.set_attribute(key, value)
                    
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator


# Global tracing setup instance
tracing_setup = AdvancedTracingSetup()


def initialize_observability(service_name: str = "claude-code-manager") -> AdvancedTracingSetup:
    """Initialize observability stack"""
    global tracing_setup
    tracing_setup = AdvancedTracingSetup(service_name)
    tracing_setup.initialize()
    return tracing_setup


def get_tracer() -> trace.Tracer:
    """Get the configured tracer"""
    return trace.get_tracer(__name__)


def get_meter() -> metrics.Meter:
    """Get the configured meter"""
    return metrics.get_meter(__name__)


# Example usage in application code
if __name__ == "__main__":
    # Initialize observability
    setup = initialize_observability()
    
    # Example traced function
    @traced_function("example_operation", {"component": "example"})
    def example_function():
        import time
        time.sleep(0.1)
        return "Hello, World!"
    
    # Example manual tracing
    with setup.trace_operation("manual_operation", {"user_id": "123"}) as span:
        span.set_attribute("operation.type", "manual")
        print("Performing manual operation")
        setup.record_custom_metric("task_count", 1, {"task_type": "example"})
    
    # Graceful shutdown
    import atexit
    atexit.register(setup.shutdown)