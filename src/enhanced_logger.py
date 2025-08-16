"""
Enhanced Logging System for Generation 2 Robustness

This module provides comprehensive logging with:
- Structured logging with JSON format
- Request tracing and correlation IDs
- Performance metrics integration
- Security event logging
- Monitoring and alerting capabilities
"""

import logging
import json
import time
import uuid
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import os
import sys
from pathlib import Path

from src.logger import get_logger as base_get_logger


@dataclass
class LogContext:
    """Enhanced logging context"""
    correlation_id: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    extra_fields: Optional[Dict[str, Any]] = None


@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_type: str  # 'authentication', 'authorization', 'data_access', 'suspicious_activity'
    severity: str    # 'low', 'medium', 'high', 'critical'
    description: str
    user_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None  # 'success', 'failure', 'blocked'


@dataclass
class PerformanceEvent:
    """Performance event for monitoring"""
    operation: str
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    resource_usage: Optional[Dict[str, float]] = None


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log structure
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': getattr(record, 'module', record.name),
            'function': getattr(record, 'funcName', None),
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_entry['correlation_id'] = record.correlation_id
        
        # Add operation context
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        
        # Add user context
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        # Add request context
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        # Add session context
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        
        # Add performance metrics
        if hasattr(record, 'duration_ms'):
            log_entry['performance'] = {
                'duration_ms': record.duration_ms,
                'operation': getattr(record, 'perf_operation', None)
            }
        
        # Add security context
        if hasattr(record, 'security_event'):
            log_entry['security'] = asdict(record.security_event)
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key.startswith('extra_'):
                field_name = key[6:]  # Remove 'extra_' prefix
                log_entry[field_name] = value
        
        return json.dumps(log_entry, default=str)


class EnhancedLogger:
    """Enhanced logger with structured logging and context management"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.context_store = threading.local()
        self.loggers: Dict[str, logging.Logger] = {}
        self.security_logger = None
        self.performance_logger = None
        
        # Initialize logging configuration
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup enhanced logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create structured formatter
        structured_formatter = StructuredFormatter()
        
        # Console handler with structured output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(structured_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.FileHandler(log_dir / 'application.json')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(structured_formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file handler
        error_handler = logging.FileHandler(log_dir / 'errors.json')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(structured_formatter)
        root_logger.addHandler(error_handler)
        
        # Setup specialized loggers
        self._setup_security_logger()
        self._setup_performance_logger()
    
    def _setup_security_logger(self):
        """Setup dedicated security event logger"""
        self.security_logger = logging.getLogger('security')
        self.security_logger.setLevel(logging.INFO)
        
        # Security events file handler
        security_handler = logging.FileHandler(Path('logs') / 'security.json')
        security_handler.setLevel(logging.INFO)
        security_handler.setFormatter(StructuredFormatter())
        self.security_logger.addHandler(security_handler)
        self.security_logger.propagate = False  # Don't propagate to root logger
    
    def _setup_performance_logger(self):
        """Setup dedicated performance logger"""
        self.performance_logger = logging.getLogger('performance')
        self.performance_logger.setLevel(logging.INFO)
        
        # Performance metrics file handler
        perf_handler = logging.FileHandler(Path('logs') / 'performance.json')
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(StructuredFormatter())
        self.performance_logger.addHandler(perf_handler)
        self.performance_logger.propagate = False  # Don't propagate to root logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        return self.loggers[name]
    
    def set_context(self, context: LogContext):
        """Set logging context for current thread"""
        self.context_store.context = context
    
    def get_context(self) -> Optional[LogContext]:
        """Get current logging context"""
        return getattr(self.context_store, 'context', None)
    
    def clear_context(self):
        """Clear current logging context"""
        if hasattr(self.context_store, 'context'):
            delattr(self.context_store, 'context')
    
    @contextmanager
    def log_context(self, context: LogContext):
        """Context manager for temporary logging context"""
        old_context = self.get_context()
        self.set_context(context)
        try:
            yield
        finally:
            if old_context:
                self.set_context(old_context)
            else:
                self.clear_context()
    
    def log_with_context(self, logger_name: str, level: int, message: str, **kwargs):
        """Log message with current context"""
        logger = self.get_logger(logger_name)
        
        # Get current context
        context = self.get_context()
        
        # Create log record
        record = logger.makeRecord(
            logger.name, level, __file__, 0, message, (), None
        )
        
        # Add context information
        if context:
            record.correlation_id = context.correlation_id
            record.operation = context.operation
            if context.user_id:
                record.user_id = context.user_id
            if context.session_id:
                record.session_id = context.session_id
            if context.request_id:
                record.request_id = context.request_id
            if context.module:
                record.module = context.module
            if context.function:
                record.function = context.function
            if context.extra_fields:
                for key, value in context.extra_fields.items():
                    setattr(record, f'extra_{key}', value)
        
        # Add additional fields
        for key, value in kwargs.items():
            if key.startswith('extra_'):
                setattr(record, key, value)
            else:
                setattr(record, f'extra_{key}', value)
        
        logger.handle(record)
    
    def log_security_event(self, event: SecurityEvent):
        """Log security event"""
        if self.security_logger:
            record = self.security_logger.makeRecord(
                self.security_logger.name,
                logging.WARNING if event.severity in ['high', 'critical'] else logging.INFO,
                __file__, 0,
                f"Security event: {event.description}",
                (), None
            )
            record.security_event = event
            
            # Add context if available
            context = self.get_context()
            if context:
                record.correlation_id = context.correlation_id
                record.operation = context.operation
            
            self.security_logger.handle(record)
    
    def log_performance_event(self, event: PerformanceEvent):
        """Log performance event"""
        if self.performance_logger:
            record = self.performance_logger.makeRecord(
                self.performance_logger.name,
                logging.WARNING if not event.success else logging.INFO,
                __file__, 0,
                f"Performance: {event.operation} took {event.duration_ms:.2f}ms",
                (), None
            )
            record.duration_ms = event.duration_ms
            record.perf_operation = event.operation
            record.success = event.success
            if event.error_type:
                record.error_type = event.error_type
            if event.resource_usage:
                for key, value in event.resource_usage.items():
                    setattr(record, f'resource_{key}', value)
            
            # Add context if available
            context = self.get_context()
            if context:
                record.correlation_id = context.correlation_id
            
            self.performance_logger.handle(record)


class LoggingDecorator:
    """Decorator for automatic operation logging"""
    
    def __init__(self, operation: str, log_args: bool = False, log_result: bool = False):
        self.operation = operation
        self.log_args = log_args
        self.log_result = log_result
        self.enhanced_logger = EnhancedLogger()
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Generate correlation ID for this operation
            correlation_id = str(uuid.uuid4())
            
            # Create logging context
            context = LogContext(
                correlation_id=correlation_id,
                operation=self.operation,
                module=func.__module__,
                function=func.__name__
            )
            
            logger = self.enhanced_logger.get_logger(func.__module__)
            
            with self.enhanced_logger.log_context(context):
                start_time = time.time()
                
                # Log operation start
                extra_fields = {}
                if self.log_args:
                    extra_fields['args_count'] = len(args)
                    extra_fields['kwargs_keys'] = list(kwargs.keys())
                
                self.enhanced_logger.log_with_context(
                    func.__module__, logging.INFO,
                    f"Starting operation: {self.operation}",
                    **extra_fields
                )
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful completion
                    duration_ms = (time.time() - start_time) * 1000
                    
                    extra_fields = {'duration_ms': duration_ms}
                    if self.log_result and result is not None:
                        extra_fields['result_type'] = type(result).__name__
                        if hasattr(result, '__len__'):
                            extra_fields['result_length'] = len(result)
                    
                    self.enhanced_logger.log_with_context(
                        func.__module__, logging.INFO,
                        f"Operation completed: {self.operation}",
                        **extra_fields
                    )
                    
                    # Log performance event
                    perf_event = PerformanceEvent(
                        operation=self.operation,
                        duration_ms=duration_ms,
                        success=True
                    )
                    self.enhanced_logger.log_performance_event(perf_event)
                    
                    return result
                    
                except Exception as e:
                    # Log error
                    duration_ms = (time.time() - start_time) * 1000
                    
                    self.enhanced_logger.log_with_context(
                        func.__module__, logging.ERROR,
                        f"Operation failed: {self.operation}",
                        error_type=e.__class__.__name__,
                        error_message=str(e),
                        duration_ms=duration_ms
                    )
                    
                    # Log performance event
                    perf_event = PerformanceEvent(
                        operation=self.operation,
                        duration_ms=duration_ms,
                        success=False,
                        error_type=e.__class__.__name__
                    )
                    self.enhanced_logger.log_performance_event(perf_event)
                    
                    raise
        
        return wrapper


# Global instance
_enhanced_logger = EnhancedLogger()


def get_enhanced_logger(name: str = None) -> logging.Logger:
    """Get enhanced logger instance"""
    if name:
        return _enhanced_logger.get_logger(name)
    else:
        return _enhanced_logger.get_logger(__name__)


def set_log_context(context: LogContext):
    """Set logging context for current thread"""
    _enhanced_logger.set_context(context)


def get_log_context() -> Optional[LogContext]:
    """Get current logging context"""
    return _enhanced_logger.get_context()


def clear_log_context():
    """Clear current logging context"""
    _enhanced_logger.clear_context()


def log_context(correlation_id: str = None, operation: str = None, **kwargs):
    """Context manager for logging context"""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    context = LogContext(
        correlation_id=correlation_id,
        operation=operation or "unknown",
        **kwargs
    )
    
    return _enhanced_logger.log_context(context)


def log_security_event(
    event_type: str,
    severity: str,
    description: str,
    **kwargs
):
    """Log security event"""
    event = SecurityEvent(
        event_type=event_type,
        severity=severity,
        description=description,
        **kwargs
    )
    _enhanced_logger.log_security_event(event)


def log_performance(operation: str, duration_ms: float, success: bool = True, **kwargs):
    """Log performance event"""
    event = PerformanceEvent(
        operation=operation,
        duration_ms=duration_ms,
        success=success,
        **kwargs
    )
    _enhanced_logger.log_performance_event(event)


# Convenience decorators
def log_operation(operation: str, log_args: bool = False, log_result: bool = False):
    """Decorator for automatic operation logging"""
    return LoggingDecorator(operation, log_args, log_result)


def log_github_operation(operation: str):
    """Decorator for GitHub operations"""
    return LoggingDecorator(f"github_{operation}", log_args=True)


def log_database_operation(operation: str):
    """Decorator for database operations"""
    return LoggingDecorator(f"db_{operation}", log_args=True)


# Performance monitoring context manager
@contextmanager
def monitor_performance(operation: str, **kwargs):
    """Context manager for performance monitoring"""
    start_time = time.time()
    success = True
    error_type = None
    
    try:
        yield
    except Exception as e:
        success = False
        error_type = e.__class__.__name__
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        log_performance(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            **kwargs
        )