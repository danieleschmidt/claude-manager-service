#!/usr/bin/env python3
"""
Enhanced Error Handling System - Generation 2
Comprehensive error tracking, analysis, and recovery mechanisms
"""

import asyncio
import json
import logging
import time
import traceback
import functools
import inspect
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
import threading
from contextlib import contextmanager, asynccontextmanager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('error_handling.log')
    ]
)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    BUSINESS_LOGIC = "business_logic"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"

class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAK = "circuit_break"
    ESCALATE = "escalate"
    IGNORE = "ignore"

@dataclass
class ErrorContext:
    """Error context information"""
    function_name: str
    module_name: str
    line_number: int
    arguments: Dict[str, Any]
    local_variables: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    request_id: Optional[str] = None
    user_id: Optional[str] = None

@dataclass
class ErrorRecord:
    """Comprehensive error record"""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    stack_trace: str
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_success: bool = False
    occurrences: int = 1
    first_occurrence: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_occurrence: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryAction:
    """Recovery action definition"""
    strategy: RecoveryStrategy
    action: Callable
    max_attempts: int = 3
    delay: float = 1.0
    conditions: Optional[Callable] = None

class ErrorAnalyzer:
    """Advanced error analysis and pattern detection"""
    
    def __init__(self):
        self.error_patterns: Dict[str, List[str]] = {}
        self.correlation_rules: List[Dict[str, Any]] = []
        self.trend_analysis: Dict[str, List[float]] = {}
        
    def analyze_error_pattern(self, error_records: List[ErrorRecord]) -> Dict[str, Any]:
        """Analyze error patterns and trends"""
        analysis = {
            "total_errors": len(error_records),
            "error_by_category": {},
            "error_by_severity": {},
            "trending_errors": {},
            "correlation_detected": [],
            "recovery_success_rate": 0.0
        }
        
        # Category analysis
        for record in error_records:
            category = record.category.value
            analysis["error_by_category"][category] = analysis["error_by_category"].get(category, 0) + 1
            
            severity = record.severity.value
            analysis["error_by_severity"][severity] = analysis["error_by_severity"].get(severity, 0) + 1
        
        # Recovery success rate
        recovery_attempts = sum(1 for r in error_records if r.recovery_attempted)
        recovery_successes = sum(1 for r in error_records if r.recovery_success)
        
        if recovery_attempts > 0:
            analysis["recovery_success_rate"] = recovery_successes / recovery_attempts
        
        # Trend analysis
        for record in error_records:
            error_key = f"{record.category.value}_{record.error_type}"
            if error_key not in self.trend_analysis:
                self.trend_analysis[error_key] = []
            self.trend_analysis[error_key].append(record.last_occurrence.timestamp())
        
        return analysis
    
    def detect_cascading_failures(self, error_records: List[ErrorRecord]) -> List[Dict[str, Any]]:
        """Detect cascading failure patterns"""
        cascading_failures = []
        
        # Group errors by time window (5 minutes)
        time_windows = {}
        for record in error_records:
            window = int(record.last_occurrence.timestamp() / 300) * 300
            if window not in time_windows:
                time_windows[window] = []
            time_windows[window].append(record)
        
        # Look for patterns in time windows with multiple errors
        for window, records in time_windows.items():
            if len(records) > 3:  # Threshold for potential cascade
                categories = [r.category.value for r in records]
                if len(set(categories)) > 2:  # Multiple different categories
                    cascading_failures.append({
                        "timestamp": window,
                        "error_count": len(records),
                        "categories_affected": list(set(categories)),
                        "severity_breakdown": {
                            s.value: sum(1 for r in records if r.severity == s)
                            for s in ErrorSeverity
                        }
                    })
        
        return cascading_failures

class EnhancedErrorHandler:
    """
    Comprehensive error handling system with recovery mechanisms
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.error_records: Dict[str, ErrorRecord] = {}
        self.recovery_actions: Dict[ErrorCategory, List[RecoveryAction]] = {}
        self.analyzer = ErrorAnalyzer()
        self.lock = threading.Lock()
        
        # Metrics
        self.total_errors = 0
        self.recovered_errors = 0
        self.critical_errors = 0
        
        # Initialize recovery actions
        self._initialize_recovery_actions()
        
        logger.info("Enhanced Error Handler initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "max_error_records": 10000,
            "error_retention_hours": 72,
            "enable_auto_recovery": True,
            "enable_pattern_detection": True,
            "alert_thresholds": {
                "error_rate_per_hour": 100,
                "critical_errors_per_hour": 5,
                "recovery_failure_rate": 0.5
            },
            "recovery_settings": {
                "max_retry_attempts": 3,
                "base_retry_delay": 1.0,
                "exponential_backoff": True,
                "circuit_breaker_threshold": 5
            }
        }
    
    def _initialize_recovery_actions(self):
        """Initialize recovery actions for different error categories"""
        self.recovery_actions = {
            ErrorCategory.NETWORK: [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    action=self._retry_with_backoff,
                    max_attempts=3,
                    delay=2.0
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.CIRCUIT_BREAK,
                    action=self._activate_circuit_breaker,
                    max_attempts=1
                )
            ],
            ErrorCategory.DATABASE: [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    action=self._retry_with_backoff,
                    max_attempts=2,
                    delay=5.0
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.FALLBACK,
                    action=self._database_fallback,
                    max_attempts=1
                )
            ],
            ErrorCategory.FILESYSTEM: [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    action=self._retry_with_backoff,
                    max_attempts=2
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    action=self._filesystem_degradation,
                    max_attempts=1
                )
            ],
            ErrorCategory.VALIDATION: [
                RecoveryAction(
                    strategy=RecoveryStrategy.ESCALATE,
                    action=self._escalate_validation_error,
                    max_attempts=1
                )
            ],
            ErrorCategory.PERFORMANCE: [
                RecoveryAction(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    action=self._performance_degradation,
                    max_attempts=1
                )
            ]
        }
    
    def error_handler(self, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     category: ErrorCategory = ErrorCategory.SYSTEM,
                     recovery_strategy: Optional[RecoveryStrategy] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Decorator for comprehensive error handling"""
        
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                return self._async_error_wrapper(func, severity, category, recovery_strategy, metadata)
            else:
                return self._sync_error_wrapper(func, severity, category, recovery_strategy, metadata)
        
        return decorator
    
    def _sync_error_wrapper(self, func: Callable, severity: ErrorSeverity,
                           category: ErrorCategory, recovery_strategy: Optional[RecoveryStrategy],
                           metadata: Optional[Dict[str, Any]]):
        """Synchronous error wrapper"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Create error context
                context = self._create_error_context(func, args, kwargs, execution_time)
                
                # Record error
                error_record = self._record_error(e, severity, category, context, metadata)
                
                # Attempt recovery if enabled
                if self.config.get("enable_auto_recovery", True):
                    recovery_result = self._attempt_recovery(error_record, func, args, kwargs)
                    if recovery_result is not None:
                        return recovery_result
                
                # Re-raise if no recovery
                raise e
        
        return wrapper
    
    def _async_error_wrapper(self, func: Callable, severity: ErrorSeverity,
                            category: ErrorCategory, recovery_strategy: Optional[RecoveryStrategy],
                            metadata: Optional[Dict[str, Any]]):
        """Asynchronous error wrapper"""
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                return await func(*args, **kwargs)
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Create error context
                context = self._create_error_context(func, args, kwargs, execution_time)
                
                # Record error
                error_record = self._record_error(e, severity, category, context, metadata)
                
                # Attempt recovery if enabled
                if self.config.get("enable_auto_recovery", True):
                    recovery_result = await self._attempt_async_recovery(error_record, func, args, kwargs)
                    if recovery_result is not None:
                        return recovery_result
                
                # Re-raise if no recovery
                raise e
        
        return wrapper
    
    def _create_error_context(self, func: Callable, args: tuple, kwargs: dict,
                             execution_time: float) -> ErrorContext:
        """Create detailed error context"""
        
        frame = inspect.currentframe()
        try:
            # Get calling frame
            caller_frame = frame.f_back.f_back
            
            return ErrorContext(
                function_name=func.__name__,
                module_name=func.__module__,
                line_number=caller_frame.f_lineno if caller_frame else 0,
                arguments={
                    "args": self._sanitize_args(args),
                    "kwargs": self._sanitize_kwargs(kwargs)
                },
                local_variables=self._sanitize_locals(caller_frame.f_locals if caller_frame else {}),
                execution_time=execution_time,
                timestamp=datetime.now(timezone.utc)
            )
        finally:
            del frame
    
    def _sanitize_args(self, args: tuple) -> List[Any]:
        """Sanitize arguments for logging"""
        sanitized = []
        for arg in args:
            try:
                # Convert to string and truncate if too long
                arg_str = str(arg)
                if len(arg_str) > 200:
                    arg_str = arg_str[:200] + "..."
                sanitized.append(arg_str)
            except:
                sanitized.append("<unprintable>")
        return sanitized
    
    def _sanitize_kwargs(self, kwargs: dict) -> Dict[str, Any]:
        """Sanitize keyword arguments for logging"""
        sanitized = {}
        for key, value in kwargs.items():
            try:
                # Skip sensitive keys
                if any(sensitive in key.lower() for sensitive in ['password', 'token', 'secret', 'key']):
                    sanitized[key] = "<redacted>"
                else:
                    value_str = str(value)
                    if len(value_str) > 200:
                        value_str = value_str[:200] + "..."
                    sanitized[key] = value_str
            except:
                sanitized[key] = "<unprintable>"
        return sanitized
    
    def _sanitize_locals(self, local_vars: dict) -> Dict[str, Any]:
        """Sanitize local variables for logging"""
        sanitized = {}
        # Only include basic types to avoid circular references
        for key, value in local_vars.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                try:
                    sanitized[key] = str(value)[:100]  # Limit length
                except:
                    sanitized[key] = "<unprintable>"
        return sanitized
    
    def _record_error(self, error: Exception, severity: ErrorSeverity,
                     category: ErrorCategory, context: ErrorContext,
                     metadata: Optional[Dict[str, Any]]) -> ErrorRecord:
        """Record error with comprehensive details"""
        
        error_id = f"{category.value}_{error.__class__.__name__}_{abs(hash(str(error)))}"
        current_time = datetime.now(timezone.utc)
        
        with self.lock:
            if error_id in self.error_records:
                # Update existing record
                record = self.error_records[error_id]
                record.occurrences += 1
                record.last_occurrence = current_time
            else:
                # Create new record
                record = ErrorRecord(
                    error_id=error_id,
                    error_type=error.__class__.__name__,
                    error_message=str(error),
                    severity=severity,
                    category=category,
                    context=context,
                    stack_trace=traceback.format_exc(),
                    metadata=metadata or {}
                )
                self.error_records[error_id] = record
            
            # Update metrics
            self.total_errors += 1
            if severity == ErrorSeverity.CRITICAL:
                self.critical_errors += 1
            
            # Clean up old records if needed
            if len(self.error_records) > self.config.get("max_error_records", 10000):
                self._cleanup_old_records()
        
        # Log error
        self._log_error(record)
        
        return record
    
    def _log_error(self, record: ErrorRecord):
        """Log error with appropriate level"""
        log_message = (
            f"Error in {record.context.function_name}: {record.error_message} "
            f"(Severity: {record.severity.value}, Category: {record.category.value}, "
            f"Occurrences: {record.occurrences})"
        )
        
        if record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif record.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif record.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _cleanup_old_records(self):
        """Clean up old error records"""
        retention_hours = self.config.get("error_retention_hours", 72)
        cutoff_time = datetime.now(timezone.utc).timestamp() - (retention_hours * 3600)
        
        old_records = [
            error_id for error_id, record in self.error_records.items()
            if record.last_occurrence.timestamp() < cutoff_time
        ]
        
        for error_id in old_records:
            del self.error_records[error_id]
        
        logger.info(f"Cleaned up {len(old_records)} old error records")
    
    def _attempt_recovery(self, error_record: ErrorRecord, func: Callable,
                         args: tuple, kwargs: dict) -> Optional[Any]:
        """Attempt synchronous error recovery"""
        
        recovery_actions = self.recovery_actions.get(error_record.category, [])
        
        for action in recovery_actions:
            try:
                logger.info(f"Attempting recovery using {action.strategy.value} for {error_record.error_id}")
                
                result = action.action(error_record, func, args, kwargs)
                
                if result is not None:
                    error_record.recovery_attempted = True
                    error_record.recovery_strategy = action.strategy
                    error_record.recovery_success = True
                    self.recovered_errors += 1
                    
                    logger.info(f"Recovery successful for {error_record.error_id}")
                    return result
                    
            except Exception as recovery_error:
                logger.warning(f"Recovery attempt failed: {recovery_error}")
        
        error_record.recovery_attempted = True
        error_record.recovery_success = False
        
        return None
    
    async def _attempt_async_recovery(self, error_record: ErrorRecord, func: Callable,
                                    args: tuple, kwargs: dict) -> Optional[Any]:
        """Attempt asynchronous error recovery"""
        
        recovery_actions = self.recovery_actions.get(error_record.category, [])
        
        for action in recovery_actions:
            try:
                logger.info(f"Attempting async recovery using {action.strategy.value} for {error_record.error_id}")
                
                if asyncio.iscoroutinefunction(action.action):
                    result = await action.action(error_record, func, args, kwargs)
                else:
                    result = action.action(error_record, func, args, kwargs)
                
                if result is not None:
                    error_record.recovery_attempted = True
                    error_record.recovery_strategy = action.strategy
                    error_record.recovery_success = True
                    self.recovered_errors += 1
                    
                    logger.info(f"Async recovery successful for {error_record.error_id}")
                    return result
                    
            except Exception as recovery_error:
                logger.warning(f"Async recovery attempt failed: {recovery_error}")
        
        error_record.recovery_attempted = True
        error_record.recovery_success = False
        
        return None
    
    # Recovery action implementations
    
    def _retry_with_backoff(self, error_record: ErrorRecord, func: Callable,
                           args: tuple, kwargs: dict) -> Optional[Any]:
        """Retry with exponential backoff"""
        max_attempts = self.config.get("recovery_settings", {}).get("max_retry_attempts", 3)
        base_delay = self.config.get("recovery_settings", {}).get("base_retry_delay", 1.0)
        
        for attempt in range(max_attempts):
            try:
                delay = base_delay * (2 ** attempt) if self.config.get("recovery_settings", {}).get("exponential_backoff", True) else base_delay
                time.sleep(delay)
                
                logger.info(f"Retry attempt {attempt + 1}/{max_attempts} for {error_record.error_id}")
                return func(*args, **kwargs)
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.warning(f"All retry attempts failed for {error_record.error_id}")
                    break
        
        return None
    
    def _activate_circuit_breaker(self, error_record: ErrorRecord, func: Callable,
                                 args: tuple, kwargs: dict) -> Optional[Any]:
        """Activate circuit breaker"""
        logger.info(f"Activating circuit breaker for {error_record.error_id}")
        # Circuit breaker logic would be implemented here
        return None
    
    def _database_fallback(self, error_record: ErrorRecord, func: Callable,
                          args: tuple, kwargs: dict) -> Optional[Any]:
        """Database fallback strategy"""
        logger.info(f"Using database fallback for {error_record.error_id}")
        # Database fallback logic would be implemented here
        return {"fallback": True, "data": None}
    
    def _filesystem_degradation(self, error_record: ErrorRecord, func: Callable,
                               args: tuple, kwargs: dict) -> Optional[Any]:
        """Filesystem graceful degradation"""
        logger.info(f"Using filesystem degradation for {error_record.error_id}")
        # Return limited functionality
        return {"degraded": True, "limited_functionality": True}
    
    def _escalate_validation_error(self, error_record: ErrorRecord, func: Callable,
                                  args: tuple, kwargs: dict) -> Optional[Any]:
        """Escalate validation error"""
        logger.critical(f"Escalating validation error {error_record.error_id}")
        # Escalation logic would be implemented here
        return None
    
    def _performance_degradation(self, error_record: ErrorRecord, func: Callable,
                                args: tuple, kwargs: dict) -> Optional[Any]:
        """Performance graceful degradation"""
        logger.info(f"Using performance degradation for {error_record.error_id}")
        # Return cached or simplified result
        return {"performance_degraded": True, "cached_result": True}
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        with self.lock:
            records = list(self.error_records.values())
        
        if not records:
            return {"total_errors": 0, "message": "No errors recorded"}
        
        analysis = self.analyzer.analyze_error_pattern(records)
        cascading_failures = self.analyzer.detect_cascading_failures(records)
        
        return {
            "total_errors": self.total_errors,
            "recovered_errors": self.recovered_errors,
            "critical_errors": self.critical_errors,
            "recovery_rate": self.recovered_errors / max(self.total_errors, 1),
            "unique_error_types": len(records),
            "analysis": analysis,
            "cascading_failures": cascading_failures,
            "top_errors": sorted(records, key=lambda r: r.occurrences, reverse=True)[:5]
        }

# Example usage and testing
async def test_enhanced_error_handler():
    """Test the enhanced error handler"""
    handler = EnhancedErrorHandler()
    
    @handler.error_handler(severity=ErrorSeverity.HIGH, category=ErrorCategory.NETWORK)
    async def failing_network_function():
        """Function that simulates network failure"""
        raise ConnectionError("Network connection failed")
    
    @handler.error_handler(severity=ErrorSeverity.MEDIUM, category=ErrorCategory.VALIDATION)
    def failing_validation_function(data: str):
        """Function that simulates validation failure"""
        if not data:
            raise ValueError("Data cannot be empty")
        return f"Processed: {data}"
    
    print("Testing Enhanced Error Handler")
    print("-" * 40)
    
    # Test network error (with recovery attempt)
    try:
        await failing_network_function()
    except ConnectionError as e:
        print(f"Network error handled: {e}")
    
    # Test validation error
    try:
        failing_validation_function("")
    except ValueError as e:
        print(f"Validation error handled: {e}")
    
    # Test successful call
    result = failing_validation_function("test data")
    print(f"Successful call: {result}")
    
    # Get statistics
    stats = handler.get_error_statistics()
    print(f"\nError Statistics:")
    for key, value in stats.items():
        if key not in ['analysis', 'cascading_failures', 'top_errors']:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_error_handler())