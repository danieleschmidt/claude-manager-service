"""
Enhanced Error Handling System for Claude Manager Service

This module provides specific exception handling, rate limiting, circuit breaker patterns,
and enhanced error reporting to replace generic exception handling throughout the codebase.

Features:
- Specific exception types for different error categories
- Rate limiting for API operations
- Enhanced circuit breaker patterns
- Error context collection and metrics
- Structured error reporting
"""

import time
import json
import os
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import functools

from logger import get_logger


# Custom Exception Classes
class EnhancedError(Exception):
    """Base class for enhanced error handling"""
    def __init__(self, message: str, error_type: str = None, context: Dict = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()


class FileOperationError(EnhancedError):
    """Specific error for file operation failures"""
    def __init__(self, message: str, file_path: str, operation: str, original_error: Exception):
        super().__init__(
            f"File operation '{operation}' failed for {file_path}: {message}",
            original_error.__class__.__name__,
            {"file_path": file_path, "operation": operation}
        )
        self.file_path = file_path
        self.operation = operation
        self.original_error = original_error


class JsonParsingError(EnhancedError):
    """Specific error for JSON parsing failures"""
    def __init__(self, message: str, file_path: str, original_error: Exception):
        super().__init__(
            f"JSON parsing failed for {file_path}: {message}",
            original_error.__class__.__name__,
            {"file_path": file_path, "json_error": str(original_error)}
        )
        self.file_path = file_path
        self.original_error = original_error


class NetworkError(EnhancedError):
    """Specific error for network operation failures"""
    def __init__(self, message: str, operation: str, original_error: Exception):
        super().__init__(
            f"Network operation '{operation}' failed: {message}",
            original_error.__class__.__name__,
            {"operation": operation, "network_error": str(original_error)}
        )
        self.operation = operation
        self.original_error = original_error


class RateLimitError(EnhancedError):
    """Specific error for API rate limiting"""
    def __init__(self, message: str, operation: str, retry_after: Optional[int] = None):
        super().__init__(
            f"Rate limit exceeded for operation '{operation}': {message}",
            "RateLimitError",
            {"operation": operation, "retry_after": retry_after}
        )
        self.operation = operation
        self.retry_after = retry_after


class AuthenticationError(EnhancedError):
    """Specific error for authentication failures"""
    def __init__(self, message: str, operation: str):
        super().__init__(
            f"Authentication failed for operation '{operation}': {message}",
            "AuthenticationError",
            {"operation": operation}
        )
        self.operation = operation


class ErrorContext(Exception):
    """Enhanced exception with full context information"""
    def __init__(self, original_exception: Exception, context: Dict[str, Any]):
        self.original_exception = original_exception
        self.operation = context.get("operation", "unknown")
        self.parameters = context.get("parameters", {})
        self.timestamp = context.get("timestamp", time.time())
        self.module = context.get("module", "unknown")
        self.function = context.get("function", "unknown")
        
        message = f"Operation '{self.operation}' failed: {str(original_exception)}"
        super().__init__(message)


# Safe Operation Functions
def safe_file_read(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Safely read file with specific exception handling
    
    Args:
        file_path: Path to file to read
        encoding: File encoding
        
    Returns:
        File contents as string
        
    Raises:
        FileOperationError: For specific file operation failures
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except FileNotFoundError as e:
        raise FileOperationError(str(e), file_path, "read", e)
    except PermissionError as e:
        raise FileOperationError(str(e), file_path, "read", e)
    except UnicodeDecodeError as e:
        raise FileOperationError(f"Encoding error: {str(e)}", file_path, "read", e)
    except OSError as e:
        raise FileOperationError(f"OS error: {str(e)}", file_path, "read", e)


def safe_json_load(file_path: str) -> Dict[str, Any]:
    """
    Safely load JSON file with specific exception handling
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileOperationError: For file access issues
        JsonParsingError: For JSON parsing issues
    """
    try:
        content = safe_file_read(file_path)
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise JsonParsingError(str(e), file_path, e)
    except FileOperationError:
        raise  # Re-raise file operation errors


def safe_api_call(func: Callable, operation: str, *args, **kwargs) -> Any:
    """
    Safely execute API call with specific exception handling
    
    Args:
        func: Function to call
        operation: Name of operation for error context
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Function result
        
    Raises:
        NetworkError: For network-related failures
        RateLimitError: For rate limiting
        AuthenticationError: For auth failures
    """
    try:
        return func(*args, **kwargs)
    except ConnectionError as e:
        raise NetworkError(str(e), operation, e)
    except TimeoutError as e:
        raise NetworkError(f"Timeout: {str(e)}", operation, e)
    except Exception as e:
        # Check for rate limiting patterns
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            if e.response.status_code == 403:
                # Check for rate limit headers
                headers = getattr(e.response, 'headers', {})
                if 'X-RateLimit-Remaining' in headers and headers['X-RateLimit-Remaining'] == '0':
                    retry_after = int(headers.get('X-RateLimit-Reset', 3600)) - int(time.time())
                    raise RateLimitError(str(e), operation, max(retry_after, 60))
            elif e.response.status_code == 401:
                raise AuthenticationError(str(e), operation)
        
        # Default to network error for unhandled cases
        raise NetworkError(str(e), operation, e)


class RateLimiter:
    """
    Rate limiter for API operations to prevent hitting API limits
    """
    
    def __init__(self, max_requests: int = None, time_window: float = None):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in time window (configurable via RATE_LIMIT_MAX_REQUESTS env var)
            time_window: Time window in seconds (configurable via RATE_LIMIT_TIME_WINDOW env var)
        """
        # Use environment variables with validation, fallback to parameters or defaults
        if max_requests is not None or time_window is not None:
            # Use provided parameters
            self.max_requests = max_requests or int(os.getenv('RATE_LIMIT_MAX_REQUESTS', '5000'))
            self.time_window = time_window or float(os.getenv('RATE_LIMIT_TIME_WINDOW', '3600.0'))
        else:
            # Use validated configuration
            try:
                from config_env import get_env_config
                config = get_env_config()
                self.max_requests = config.rate_limit_max_requests
                self.time_window = config.rate_limit_time_window
            except ImportError:
                # Fallback to direct environment variable access
                self.max_requests = int(os.getenv('RATE_LIMIT_MAX_REQUESTS', '5000'))
                self.time_window = float(os.getenv('RATE_LIMIT_TIME_WINDOW', '3600.0'))
        self.request_history: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def can_proceed(self, key: str) -> bool:
        """
        Check if request can proceed without hitting rate limit
        
        Args:
            key: Identifier for rate limiting (e.g., 'github_api')
            
        Returns:
            True if request can proceed, False if rate limited
        """
        current_time = time.time()
        
        with self.lock:
            # Clean old entries
            history = self.request_history[key]
            cutoff_time = current_time - self.time_window
            
            while history and history[0] < cutoff_time:
                history.popleft()
            
            # Check if we can proceed
            if len(history) >= self.max_requests:
                self.logger.warning(f"Rate limit exceeded for key '{key}': {len(history)}/{self.max_requests}")
                return False
            
            # Record this request
            history.append(current_time)
            return True
    
    def get_remaining_quota(self, key: str) -> int:
        """Get remaining requests in current window"""
        with self.lock:
            current_time = time.time()
            history = self.request_history[key]
            cutoff_time = current_time - self.time_window
            
            # Clean old entries
            while history and history[0] < cutoff_time:
                history.popleft()
            
            return max(0, self.max_requests - len(history))


class OperationCircuitBreaker:
    """
    Enhanced circuit breaker with operation-specific state tracking
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 300.0):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.operation_states: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "failure_count": 0,
                "last_failure": None,
                "state": "closed"  # closed, open, half-open
            }
        )
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def can_proceed(self, module: str, operation: str) -> bool:
        """
        Check if operation can proceed based on circuit breaker state
        
        Args:
            module: Module name (e.g., 'github_api')
            operation: Operation name (e.g., 'create_issue')
            
        Returns:
            True if operation can proceed
        """
        key = f"{module}.{operation}"
        current_time = time.time()
        
        with self.lock:
            state = self.operation_states[key]
            
            # If circuit is closed, allow operation
            if state["state"] == "closed":
                return True
            
            # If circuit is open, check if recovery timeout has passed
            if state["state"] == "open":
                if (current_time - state["last_failure"]) > self.recovery_timeout:
                    state["state"] = "half-open"
                    self.logger.info(f"Circuit breaker for {key} moving to half-open state")
                    return True
                return False
            
            # If half-open, allow one test request
            if state["state"] == "half-open":
                return True
            
            return False
    
    def record_success(self, module: str, operation: str):
        """Record successful operation"""
        key = f"{module}.{operation}"
        
        with self.lock:
            state = self.operation_states[key]
            
            if state["state"] == "half-open":
                # Recovery successful, close circuit
                state["state"] = "closed"
                state["failure_count"] = 0
                self.logger.info(f"Circuit breaker for {key} recovered, moving to closed state")
            elif state["state"] == "closed":
                # Reset failure count on successful operation
                state["failure_count"] = max(0, state["failure_count"] - 1)
    
    def record_failure(self, module: str, operation: str):
        """Record failed operation"""
        key = f"{module}.{operation}"
        current_time = time.time()
        
        with self.lock:
            state = self.operation_states[key]
            state["failure_count"] += 1
            state["last_failure"] = current_time
            
            if state["failure_count"] >= self.failure_threshold:
                if state["state"] != "open":
                    state["state"] = "open"
                    self.logger.warning(
                        f"Circuit breaker for {key} opened after {state['failure_count']} failures"
                    )
            elif state["state"] == "half-open":
                # Half-open test failed, go back to open
                state["state"] = "open"
                self.logger.warning(f"Circuit breaker for {key} test failed, returning to open state")


class ErrorTracker:
    """
    Track and analyze error patterns for metrics and monitoring
    """
    
    def __init__(self):
        self.error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.recent_errors: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def record_error(self, module: str, function: str, error_type: str, message: str):
        """Record error occurrence for analysis"""
        key = f"{module}.{function}"
        
        with self.lock:
            self.error_counts[key][error_type] += 1
            self.recent_errors.append({
                "timestamp": time.time(),
                "module": module,
                "function": function,
                "error_type": error_type,
                "message": message
            })
    
    def get_error_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get error statistics by function and error type"""
        with self.lock:
            return dict(self.error_counts)
    
    def get_recent_errors(self, count: int = 50) -> List[Dict]:
        """Get recent errors for debugging"""
        with self.lock:
            return list(self.recent_errors)[-count:]


def collect_error_context(error: Exception, context: Dict[str, Any]) -> ErrorContext:
    """
    Collect full error context for enhanced error reporting
    
    Args:
        error: Original exception
        context: Additional context information
        
    Returns:
        ErrorContext with full information
    """
    return ErrorContext(error, context)


# Global instances
_rate_limiter = RateLimiter()
_circuit_breaker = OperationCircuitBreaker()
_error_tracker = ErrorTracker()


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    return _rate_limiter


def get_circuit_breaker() -> OperationCircuitBreaker:
    """Get global circuit breaker instance"""
    return _circuit_breaker


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance"""
    return _error_tracker


def with_enhanced_error_handling(
    operation: str,
    module: str = None,
    use_rate_limiter: bool = False,
    use_circuit_breaker: bool = False,
    rate_limit_key: str = None
):
    """
    Decorator for enhanced error handling with rate limiting and circuit breaker
    
    Args:
        operation: Operation name for error context
        module: Module name for circuit breaker
        use_rate_limiter: Whether to apply rate limiting
        use_circuit_breaker: Whether to use circuit breaker
        rate_limit_key: Custom key for rate limiting
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine module name
            func_module = module or func.__module__
            rate_key = rate_limit_key or func_module
            
            # Check rate limiting
            if use_rate_limiter:
                if not _rate_limiter.can_proceed(rate_key):
                    raise RateLimitError(f"Rate limit exceeded", operation)
            
            # Check circuit breaker
            if use_circuit_breaker:
                if not _circuit_breaker.can_proceed(func_module, func.__name__):
                    raise NetworkError(f"Circuit breaker open", operation, 
                                     Exception("Circuit breaker protection"))
            
            # Execute function with error handling
            context = {
                "operation": operation,
                "module": func_module,
                "function": func.__name__,
                "parameters": {"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                "timestamp": time.time()
            }
            
            try:
                result = func(*args, **kwargs)
                
                # Record success for circuit breaker
                if use_circuit_breaker:
                    _circuit_breaker.record_success(func_module, func.__name__)
                
                return result
                
            except Exception as e:
                # Record failure for circuit breaker
                if use_circuit_breaker:
                    _circuit_breaker.record_failure(func_module, func.__name__)
                
                # Record error for tracking
                _error_tracker.record_error(
                    func_module, func.__name__, 
                    e.__class__.__name__, str(e)
                )
                
                # Re-raise enhanced error if it's already enhanced
                if isinstance(e, EnhancedError):
                    raise
                
                # Otherwise, wrap in error context
                raise collect_error_context(e, context)
        
        return wrapper
    return decorator


# Convenience decorators for common patterns
def github_api_operation(operation: str):
    """Decorator for GitHub API operations with full protection"""
    return with_enhanced_error_handling(
        operation=operation,
        module="github_api",
        use_rate_limiter=True,
        use_circuit_breaker=True,
        rate_limit_key="github_api"
    )


def file_operation(operation: str):
    """Decorator for file operations with enhanced error handling"""
    return with_enhanced_error_handling(
        operation=operation,
        use_rate_limiter=False,
        use_circuit_breaker=False
    )


def network_operation(operation: str, service: str = "generic"):
    """Decorator for network operations with protection"""
    return with_enhanced_error_handling(
        operation=operation,
        use_rate_limiter=True,
        use_circuit_breaker=True,
        rate_limit_key=service
    )


# Backward compatibility functions
def with_error_recovery(operation: str):
    """Legacy decorator for error recovery - redirects to enhanced error handling"""
    return with_enhanced_error_handling(operation=operation)


def safe_github_operation(operation: str):
    """Legacy decorator for GitHub operations - redirects to github_api_operation"""
    return github_api_operation(operation)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Legacy retry decorator - provides basic retry functionality"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay * (attempt + 1))
                    else:
                        raise last_exception
            return None
        return wrapper
    return decorator


def handle_github_api_error(func: Callable) -> Callable:
    """Legacy GitHub API error handler"""
    return github_api_operation("github_api_call")(func)


def handle_network_error(func: Callable) -> Callable:
    """Legacy network error handler"""
    return network_operation("network_call")(func)


def handle_general_error(func: Callable) -> Callable:
    """Legacy general error handler"""
    return with_enhanced_error_handling("general_operation")(func)


def with_fallback(fallback_value=None):
    """Legacy fallback decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return fallback_value
        return wrapper
    return decorator


class CircuitBreakerError(Exception):
    """Legacy circuit breaker error"""
    pass


class CircuitBreaker:
    """Legacy circuit breaker - redirects to OperationCircuitBreaker"""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 300.0):
        self._breaker = OperationCircuitBreaker(failure_threshold, recovery_timeout)
        self.module = "legacy"
        self.operation = "circuit_breaker"
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if not self._breaker.can_proceed(self.module, self.operation):
            raise CircuitBreakerError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._breaker.record_success(self.module, self.operation)
            return result
        except Exception as e:
            self._breaker.record_failure(self.module, self.operation)
            raise


def generate_error_summary(errors: List[Dict]) -> str:
    """Generate error summary from error list"""
    if not errors:
        return "No errors recorded"
    
    error_types = defaultdict(int)
    for error in errors:
        error_types[error.get("error_type", "Unknown")] += 1
    
    summary = f"Total errors: {len(errors)}\n"
    for error_type, count in error_types.items():
        summary += f"  {error_type}: {count}\n"
    
    return summary


class ErrorMetrics:
    """Legacy error metrics - redirects to ErrorTracker"""
    def __init__(self):
        self._tracker = get_error_tracker()
    
    def record_error(self, module: str, function: str, error_type: str, message: str):
        """Record error occurrence"""
        self._tracker.record_error(module, function, error_type, message)
    
    def get_statistics(self):
        """Get error statistics"""
        return self._tracker.get_error_statistics()
    
    def get_recent_errors(self, count: int = 50):
        """Get recent errors"""
        return self._tracker.get_recent_errors(count)


class ErrorHandler:
    """Legacy error handler class for backward compatibility"""
    def __init__(self):
        self.error_tracker = get_error_tracker()
        self.logger = get_logger(__name__)
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None):
        """Handle error with context"""
        context = context or {}
        module = context.get("module", "unknown")
        function = context.get("function", "unknown")
        
        self.error_tracker.record_error(
            module, function, error.__class__.__name__, str(error)
        )
        
        self.logger.error(f"Error in {module}.{function}: {str(error)}")
        return collect_error_context(error, context)