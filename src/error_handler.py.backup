"""
Enhanced error handling and recovery module for Claude Manager Service

This module provides comprehensive error handling including:
- Retry mechanisms with exponential backoff
- Graceful error recovery
- Circuit breaker pattern
- Detailed error reporting and metrics
"""
import time
import sys
import traceback
import functools
from typing import Dict, Any, Callable, Optional, Type, Union, List
from datetime import datetime, timezone
from collections import defaultdict
from logger import get_logger

logger = get_logger(__name__)


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Failures exceeded threshold, calls fail fast
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold (int): Number of failures before opening circuit
            recovery_timeout (float): Time to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """
        Call function through circuit breaker
        
        Args:
            func (Callable): Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.debug("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_multiplier: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function calls on failure with exponential backoff
    
    Args:
        max_attempts (int): Maximum number of retry attempts
        delay (float): Initial delay between retries in seconds
        backoff_multiplier (float): Multiplier for exponential backoff
        exceptions (tuple): Tuple of exception types to retry on
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:  # Don't sleep on last attempt
                        logger.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}/{max_attempts}: {e}. "
                            f"Retrying in {current_delay:.2f} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_multiplier
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts. "
                            f"Final error: {e}"
                        )
            
            # Re-raise the last exception if all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


def handle_github_api_error(error: Exception, operation: str) -> bool:
    """
    Handle GitHub API specific errors with appropriate logging and recovery
    
    Args:
        error (Exception): The GitHub API error
        operation (str): Description of the operation that failed
        
    Returns:
        bool: False to indicate operation should be considered failed
    """
    from github import GithubException, RateLimitExceededException
    
    if isinstance(error, RateLimitExceededException):
        headers = getattr(error, 'headers', None)
        if headers and hasattr(headers, 'get'):
            reset_time = headers.get('X-RateLimit-Reset', 'unknown')
        else:
            reset_time = 'unknown'
        logger.warning(
            f"GitHub API rate limit exceeded during {operation}. "
            f"Rate limit resets at: {reset_time}. "
            f"Consider implementing rate limit handling or increasing delays."
        )
        return False
    elif isinstance(error, GithubException):
        # Handle different data formats (dict, string, or None)
        if hasattr(error, 'data') and error.data:
            if isinstance(error.data, dict):
                message = error.data.get('message', str(error))
            else:
                message = str(error.data)
        else:
            message = str(error)
        
        logger.error(
            f"GitHub API error during {operation}: {message} "
            f"(Status: {getattr(error, 'status', 'unknown')})"
        )
        return False
    else:
        logger.error(f"Unexpected error during GitHub API {operation}: {error}")
        return False


def handle_network_error(error: Exception, operation: str) -> bool:
    """
    Handle network-related errors
    
    Args:
        error (Exception): The network error
        operation (str): Description of the operation that failed
        
    Returns:
        bool: False to indicate operation should be considered failed
    """
    from requests.exceptions import ConnectionError, Timeout, RequestException
    
    if isinstance(error, (ConnectionError, Timeout)):
        logger.error(
            f"Network error during {operation}: {error}. "
            f"Check internet connection and GitHub API availability."
        )
    elif isinstance(error, RequestException):
        logger.error(f"HTTP request error during {operation}: {error}")
    else:
        logger.error(f"Unexpected network error during {operation}: {error}")
    
    return False


def handle_general_error(error: Exception, operation: str, context: Dict[str, Any] = None) -> bool:
    """
    Handle general errors with context logging
    
    Args:
        error (Exception): The error that occurred
        operation (str): Description of the operation that failed
        context (Dict[str, Any]): Additional context information
        
    Returns:
        bool: False to indicate operation should be considered failed
    """
    context_str = ""
    if context:
        context_str = f" Context: {context}"
    
    logger.error(
        f"Error during {operation}: {error}{context_str}",
        exc_info=True
    )
    return False


def with_error_recovery(
    operation_name: str,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_multiplier: float = 1.5
):
    """
    Decorator that combines retry logic with comprehensive error handling
    
    Args:
        operation_name (str): Name of the operation for logging
        max_attempts (int): Maximum retry attempts
        delay (float): Initial delay between retries
        backoff_multiplier (float): Backoff multiplier for delays
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @retry_on_failure(
            max_attempts=max_attempts,
            delay=delay,
            backoff_multiplier=backoff_multiplier,
            exceptions=(Exception,)
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log detailed error information
                error_context = collect_error_context(
                    operation=operation_name,
                    module=func.__module__,
                    function=func.__name__
                )
                
                # Record error metrics
                if hasattr(wrapper, '_error_metrics'):
                    wrapper._error_metrics.record_error(func.__module__, type(e).__name__)
                
                # Handle specific error types
                from github import GithubException
                if isinstance(e, GithubException):
                    handle_github_api_error(e, operation_name)
                elif isinstance(e, (ConnectionError, TimeoutError)):
                    handle_network_error(e, operation_name)
                else:
                    handle_general_error(e, operation_name, error_context)
                
                raise
        
        # Attach error metrics to function
        wrapper._error_metrics = ErrorMetrics()
        return wrapper
    return decorator


def with_fallback(
    primary_func: Callable,
    fallback_func: Callable,
    operation_name: str,
    *args, **kwargs
) -> Any:
    """
    Execute primary function with fallback on failure
    
    Args:
        primary_func (Callable): Primary function to try
        fallback_func (Callable): Fallback function if primary fails
        operation_name (str): Name of operation for logging
        *args: Arguments to pass to functions
        **kwargs: Keyword arguments to pass to functions
        
    Returns:
        Result from primary or fallback function
    """
    try:
        logger.debug(f"Attempting primary operation: {operation_name}")
        return primary_func(*args, **kwargs)
    except Exception as e:
        logger.warning(
            f"Primary operation {operation_name} failed: {e}. "
            f"Attempting fallback..."
        )
        try:
            result = fallback_func(*args, **kwargs)
            logger.info(f"Fallback operation succeeded for {operation_name}")
            return result
        except Exception as fallback_error:
            logger.error(
                f"Both primary and fallback operations failed for {operation_name}. "
                f"Primary error: {e}, Fallback error: {fallback_error}"
            )
            raise fallback_error


def collect_error_context(operation: str, module: str, additional_data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
    """
    Collect context information for error reporting
    
    Args:
        operation (str): Name of the operation
        module (str): Module where error occurred
        additional_data (Dict[str, Any]): Additional context data
        **kwargs: Additional keyword arguments
        
    Returns:
        Dict[str, Any]: Context information
    """
    context = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "module": module,
        "python_version": sys.version,
        **kwargs
    }
    
    if additional_data:
        context.update(additional_data)
    
    return context


def generate_error_summary(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive error summary
    
    Args:
        error (Exception): The error that occurred
        context (Dict[str, Any]): Context information
        
    Returns:
        Dict[str, Any]: Error summary
    """
    return {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc(),
        **context
    }


class ErrorMetrics:
    """
    Track and report error metrics for monitoring and debugging
    """
    
    def __init__(self):
        """Initialize error metrics tracking"""
        self.errors_by_module = defaultdict(int)
        self.errors_by_type = defaultdict(int)
        self.total_errors = 0
        self.first_error_time = None
        self.last_error_time = None
    
    def record_error(self, module: str, error_type: str):
        """
        Record an error occurrence
        
        Args:
            module (str): Module where error occurred
            error_type (str): Type of error
        """
        self.errors_by_module[module] += 1
        self.errors_by_type[error_type] += 1
        self.total_errors += 1
        
        current_time = datetime.now(timezone.utc)
        if self.first_error_time is None:
            self.first_error_time = current_time
        self.last_error_time = current_time
        
        logger.debug(f"Recorded error: {error_type} in {module}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error statistics
        
        Returns:
            Dict[str, Any]: Error statistics
        """
        return {
            "total_errors": self.total_errors,
            "errors_by_module": dict(self.errors_by_module),
            "errors_by_type": dict(self.errors_by_type),
            "first_error_time": self.first_error_time.isoformat() if self.first_error_time else None,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None
        }
    
    def reset_metrics(self):
        """Reset all error metrics"""
        self.errors_by_module.clear()
        self.errors_by_type.clear()
        self.total_errors = 0
        self.first_error_time = None
        self.last_error_time = None
        logger.debug("Error metrics reset")


# Global error metrics instance
_global_error_metrics = ErrorMetrics()


def get_global_error_metrics() -> ErrorMetrics:
    """
    Get the global error metrics instance
    
    Returns:
        ErrorMetrics: Global error metrics
    """
    return _global_error_metrics


# Utility functions for common error handling patterns

def safe_github_operation(operation_func: Callable, operation_name: str, *args, **kwargs) -> Optional[Any]:
    """
    Safely execute a GitHub operation with comprehensive error handling
    
    Args:
        operation_func (Callable): GitHub operation function
        operation_name (str): Name of operation for logging
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Optional[Any]: Operation result or None if failed
    """
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        from github import GithubException
        if isinstance(e, GithubException):
            handle_github_api_error(e, operation_name)
        else:
            handle_general_error(e, operation_name)
        
        # Record error in global metrics
        _global_error_metrics.record_error("github_api", type(e).__name__)
        return None


def safe_file_operation(operation_func: Callable, operation_name: str, *args, **kwargs) -> Optional[Any]:
    """
    Safely execute a file operation with error handling
    
    Args:
        operation_func (Callable): File operation function
        operation_name (str): Name of operation for logging
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Optional[Any]: Operation result or None if failed
    """
    try:
        return operation_func(*args, **kwargs)
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.error(f"File operation {operation_name} failed: {e}")
        _global_error_metrics.record_error("file_operations", type(e).__name__)
        return None
    except Exception as e:
        handle_general_error(e, operation_name)
        _global_error_metrics.record_error("file_operations", type(e).__name__)
        return None


# Example usage and testing
if __name__ == "__main__":
    # Test retry decorator
    @retry_on_failure(max_attempts=3, delay=0.1)
    def test_retry_function():
        print("Testing retry functionality...")
        raise ConnectionError("Test connection error")
    
    # Test circuit breaker
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
    
    def test_circuit_breaker():
        print("Testing circuit breaker...")
        raise ValueError("Test error")
    
    # Test error metrics
    metrics = ErrorMetrics()
    metrics.record_error("test_module", "TestError")
    print(f"Error statistics: {metrics.get_error_statistics()}")
    
    print("Error handling module test completed")