"""
Unit tests for enhanced error handling and recovery functionality
"""
import pytest
import time
from unittest.mock import Mock, patch, call
from requests.exceptions import RequestException, ConnectionError, Timeout
from github import GithubException, RateLimitExceededException

import sys
sys.path.append('/root/repo/src')


class TestRetryDecorator:
    """Test cases for retry decorator functionality"""
    
    def test_retry_success_on_first_attempt(self):
        """Test successful execution on first attempt"""
        from error_handler import retry_on_failure
        
        @retry_on_failure(max_attempts=3, delay=0.1)
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_retry_success_after_failures(self):
        """Test successful execution after initial failures"""
        from error_handler import retry_on_failure
        
        call_count = 0
        
        @retry_on_failure(max_attempts=3, delay=0.1)
        def function_that_fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary connection error")
            return "success"
        
        result = function_that_fails_twice()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhausts_all_attempts(self):
        """Test retry behavior when all attempts are exhausted"""
        from error_handler import retry_on_failure
        
        call_count = 0
        
        @retry_on_failure(max_attempts=3, delay=0.1)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent connection error")
        
        with pytest.raises(ConnectionError):
            always_failing_function()
        
        assert call_count == 3
    
    def test_retry_with_custom_exceptions(self):
        """Test retry with custom exception types"""
        from error_handler import retry_on_failure
        
        call_count = 0
        
        @retry_on_failure(max_attempts=2, delay=0.1, 
                         exceptions=(ValueError, TypeError))
        def function_with_value_error():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return "success"
        
        result = function_with_value_error()
        assert result == "success"
        assert call_count == 2
    
    def test_retry_ignores_non_specified_exceptions(self):
        """Test that retry doesn't catch non-specified exceptions"""
        from error_handler import retry_on_failure
        
        @retry_on_failure(max_attempts=3, delay=0.1, 
                         exceptions=(ConnectionError,))
        def function_with_different_error():
            raise ValueError("This should not be retried")
        
        with pytest.raises(ValueError):
            function_with_different_error()
    
    def test_retry_with_exponential_backoff(self):
        """Test retry with exponential backoff"""
        from error_handler import retry_on_failure
        
        call_times = []
        
        @retry_on_failure(max_attempts=3, delay=0.1, backoff_multiplier=2.0)
        def function_tracking_timing():
            call_times.append(time.time())
            raise ConnectionError("Always fails")
        
        with pytest.raises(ConnectionError):
            function_tracking_timing()
        
        # Should have 3 attempts
        assert len(call_times) == 3
        
        # Check that delays are increasing (allowing for some timing variance)
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            assert delay1 >= 0.09  # ~0.1 seconds with some tolerance
        
        if len(call_times) >= 3:
            delay2 = call_times[2] - call_times[1]
            assert delay2 >= 0.19  # ~0.2 seconds with some tolerance


class TestGracefulErrorHandler:
    """Test cases for graceful error handling"""
    
    def test_github_api_error_handling(self):
        """Test handling of GitHub API errors"""
        from error_handler import handle_github_api_error
        
        # Test rate limit error
        rate_limit_error = RateLimitExceededException(status=403, data={'message': 'Rate limit exceeded'})
        
        with patch('error_handler.logger') as mock_logger:
            result = handle_github_api_error(rate_limit_error, "test operation")
            
            assert result is False
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "rate limit" in warning_call.lower()
    
    def test_network_error_handling(self):
        """Test handling of network-related errors"""
        from error_handler import handle_network_error
        
        # Test connection error
        conn_error = ConnectionError("Connection failed")
        
        with patch('error_handler.logger') as mock_logger:
            result = handle_network_error(conn_error, "GitHub API call")
            
            assert result is False
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args[0][0]
            assert "network error" in error_call.lower()
    
    def test_general_exception_handling(self):
        """Test handling of general exceptions"""
        from error_handler import handle_general_error
        
        test_error = ValueError("Invalid input")
        
        with patch('error_handler.logger') as mock_logger:
            result = handle_general_error(test_error, "test operation", {"key": "value"})
            
            assert result is False
            mock_logger.error.assert_called()
            # Check that context is included in error logging
            error_call = mock_logger.error.call_args[0][0]
            assert "test operation" in error_call


class TestErrorRecovery:
    """Test cases for error recovery mechanisms"""
    
    def test_github_api_recovery_with_retry(self):
        """Test GitHub API operations with retry and recovery"""
        from error_handler import with_error_recovery
        
        call_count = 0
        
        @with_error_recovery(operation_name="test_github_operation")
        def github_operation_with_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary network error")
            return {"status": "success"}
        
        result = github_operation_with_retry()
        assert result == {"status": "success"}
        assert call_count == 2
    
    def test_fallback_mechanism(self):
        """Test fallback mechanism when primary operation fails"""
        from error_handler import with_fallback
        
        def primary_operation():
            raise ConnectionError("Primary operation failed")
        
        def fallback_operation():
            return "fallback_result"
        
        result = with_fallback(primary_operation, fallback_operation, "test operation")
        assert result == "fallback_result"
    
    def test_fallback_with_primary_success(self):
        """Test that fallback is not used when primary operation succeeds"""
        from error_handler import with_fallback
        
        def primary_operation():
            return "primary_result"
        
        def fallback_operation():
            return "fallback_result"
        
        result = with_fallback(primary_operation, fallback_operation, "test operation")
        assert result == "primary_result"
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern for preventing cascading failures"""
        from error_handler import CircuitBreaker
        
        # Create circuit breaker with low thresholds for testing
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Operation failed")
        
        # First failure - should execute
        with pytest.raises(ConnectionError):
            breaker.call(failing_operation)
        
        # Second failure - should execute and open circuit
        with pytest.raises(ConnectionError):
            breaker.call(failing_operation)
        
        # Third attempt - circuit is open, should raise CircuitBreakerError
        from error_handler import CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            breaker.call(failing_operation)
        
        # Verify the third call didn't actually execute the function
        assert call_count == 2


class TestDetailedErrorReporting:
    """Test cases for detailed error reporting"""
    
    def test_error_context_collection(self):
        """Test collection of error context information"""
        from error_handler import collect_error_context
        
        context = collect_error_context(
            operation="test_operation",
            module="test_module",
            additional_data={"repo": "test/repo", "issue_number": 123}
        )
        
        assert context["operation"] == "test_operation"
        assert context["module"] == "test_module"
        assert context["repo"] == "test/repo"
        assert context["issue_number"] == 123
        assert "timestamp" in context
        assert "python_version" in context
    
    def test_error_summary_generation(self):
        """Test generation of error summaries"""
        from error_handler import generate_error_summary
        
        test_error = ValueError("Test error message")
        context = {"operation": "test_op", "module": "test_module"}
        
        summary = generate_error_summary(test_error, context)
        
        assert summary["error_type"] == "ValueError"
        assert summary["error_message"] == "Test error message"
        assert summary["operation"] == "test_op"
        assert summary["module"] == "test_module"
        assert "stack_trace" in summary
    
    def test_error_metric_tracking(self):
        """Test tracking of error metrics"""
        from error_handler import ErrorMetrics
        
        metrics = ErrorMetrics()
        
        # Record some errors
        metrics.record_error("github_api", "ConnectionError")
        metrics.record_error("github_api", "ConnectionError") 
        metrics.record_error("github_api", "TimeoutError")
        metrics.record_error("task_analyzer", "ValueError")
        
        stats = metrics.get_error_statistics()
        
        assert stats["total_errors"] == 4
        assert stats["errors_by_module"]["github_api"] == 3
        assert stats["errors_by_module"]["task_analyzer"] == 1
        assert stats["errors_by_type"]["ConnectionError"] == 2
        assert stats["errors_by_type"]["TimeoutError"] == 1
        assert stats["errors_by_type"]["ValueError"] == 1