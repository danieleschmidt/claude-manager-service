"""
Tests for enhanced error handling and security validation improvements

This module tests the new enhanced error handling system that replaces
generic exception handling with specific, actionable error handling patterns.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestSpecificExceptionHandling:
    """Test specific exception handling patterns"""
    
    def test_file_operation_specific_exceptions(self):
        """Test that file operations handle specific exceptions"""
        from enhanced_error_handler import (
            safe_file_read,
            safe_json_load,
            FileOperationError,
            JsonParsingError
        )
        
        # Test file not found
        with pytest.raises(FileOperationError) as exc_info:
            safe_file_read("/nonexistent/file.txt")
        
        assert exc_info.value.error_type == "FileNotFoundError"
        assert "/nonexistent/file.txt" in str(exc_info.value)
        
        # Test directory access (trying to read a directory as file)
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(FileOperationError) as exc_info:
                safe_file_read(temp_dir)  # Try to read directory as file
            
            # Should get IsADirectoryError or similar
            assert exc_info.value.file_path == temp_dir
            assert exc_info.value.operation == "read"
        finally:
            os.rmdir(temp_dir)
    
    def test_json_parsing_specific_exceptions(self):
        """Test specific JSON parsing error handling"""
        from enhanced_error_handler import safe_json_load, JsonParsingError
        
        # Test invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write("{ invalid json }")
            temp_path = temp_file.name
        
        try:
            with pytest.raises(JsonParsingError) as exc_info:
                safe_json_load(temp_path)
            
            assert exc_info.value.error_type == "JSONDecodeError"
            assert temp_path in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_network_operation_specific_exceptions(self):
        """Test specific network operation error handling"""
        from enhanced_error_handler import (
            safe_api_call,
            NetworkError,
            RateLimitError,
            AuthenticationError
        )
        
        # Mock function that simulates different network errors
        def mock_api_call():
            raise ConnectionError("Network unreachable")
        
        with pytest.raises(NetworkError) as exc_info:
            safe_api_call(mock_api_call, "test_operation")
        
        assert exc_info.value.error_type == "ConnectionError"
        assert exc_info.value.operation == "test_operation"
        
        # Test rate limit error
        def mock_rate_limit():
            from requests.exceptions import HTTPError
            error = HTTPError()
            error.response = Mock()
            error.response.status_code = 403
            error.response.headers = {"X-RateLimit-Remaining": "0"}
            raise error
        
        with pytest.raises(RateLimitError):
            safe_api_call(mock_rate_limit, "rate_limited_operation")


class TestSecurityValidationEnhancements:
    """Test enhanced security validation patterns"""
    
    def test_enhanced_token_validation(self):
        """Test improved token validation with specific patterns"""
        from enhanced_security import (
            validate_token_enhanced,
            TokenValidationError,
            InvalidTokenFormatError,
            ExpiredTokenError,
            WeakTokenError
        )
        
        # Test valid token patterns
        valid_github_token = "ghp_" + "a" * 36
        assert validate_token_enhanced(valid_github_token, "github") is True
        
        valid_github_app_token = "ghs_" + "a" * 36  
        assert validate_token_enhanced(valid_github_app_token, "github") is True
        
        # Test invalid format (weak token will be caught first)
        with pytest.raises(WeakTokenError):
            validate_token_enhanced("invalid_token", "github")
        
        # Test weak token (too short)
        with pytest.raises(WeakTokenError):
            validate_token_enhanced("ghp_short", "github")
        
        # Test potentially expired pattern (starts with old format)
        with pytest.raises(ExpiredTokenError):
            validate_token_enhanced("abc123def456", "github")  # Old 40-char hex
    
    def test_path_traversal_prevention(self):
        """Test path traversal attack prevention"""
        from enhanced_security import (
            safe_path_join,
            PathTraversalError,
            validate_safe_path
        )
        
        base_dir = "/safe/directory"
        
        # Test normal path
        safe_path = safe_path_join(base_dir, "subdir/file.txt")
        assert safe_path == "/safe/directory/subdir/file.txt"
        
        # Test path traversal attempts
        with pytest.raises(PathTraversalError):
            safe_path_join(base_dir, "../../../etc/passwd")
        
        with pytest.raises(PathTraversalError):
            safe_path_join(base_dir, "subdir/../../outside.txt")
        
        # Test absolute path injection
        with pytest.raises(PathTraversalError):
            safe_path_join(base_dir, "/absolute/path/file.txt")
    
    def test_input_sanitization_enhancement(self):
        """Test enhanced input sanitization"""
        from enhanced_security import (
            sanitize_input_enhanced,
            sanitize_repo_name,
            sanitize_issue_content_enhanced
        )
        
        # Test repository name sanitization
        clean_repo = sanitize_repo_name("user/repo-name_123")
        assert clean_repo == "user/repo-name_123"
        
        with pytest.raises(ValueError):
            sanitize_repo_name("../malicious/repo")
        
        with pytest.raises(ValueError):
            sanitize_repo_name("user/repo\x00null")
        
        # Test issue content sanitization
        malicious_content = "<script>alert('xss')</script>TODO: Fix this"
        clean_content = sanitize_issue_content_enhanced(malicious_content)
        assert "<script>" not in clean_content
        assert "TODO: Fix this" in clean_content


class TestRateLimitingSystem:
    """Test API rate limiting implementation"""
    
    def test_rate_limiter_creation(self):
        """Test rate limiter basic functionality"""
        from enhanced_error_handler import RateLimiter
        
        # Create rate limiter: 5 requests per 10 seconds
        limiter = RateLimiter(max_requests=5, time_window=10.0)
        
        # Should allow first 5 requests
        for i in range(5):
            assert limiter.can_proceed("test_key") is True
        
        # 6th request should be rate limited
        assert limiter.can_proceed("test_key") is False
    
    def test_rate_limiter_window_reset(self):
        """Test rate limiter time window reset"""
        from enhanced_error_handler import RateLimiter
        
        limiter = RateLimiter(max_requests=2, time_window=0.1)  # Very short window
        
        # Use up quota
        assert limiter.can_proceed("test_key") is True
        assert limiter.can_proceed("test_key") is True
        assert limiter.can_proceed("test_key") is False
        
        # Wait for window to reset
        time.sleep(0.2)
        
        # Should be able to proceed again
        assert limiter.can_proceed("test_key") is True
    
    def test_rate_limiter_per_key_isolation(self):
        """Test that rate limiting is isolated per key"""
        from enhanced_error_handler import RateLimiter
        
        limiter = RateLimiter(max_requests=1, time_window=10.0)
        
        # Different keys should have separate quotas
        assert limiter.can_proceed("key1") is True
        assert limiter.can_proceed("key2") is True
        
        # Both keys should now be exhausted
        assert limiter.can_proceed("key1") is False
        assert limiter.can_proceed("key2") is False


class TestEnhancedErrorReporting:
    """Test enhanced error reporting and context"""
    
    def test_error_context_collection(self):
        """Test that errors collect relevant context"""
        from enhanced_error_handler import ErrorContext, collect_error_context
        
        def failing_function():
            context = {
                "operation": "test_operation",
                "parameters": {"param1": "value1"},
                "timestamp": time.time()
            }
            
            try:
                raise ValueError("Test error")
            except Exception as e:
                error_context = collect_error_context(e, context)
                raise error_context
        
        with pytest.raises(ErrorContext) as exc_info:
            failing_function()
        
        error = exc_info.value
        assert error.operation == "test_operation"
        assert error.parameters["param1"] == "value1"
        assert error.original_exception.__class__.__name__ == "ValueError"
        assert "Test error" in str(error)
    
    def test_error_metrics_collection(self):
        """Test that errors are properly tracked for metrics"""
        from enhanced_error_handler import ErrorTracker
        
        tracker = ErrorTracker()
        
        # Record some errors
        tracker.record_error("module1", "function1", "ValueError", "Test error 1")
        tracker.record_error("module1", "function1", "ValueError", "Test error 2")
        tracker.record_error("module1", "function2", "TypeError", "Test error 3")
        
        # Check error statistics
        stats = tracker.get_error_statistics()
        
        assert stats["module1.function1"]["ValueError"] == 2
        assert stats["module1.function2"]["TypeError"] == 1
        assert len(stats) == 2  # Two functions recorded


class TestCircuitBreakerEnhancements:
    """Test enhanced circuit breaker patterns"""
    
    def test_operation_specific_circuit_breakers(self):
        """Test circuit breakers per operation type"""
        from enhanced_error_handler import OperationCircuitBreaker
        
        breaker = OperationCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1  # Short timeout for testing
        )
        
        # Test github_api operations
        for i in range(3):
            breaker.record_failure("github_api", "create_issue")
        
        # Should be open for github_api operations
        assert breaker.can_proceed("github_api", "create_issue") is False
        
        # But other operations should still work
        assert breaker.can_proceed("task_analyzer", "find_todos") is True
        
        # Wait for recovery
        time.sleep(0.2)
        
        # Should allow one test request
        assert breaker.can_proceed("github_api", "create_issue") is True


class TestValidationEnhancements:
    """Test enhanced input validation patterns"""
    
    def test_configuration_schema_validation(self):
        """Test schema-based configuration validation"""
        from enhanced_validation import (
            validate_config_schema,
            ConfigurationValidationError
        )
        
        # Valid configuration
        valid_config = {
            "github": {
                "username": "testuser",
                "managerRepo": "testuser/manager",
                "reposToScan": ["testuser/repo1", "testuser/repo2"]
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True,
                "maxIssuesPerRepo": 10
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
        
        assert validate_config_schema(valid_config) is True
        
        # Invalid configuration - missing required field
        invalid_config = {
            "github": {
                "username": "testuser"
                # Missing managerRepo and reposToScan
            }
        }
        
        with pytest.raises(ConfigurationValidationError) as exc_info:
            validate_config_schema(invalid_config)
        
        assert "managerRepo" in str(exc_info.value)
    
    def test_api_parameter_validation(self):
        """Test API parameter validation"""
        from enhanced_validation import (
            validate_api_parameters,
            ParameterValidationError
        )
        
        # Valid GitHub API parameters
        valid_params = {
            "repo_name": "user/repo",
            "title": "Valid issue title",
            "body": "Valid issue body",
            "labels": ["bug", "enhancement"]
        }
        
        assert validate_api_parameters(valid_params, "create_issue") is True
        
        # Invalid parameters
        invalid_params = {
            "repo_name": "../malicious",  # Invalid repo name
            "title": "",  # Empty title
            "labels": ["very-long-label-name-that-exceeds-github-limits" * 10]  # Too long
        }
        
        with pytest.raises(ParameterValidationError):
            validate_api_parameters(invalid_params, "create_issue")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])