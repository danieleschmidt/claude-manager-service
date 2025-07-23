"""
Integration tests for Enhanced Error Handling system

These tests verify that the enhanced error handling integrates correctly
with existing modules and provides improved error reporting and handling.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from error_handler import (
    NetworkError, RateLimitError, FileOperationError,
    get_rate_limiter, get_circuit_breaker, get_error_tracker
)
from security import (
    validate_token_enhanced, sanitize_repo_name, 
    InvalidTokenFormatError, PathTraversalError
)
from validation import validate_config_schema, ConfigurationValidationError


class TestGitHubAPIEnhancedErrorHandling:
    """Test enhanced error handling in GitHub API operations"""
    
    @pytest.fixture
    def mock_github_client(self):
        """Mock GitHub client for testing"""
        with patch('github_api.Github') as mock_github:
            mock_client = MagicMock()
            mock_github.return_value = mock_client
            
            # Mock user for initialization
            mock_user = MagicMock()
            mock_user.login = "testuser"
            mock_client.get_user.return_value = mock_user
            
            yield mock_client
    
    @pytest.fixture
    def mock_secure_config(self):
        """Mock secure configuration"""
        with patch('github_api.get_secure_config') as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get_github_token.return_value = "ghp_" + "a" * 36
            mock_config.return_value = mock_config_obj
            yield mock_config_obj
    
    def test_github_api_specific_error_handling(self, mock_github_client, mock_secure_config):
        """Test that GitHub API operations use specific error handling"""
        from github_api import GitHubAPI
        from github import GithubException
        
        # Create API instance
        api = GitHubAPI()
        
        # Mock repository not found (404)
        mock_github_client.get_repo.side_effect = GithubException(404, "Not Found", None)
        
        # Should return None for 404 (not found)
        result = api.get_repo("nonexistent/repo")
        assert result is None
        
        # Mock permission denied (403)
        mock_github_client.get_repo.side_effect = GithubException(403, "Forbidden", None)
        
        # Should raise RateLimitError for 403
        with pytest.raises(RateLimitError):
            api.get_repo("forbidden/repo")
    
    def test_input_validation_integration(self, mock_github_client, mock_secure_config):
        """Test input validation integration in API calls"""
        from github_api import GitHubAPI
        
        api = GitHubAPI()
        
        # Test invalid repository name
        with pytest.raises(NetworkError) as exc_info:
            api.get_repo("../malicious/path")
        
        assert "Invalid repository name" in str(exc_info.value)
    
    def test_rate_limiting_integration(self):
        """Test rate limiting integration"""
        from error_handler import get_rate_limiter
        
        limiter = get_rate_limiter()
        
        # Should be able to proceed initially
        assert limiter.can_proceed("test_integration") is True
        
        # Check remaining quota
        remaining = limiter.get_remaining_quota("test_integration")
        assert remaining < limiter.max_requests  # Should be reduced by 1


class TestConfigurationValidationIntegration:
    """Test configuration validation integration"""
    
    def test_real_config_validation(self):
        """Test validation with real configuration files"""
        # Test with valid configuration
        valid_config = {
            "github": {
                "username": "testuser",
                "managerRepo": "testuser/manager",
                "reposToScan": ["testuser/repo1", "testuser/repo2"]
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
        
        # Should validate successfully
        assert validate_config_schema(valid_config) is True
        
        # Test with invalid configuration
        invalid_config = {
            "github": {
                "username": "",  # Empty username
                "managerRepo": "invalid_repo_format",  # Missing slash
                "reposToScan": []  # Empty list
            }
        }
        
        with pytest.raises(ConfigurationValidationError):
            validate_config_schema(invalid_config)
    
    def test_config_file_integration(self):
        """Test loading and validating actual config files"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "github": {
                    "username": "testuser",
                    "managerRepo": "testuser/manager",
                    "reposToScan": ["testuser/repo"]
                },
                "analyzer": {
                    "scanForTodos": True,
                    "scanOpenIssues": True
                },
                "executor": {
                    "terragonUsername": "@terragon-labs"
                }
            }
            json.dump(config, f)
            temp_path = f.name
        
        try:
            # Load and validate
            from error_handler import safe_json_load
            loaded_config = safe_json_load(temp_path)
            
            # Should validate successfully
            assert validate_config_schema(loaded_config) is True
            
        finally:
            os.unlink(temp_path)


class TestSecurityValidationIntegration:
    """Test security validation integration"""
    
    def test_token_validation_integration(self):
        """Test token validation with different patterns"""
        # Valid GitHub tokens
        valid_tokens = [
            "ghp_" + "a" * 36,  # Personal access token
            "ghs_" + "b" * 36,  # Server-to-server token
            "gho_" + "c" * 36,  # OAuth token
        ]
        
        for token in valid_tokens:
            assert validate_token_enhanced(token, "github") is True
        
        # Invalid tokens (weak token gets caught first)
        from security import WeakTokenError
        with pytest.raises(WeakTokenError):
            validate_token_enhanced("invalid_token", "github")
        
        # Legacy token should raise ExpiredTokenError
        from security import ExpiredTokenError
        with pytest.raises(ExpiredTokenError):
            validate_token_enhanced("a" * 40, "github")  # 40-char hex (old format)
    
    def test_path_security_integration(self):
        """Test path security validation"""
        from security import safe_path_join, validate_safe_path
        
        # Safe path operations
        safe_path = safe_path_join("/base", "subdir", "file.txt")
        assert safe_path == "/base/subdir/file.txt"
        
        # Path traversal should be blocked
        with pytest.raises(PathTraversalError):
            safe_path_join("/base", "../../../etc/passwd")
        
        with pytest.raises(PathTraversalError):
            validate_safe_path("../../../etc/passwd", "/safe/base")
    
    def test_repository_name_security(self):
        """Test repository name security validation"""
        # Valid repository names
        valid_repos = ["user/repo", "org-name/repo-name", "user/repo.name", "user123/repo_456"]
        
        for repo in valid_repos:
            sanitized = sanitize_repo_name(repo)
            assert sanitized == repo
        
        # Invalid repository names
        invalid_repos = ["../malicious", "user/repo\x00null", "user", "user/", "/repo"]
        
        for repo in invalid_repos:
            with pytest.raises(ValueError):
                sanitize_repo_name(repo)


class TestErrorMetricsIntegration:
    """Test error tracking and metrics integration"""
    
    def test_error_tracking(self):
        """Test error tracking integration"""
        tracker = get_error_tracker()
        
        # Record some errors
        tracker.record_error("github_api", "get_repo", "NetworkError", "Connection failed")
        tracker.record_error("github_api", "create_issue", "ValidationError", "Invalid data")
        tracker.record_error("github_api", "get_repo", "NetworkError", "Timeout")
        
        # Get statistics
        stats = tracker.get_error_statistics()
        
        assert "github_api.get_repo" in stats
        assert stats["github_api.get_repo"]["NetworkError"] == 2
        assert "github_api.create_issue" in stats
        assert stats["github_api.create_issue"]["ValidationError"] == 1
        
        # Get recent errors
        recent = tracker.get_recent_errors(5)
        assert len(recent) == 3
        assert recent[-1]["error_type"] == "NetworkError"
    
    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration"""
        breaker = get_circuit_breaker()
        
        # Should start in closed state
        assert breaker.can_proceed("test_module", "test_operation") is True
        
        # Record failures to open circuit
        for _ in range(5):  # Default threshold
            breaker.record_failure("test_module", "test_operation")
        
        # Should now be open
        assert breaker.can_proceed("test_module", "test_operation") is False
        
        # Different operation should still work
        assert breaker.can_proceed("test_module", "different_operation") is True


class TestEndToEndErrorFlow:
    """Test complete error flow from detection to reporting"""
    
    @pytest.fixture
    def mock_environment(self):
        """Mock complete environment for testing"""
        with patch('github_api.get_secure_config') as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get_github_token.return_value = "ghp_" + "a" * 36
            mock_config.return_value = mock_config_obj
            
            with patch('github_api.Github') as mock_github:
                mock_client = MagicMock()
                mock_github.return_value = mock_client
                
                # Mock successful user auth
                mock_user = MagicMock()
                mock_user.login = "testuser"
                mock_client.get_user.return_value = mock_user
                
                yield mock_client
    
    def test_complete_error_flow(self, mock_environment):
        """Test complete error flow from API call to error reporting"""
        from github_api import GitHubAPI
        from github import GithubException
        
        # Set up API client
        api = GitHubAPI()
        
        # Set up error tracking
        tracker = get_error_tracker()
        initial_error_count = len(tracker.get_recent_errors())
        
        # Mock API failure
        mock_environment.get_repo.side_effect = GithubException(500, "Internal Server Error", None)
        
        # Should raise NetworkError and record it
        with pytest.raises(NetworkError):
            api.get_repo("test/repo")
        
        # Error should be tracked
        errors = tracker.get_recent_errors()
        assert len(errors) > initial_error_count
        
        # Find our error
        our_error = None
        for error in errors:
            if error["function"] == "get_repo" and error["module"] == "github_api":
                our_error = error
                break
        
        assert our_error is not None
        assert our_error["error_type"] == "NetworkError"
    
    def test_validation_error_chain(self):
        """Test validation error propagation"""
        from github_api import GitHubAPI
        
        with patch('github_api.get_secure_config') as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get_github_token.return_value = "ghp_" + "a" * 36
            mock_config.return_value = mock_config_obj
            
            with patch('github_api.Github') as mock_github:
                mock_client = MagicMock()
                mock_github.return_value = mock_client
                
                # Mock successful user auth
                mock_user = MagicMock()
                mock_user.login = "testuser"
                mock_client.get_user.return_value = mock_user
                
                api = GitHubAPI()
                
                # Test validation error propagation
                with pytest.raises(NetworkError) as exc_info:
                    api.get_repo("invalid/../repo")
                
                # Should contain validation error information
                assert "Invalid repository name" in str(exc_info.value)
                assert exc_info.value.operation == "get_repository"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])