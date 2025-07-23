"""
Unit tests for web dashboard security features

Tests rate limiting, input validation, XSS protection, and security headers
"""

import pytest
import time
from unittest.mock import patch, MagicMock
import json

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'web'))

# Mock environment configuration to avoid strict validation during tests
os.environ['RATE_LIMIT_MAX_REQUESTS'] = '100'  # Set to valid value
os.environ['RATE_LIMIT_WINDOW_SECONDS'] = '60'

from app import app, DashboardAPI


class TestWebSecurityFeatures:
    """Test security features in web dashboard"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        
        # Reset rate limiter for each test
        from app import rate_limiter
        rate_limiter.clients.clear()
        rate_limiter.max_requests = 100  # Set to high limit for most tests
        
        with app.test_client() as client:
            yield client
    
    def test_rate_limiting_on_api_endpoints(self, client):
        """Test that API endpoints have rate limiting"""
        # Override rate limiter for testing with more restrictive limits
        from app import rate_limiter
        rate_limiter.max_requests = 5  # Set low limit for testing
        rate_limiter.clients.clear()  # Clear any existing state
        
        # Make multiple rapid requests to API endpoint
        responses = []
        for i in range(10):  # Exceed test rate limit
            response = client.get('/api/health')
            responses.append(response.status_code)
        
        # Should get 429 (Too Many Requests) after hitting limit
        assert 429 in responses, f"Rate limiting should trigger 429 status. Got: {responses}"
    
    def test_input_validation_on_query_parameters(self, client):
        """Test input validation on query parameters"""
        # Test malicious limit parameter
        response = client.get('/api/tasks?limit=9999999')
        assert response.status_code == 400, "Should reject excessive limit values"
        
        # Test SQL injection attempt
        response = client.get('/api/tasks?limit=1; DROP TABLE users;--')
        assert response.status_code == 400, "Should reject SQL injection attempts"
        
        # Test XSS attempt in parameter
        response = client.get('/api/tasks?limit=<script>alert("xss")</script>')
        assert response.status_code == 400, "Should reject XSS attempts"
    
    def test_security_headers_present(self, client):
        """Test that security headers are present in responses"""
        response = client.get('/')
        
        # Check for important security headers
        assert 'X-Content-Type-Options' in response.headers
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        
        assert 'X-Frame-Options' in response.headers
        assert response.headers['X-Frame-Options'] == 'DENY'
        
        assert 'X-XSS-Protection' in response.headers
        assert response.headers['X-XSS-Protection'] == '1; mode=block'
        
        assert 'Content-Security-Policy' in response.headers
    
    def test_api_response_sanitization(self, client):
        """Test that API responses sanitize potentially dangerous content"""
        with patch.object(DashboardAPI, 'get_recent_tasks') as mock_tasks:
            # Mock data with potentially dangerous content
            mock_tasks.return_value = [
                {
                    "title": "<script>alert('xss')</script>Malicious Task",
                    "description": "javascript:alert('xss')",
                    "repository": "<img src=x onerror=alert('xss')>"
                }
            ]
            
            response = client.get('/api/tasks')
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Check that dangerous content is escaped/sanitized
            task = data[0]
            assert '&lt;script&gt;' in task['title']  # Should be HTML escaped
            assert 'javascript:' not in task['description']  # Should be removed
            assert 'onerror=' not in task['repository']  # Should be removed
    
    def test_error_response_information_disclosure(self, client):
        """Test that error responses don't leak sensitive information"""
        # Force an error by providing invalid input
        with patch('web.app.dashboard_api.get_backlog_status', side_effect=Exception("Database password: secret123")):
            response = client.get('/api/backlog')
            
            # Error response should be generic, not leak internal details
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data
            assert 'secret123' not in str(data), "Should not leak sensitive information"
            assert 'Database password' not in str(data), "Should not leak internal error details"
    
    def test_configuration_endpoint_security(self, client):
        """Test that configuration endpoint sanitizes sensitive data"""
        response = client.get('/api/config')
        data = json.loads(response.data)
        
        # Should not contain any tokens, passwords, or keys
        config_str = str(data).lower()
        sensitive_keywords = ['token', 'password', 'key', 'secret', 'auth']
        
        for keyword in sensitive_keywords:
            assert keyword not in config_str or 'redacted' in config_str or 'sanitized' in config_str, \
                f"Configuration should not expose {keyword}"


class TestInputValidation:
    """Test input validation functions"""
    
    def test_validate_limit_parameter(self):
        """Test limit parameter validation"""
        from app import validate_limit_parameter
        
        # Valid limits
        assert validate_limit_parameter("10") == 10
        assert validate_limit_parameter("1") == 1
        assert validate_limit_parameter("100") == 100
        
        # Invalid limits
        with pytest.raises(ValueError):
            validate_limit_parameter("0")  # Too low
        
        with pytest.raises(ValueError):
            validate_limit_parameter("1001")  # Too high
        
        with pytest.raises(ValueError):
            validate_limit_parameter("abc")  # Not a number
        
        with pytest.raises(ValueError):
            validate_limit_parameter("-5")  # Negative
    
    def test_sanitize_user_input(self):
        """Test user input sanitization"""
        from app import sanitize_user_input
        
        # Test XSS prevention - html.escape quotes single quotes as &#x27;
        result = sanitize_user_input("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result and "&lt;/script&gt;" in result
        assert "alert" in result  # Function name should be preserved
        
        # Test SQL injection prevention
        assert sanitize_user_input("'; DROP TABLE users; --") == "&#x27;; DROP TABLE users; --"
        
        # Test normal text is preserved
        assert sanitize_user_input("Normal text 123") == "Normal text 123"