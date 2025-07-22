"""
Enhanced Security Validation for Claude Manager Service

This module provides improved security validation patterns including:
- Enhanced token validation with specific patterns
- Path traversal prevention
- Improved input sanitization
- Security-focused validation functions

Features:
- Token format validation with specific patterns for different services
- Path traversal attack prevention
- Enhanced input sanitization for various content types
- Security-focused repository name validation
"""

import re
import os
import html
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import logging
import hashlib
import time

from logger import get_logger


# Custom Security Exceptions
class SecurityValidationError(Exception):
    """Base class for security validation errors"""
    pass


class TokenValidationError(SecurityValidationError):
    """Base class for token validation errors"""
    pass


class InvalidTokenFormatError(TokenValidationError):
    """Token format is invalid for the specified service"""
    pass


class WeakTokenError(TokenValidationError):
    """Token is too weak or short"""
    pass


class ExpiredTokenError(TokenValidationError):
    """Token appears to be expired or using deprecated format"""
    pass


class PathTraversalError(SecurityValidationError):
    """Path contains traversal attack patterns"""
    pass


class InputSanitizationError(SecurityValidationError):
    """Input failed sanitization checks"""
    pass


# Token Validation Patterns
TOKEN_PATTERNS = {
    "github": {
        "personal_access_token": re.compile(r"^ghp_[A-Za-z0-9]{36}$"),
        "oauth_token": re.compile(r"^gho_[A-Za-z0-9]{36}$"),
        "user_to_server": re.compile(r"^ghu_[A-Za-z0-9]{36}$"),
        "server_to_server": re.compile(r"^ghs_[A-Za-z0-9]{36}$"),
        "refresh_token": re.compile(r"^ghr_[A-Za-z0-9]+$"),
        "legacy_40_char": re.compile(r"^[a-f0-9]{40}$")  # Old format, likely expired
    },
    "terragon": {
        "api_key": re.compile(r"^tr_[A-Za-z0-9]{32,64}$"),
        "session_token": re.compile(r"^ts_[A-Za-z0-9]{40}$")
    }
}

# Minimum token lengths
MIN_TOKEN_LENGTHS = {
    "github": 40,
    "terragon": 35
}

# Dangerous patterns for input sanitization
DANGEROUS_PATTERNS = [
    re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
    re.compile(r'javascript:', re.IGNORECASE),
    re.compile(r'vbscript:', re.IGNORECASE),
    re.compile(r'onload\s*=', re.IGNORECASE),
    re.compile(r'onerror\s*=', re.IGNORECASE),
    re.compile(r'onclick\s*=', re.IGNORECASE),
    re.compile(r'<iframe[^>]*>', re.IGNORECASE),
    re.compile(r'<object[^>]*>', re.IGNORECASE),
    re.compile(r'<embed[^>]*>', re.IGNORECASE),
    re.compile(r'\x00'),  # Null bytes
    re.compile(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]')  # Control characters
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    re.compile(r'\.\./', re.IGNORECASE),
    re.compile(r'\.\.\\\\', re.IGNORECASE),
    re.compile(r'%2e%2e%2f', re.IGNORECASE),
    re.compile(r'%2e%2e%5c', re.IGNORECASE),
    re.compile(r'%252e%252e%252f', re.IGNORECASE),
    re.compile(r'%c0%ae%c0%ae%c0%af', re.IGNORECASE)
]


def validate_token_enhanced(token: str, service: str) -> bool:
    """
    Enhanced token validation with specific patterns for different services
    
    Args:
        token: Token to validate
        service: Service type ('github', 'terragon', etc.)
        
    Returns:
        True if token is valid
        
    Raises:
        InvalidTokenFormatError: If token format is invalid
        WeakTokenError: If token is too weak
        ExpiredTokenError: If token appears expired
    """
    logger = get_logger(__name__)
    
    if not token or not isinstance(token, str):
        raise InvalidTokenFormatError("Token must be a non-empty string")
    
    if service not in TOKEN_PATTERNS:
        raise InvalidTokenFormatError(f"Unknown service: {service}")
    
    # Check minimum length
    min_length = MIN_TOKEN_LENGTHS.get(service, 32)
    if len(token) < min_length:
        raise WeakTokenError(f"Token too short: {len(token)} < {min_length} characters")
    
    # Check against known patterns
    patterns = TOKEN_PATTERNS[service]
    pattern_matched = False
    
    for pattern_name, pattern in patterns.items():
        if pattern.match(token):
            pattern_matched = True
            
            # Special handling for legacy patterns
            if pattern_name == "legacy_40_char":
                logger.warning(f"Legacy {service} token format detected - may be expired")
                raise ExpiredTokenError(f"Token uses deprecated format for {service}")
            
            logger.debug(f"Token matched {service} {pattern_name} pattern")
            break
    
    if not pattern_matched:
        raise InvalidTokenFormatError(f"Token format invalid for {service}")
    
    return True


def validate_safe_path(path: str, base_directory: Optional[str] = None) -> bool:
    """
    Validate that a path is safe and doesn't contain traversal attacks
    
    Args:
        path: Path to validate
        base_directory: Base directory to restrict access to
        
    Returns:
        True if path is safe
        
    Raises:
        PathTraversalError: If path contains traversal patterns
    """
    if not path:
        raise PathTraversalError("Empty path")
    
    # Check for path traversal patterns
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if pattern.search(path):
            raise PathTraversalError(f"Path traversal pattern detected in: {path}")
    
    # Normalize path and check for traversal
    try:
        normalized = os.path.normpath(path)
        
        # Only check for absolute paths if the path wasn't already absolute and base_directory exists
        if os.path.isabs(normalized) and base_directory and not os.path.isabs(path):
            raise PathTraversalError(f"Path resolves to absolute when relative expected: {path}")
        
        # If base directory is specified, ensure path stays within it
        if base_directory:
            base_abs = os.path.abspath(base_directory)
            
            if os.path.isabs(normalized):
                target_abs = normalized
            else:
                target_abs = os.path.abspath(os.path.join(base_abs, normalized))
            
            # Check if target is within base directory
            if not target_abs.startswith(base_abs + os.sep) and target_abs != base_abs:
                raise PathTraversalError(f"Path escapes base directory: {path}")
        
    except (ValueError, OSError) as e:
        raise PathTraversalError(f"Invalid path: {path} - {str(e)}")
    
    return True


def safe_path_join(base_dir: str, *paths: str) -> str:
    """
    Safely join paths with traversal protection
    
    Args:
        base_dir: Base directory
        *paths: Path components to join
        
    Returns:
        Safe joined path
        
    Raises:
        PathTraversalError: If any path component is unsafe
    """
    # Validate each path component
    for path in paths:
        validate_safe_path(path, base_dir)
    
    # Join paths
    result = os.path.join(base_dir, *paths)
    
    # Final validation
    validate_safe_path(result, base_dir)
    
    return result


def sanitize_repo_name(repo_name: str) -> str:
    """
    Sanitize and validate repository name
    
    Args:
        repo_name: Repository name (e.g., "user/repo")
        
    Returns:
        Sanitized repository name
        
    Raises:
        ValueError: If repository name is invalid
    """
    if not repo_name or not isinstance(repo_name, str):
        raise ValueError("Repository name must be a non-empty string")
    
    # Remove any leading/trailing whitespace
    repo_name = repo_name.strip()
    
    # Check for null bytes and control characters
    for pattern in DANGEROUS_PATTERNS[-2:]:  # Last two are null bytes and control chars
        if pattern.search(repo_name):
            raise ValueError(f"Repository name contains forbidden characters: {repo_name}")
    
    # Check for path traversal attempts
    try:
        validate_safe_path(repo_name)
    except PathTraversalError as e:
        raise ValueError(f"Repository name contains path traversal: {str(e)}")
    
    # Validate GitHub repository name format (user/repo or org/repo)
    repo_pattern = re.compile(r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$')
    if not repo_pattern.match(repo_name):
        raise ValueError(f"Invalid repository name format: {repo_name}")
    
    # Check length constraints
    if len(repo_name) > 100:  # GitHub limit
        raise ValueError(f"Repository name too long: {len(repo_name)} > 100 characters")
    
    parts = repo_name.split('/')
    if len(parts) != 2:
        raise ValueError(f"Repository name must be in format 'owner/repo': {repo_name}")
    
    owner, repo = parts
    if len(owner) < 1 or len(owner) > 39:  # GitHub limits
        raise ValueError(f"Repository owner name invalid length: {len(owner)}")
    
    if len(repo) < 1 or len(repo) > 100:  # GitHub limits
        raise ValueError(f"Repository name invalid length: {len(repo)}")
    
    return repo_name


def sanitize_issue_content_enhanced(content: str) -> str:
    """
    Enhanced sanitization for issue content
    
    Args:
        content: Content to sanitize
        
    Returns:
        Sanitized content
    """
    if not content:
        return ""
    
    # Remove dangerous patterns
    sanitized = content
    for pattern in DANGEROUS_PATTERNS:
        sanitized = pattern.sub('', sanitized)
    
    # HTML encode any remaining special characters while preserving markdown
    # This is a simple approach - for production, consider using a markdown-aware sanitizer
    sanitized = html.escape(sanitized, quote=False)
    
    # Truncate if too long (GitHub API limit is ~65536 chars, configurable via environment)
    try:
        from config_env import get_env_config
        config = get_env_config()
        max_length = config.security_enhanced_max_content_length
    except ImportError:
        max_length = int(os.getenv('SECURITY_ENHANCED_MAX_CONTENT_LENGTH', '60000'))
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "\n\n[Content truncated for safety]"
    
    return sanitized


def sanitize_input_enhanced(input_data: Union[str, Dict, List], input_type: str = "general") -> Union[str, Dict, List]:
    """
    Enhanced input sanitization based on input type
    
    Args:
        input_data: Data to sanitize
        input_type: Type of input (general, filename, url, etc.)
        
    Returns:
        Sanitized input
    """
    if isinstance(input_data, str):
        return _sanitize_string_by_type(input_data, input_type)
    elif isinstance(input_data, dict):
        return {key: sanitize_input_enhanced(value, input_type) for key, value in input_data.items()}
    elif isinstance(input_data, list):
        return [sanitize_input_enhanced(item, input_type) for item in input_data]
    else:
        return input_data


def _sanitize_string_by_type(text: str, input_type: str) -> str:
    """Sanitize string based on specific input type"""
    if not text:
        return text
    
    # Common sanitization
    sanitized = text
    
    if input_type == "filename":
        # Remove path separators and dangerous characters
        dangerous_filename_chars = r'[<>:"/\\|?*\x00-\x1f]'
        sanitized = re.sub(dangerous_filename_chars, '_', sanitized)
        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
    
    elif input_type == "url":
        # Basic URL validation and sanitization
        if not sanitized.startswith(('http://', 'https://')):
            raise InputSanitizationError("URL must start with http:// or https://")
        # URL encode any dangerous characters
        sanitized = urllib.parse.quote(sanitized, safe=':/?#[]@!$&\'()*+,;=')
    
    elif input_type == "issue_content":
        sanitized = sanitize_issue_content_enhanced(sanitized)
    
    elif input_type == "repo_name":
        sanitized = sanitize_repo_name(sanitized)
    
    else:  # General sanitization
        # Remove null bytes and most control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
    
    return sanitized


def generate_secure_hash(data: str, salt: Optional[str] = None) -> str:
    """
    Generate secure hash for data validation
    
    Args:
        data: Data to hash
        salt: Optional salt for hashing
        
    Returns:
        Hex-encoded hash
    """
    if salt is None:
        salt = str(time.time())
    
    hasher = hashlib.sha256()
    hasher.update(salt.encode('utf-8'))
    hasher.update(data.encode('utf-8'))
    
    return hasher.hexdigest()


def validate_configuration_security(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration for security issues
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of security warnings/issues found
    """
    warnings = []
    
    # Check for hardcoded tokens
    def check_for_tokens(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and ("token" in key.lower() or "key" in key.lower()):
                    if len(value) > 10:  # Likely a real token
                        warnings.append(f"Potential hardcoded token found at {current_path}")
                check_for_tokens(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_for_tokens(item, f"{path}[{i}]")
    
    check_for_tokens(config)
    
    # Check for insecure URLs
    if "urls" in config:
        urls = config["urls"]
        if isinstance(urls, dict):
            for key, url in urls.items():
                if isinstance(url, str) and url.startswith("http://"):
                    warnings.append(f"Insecure HTTP URL found: {key} = {url}")
    
    # Check repository names for security
    if "github" in config and "reposToScan" in config["github"]:
        repos = config["github"]["reposToScan"]
        for repo in repos:
            try:
                sanitize_repo_name(repo)
            except ValueError as e:
                warnings.append(f"Invalid repository name: {repo} - {str(e)}")
    
    return warnings


class SecurityValidator:
    """
    Centralized security validation class
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._validation_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_age = 300  # 5 minutes
    
    def validate_token(self, token: str, service: str, use_cache: bool = True) -> bool:
        """
        Validate token with optional caching
        
        Args:
            token: Token to validate
            service: Service type
            use_cache: Whether to use validation cache
            
        Returns:
            True if valid
        """
        if use_cache:
            cache_key = hashlib.md5(f"{service}:{token}".encode()).hexdigest()
            
            if cache_key in self._validation_cache:
                cached = self._validation_cache[cache_key]
                if time.time() - cached["timestamp"] < self._cache_max_age:
                    return cached["valid"]
        
        try:
            is_valid = validate_token_enhanced(token, service)
            
            if use_cache:
                self._validation_cache[cache_key] = {
                    "valid": is_valid,
                    "timestamp": time.time()
                }
            
            return is_valid
            
        except TokenValidationError:
            if use_cache:
                self._validation_cache[cache_key] = {
                    "valid": False,
                    "timestamp": time.time()
                }
            raise
    
    def validate_input_batch(self, inputs: Dict[str, Any], types: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate multiple inputs in batch
        
        Args:
            inputs: Dictionary of inputs to validate
            types: Dictionary mapping input names to their types
            
        Returns:
            Dictionary of sanitized inputs
        """
        results = {}
        
        for key, value in inputs.items():
            input_type = types.get(key, "general")
            try:
                results[key] = sanitize_input_enhanced(value, input_type)
            except (InputSanitizationError, ValueError, PathTraversalError) as e:
                self.logger.error(f"Input validation failed for {key}: {str(e)}")
                raise
        
        return results
    
    def clear_cache(self):
        """Clear validation cache"""
        self._validation_cache.clear()


# Global validator instance
_security_validator = SecurityValidator()


def get_security_validator() -> SecurityValidator:
    """Get global security validator instance"""
    return _security_validator