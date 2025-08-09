"""
Security utilities for Claude Manager Service

This module provides secure handling of sensitive data like tokens,
environment variables, and subprocess execution.

Enhanced Features:
- Enhanced token validation with specific patterns
- Path traversal prevention
- Improved input sanitization
- Security-focused validation functions
"""
import os
import re
import html
import urllib.parse
import subprocess
import tempfile
import hashlib
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from .logger import get_logger

logger = get_logger(__name__)


# Enhanced Security Exceptions
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
    re.compile(r'\.\.[\\/]'),
    re.compile(r'[\\/]\.\.'),
    re.compile(r'%2e%2e%2f', re.IGNORECASE),
    re.compile(r'%2e%2e%5c', re.IGNORECASE),
    re.compile(r'%252e%252e%252f', re.IGNORECASE),
    re.compile(r'%c0%ae'),
    re.compile(r'%c1%9c')
]


class SecureConfig:
    """
    Secure configuration manager that handles sensitive data
    
    Features:
    - Environment variable validation
    - Token sanitization for logging
    - Secure temporary file handling
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.SecureConfig")
        self._config_cache = {}
        self._validate_required_vars()
    
    def _validate_required_vars(self):
        """Validate that required environment variables are set"""
        required_vars = ['GITHUB_TOKEN']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"Missing required environment variables: {missing_vars}")
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        self.logger.info("All required environment variables are set")
    
    def get_github_token(self) -> str:
        """
        Get GitHub token from environment with validation
        
        Returns:
            str: GitHub token
            
        Raises:
            ValueError: If token is not set or invalid
        """
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            self.logger.error("GITHUB_TOKEN environment variable not set")
            raise ValueError("GITHUB_TOKEN environment variable not set")
        
        if len(token) < 20:  # Basic validation - GitHub tokens are much longer
            self.logger.error("GITHUB_TOKEN appears to be invalid (too short)")
            raise ValueError("GITHUB_TOKEN appears to be invalid")
        
        # Test for common token prefixes
        valid_prefixes = ['ghp_', 'github_pat_', 'gho_', 'ghu_', 'ghs_', 'ghr_']
        if not any(token.startswith(prefix) for prefix in valid_prefixes):
            self.logger.warning("GITHUB_TOKEN does not have expected prefix - may be invalid")
        
        self.logger.debug("GitHub token retrieved and validated")
        return token
    
    def get_optional_token(self, token_name: str) -> Optional[str]:
        """
        Get optional token from environment
        
        Args:
            token_name (str): Name of the environment variable
            
        Returns:
            Optional[str]: Token if set, None otherwise
        """
        token = os.getenv(token_name)
        if token:
            self.logger.debug(f"Optional token {token_name} is available")
        else:
            self.logger.debug(f"Optional token {token_name} is not set")
        return token
    
    def sanitize_for_logging(self, text: str) -> str:
        """
        Sanitize text for safe logging by masking sensitive data
        
        Args:
            text (str): Text that might contain sensitive data
            
        Returns:
            str: Sanitized text with sensitive data masked
        """
        if not text:
            return text
        
        # Mask GitHub tokens
        import re
        
        # Common GitHub token patterns
        token_patterns = [
            r'ghp_[a-zA-Z0-9]{36}',  # Personal access tokens
            r'github_pat_[a-zA-Z0-9_]{82}',  # Fine-grained personal access tokens
            r'gho_[a-zA-Z0-9]{36}',  # OAuth tokens
            r'ghu_[a-zA-Z0-9]{36}',  # GitHub App user-to-server tokens
            r'ghs_[a-zA-Z0-9]{36}',  # GitHub App server-to-server tokens
            r'ghr_[a-zA-Z0-9]{36}',  # GitHub App refresh tokens
        ]
        
        sanitized = text
        for pattern in token_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized)
        
        # Also mask anything that looks like a token in URLs
        sanitized = re.sub(r'://[^:]+:[^@]+@', '://[REDACTED]:[REDACTED]@', sanitized)
        
        return sanitized
    
    def get_log_level(self) -> str:
        """Get log level from environment with validation"""
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        if level not in valid_levels:
            self.logger.warning(f"Invalid log level '{level}', defaulting to INFO")
            return 'INFO'
        
        return level


class SecureSubprocess:
    """
    Secure subprocess execution utilities
    
    Features:
    - Token sanitization in command logging
    - Secure temporary file handling
    - Environment variable sanitization
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.SecureSubprocess")
        self.config = SecureConfig()
    
    def run_git_clone(self, repo_url: str, target_dir: str, token: str, timeout: int = 300) -> subprocess.CompletedProcess:
        """
        Securely clone a repository using a token
        
        Args:
            repo_url (str): Repository URL (without authentication)
            target_dir (str): Target directory for cloning
            token (str): GitHub token for authentication
            timeout (int): Timeout in seconds
            
        Returns:
            subprocess.CompletedProcess: Result of the clone operation
        """
        # Construct authenticated URL
        if repo_url.startswith('https://github.com/'):
            repo_path = repo_url.replace('https://github.com/', '')
            auth_url = f"https://x-access-token:{token}@github.com/{repo_path}"
        else:
            self.logger.error(f"Unsupported repository URL format: {repo_url}")
            raise ValueError(f"Unsupported repository URL format: {repo_url}")
        
        # Prepare command
        cmd = ['git', 'clone', auth_url, target_dir]
        
        # Log sanitized version
        sanitized_cmd = ['git', 'clone', self.config.sanitize_for_logging(auth_url), target_dir]
        self.logger.info(f"Executing git clone: {' '.join(sanitized_cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False  # Don't raise on non-zero exit
            )
            
            if result.returncode == 0:
                self.logger.info(f"Git clone completed successfully")
            else:
                # Sanitize error output before logging
                sanitized_stderr = self.config.sanitize_for_logging(result.stderr)
                self.logger.error(f"Git clone failed: {sanitized_stderr}")
            
            return result
            
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Git clone timed out after {timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Error during git clone: {e}")
            raise
    
    def run_with_sanitized_logging(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        """
        Run subprocess with sanitized command logging
        
        Args:
            cmd (List[str]): Command to execute
            **kwargs: Additional arguments to pass to subprocess.run
            
        Returns:
            subprocess.CompletedProcess: Result of the command execution
        """
        # Sanitize command for logging
        sanitized_cmd = [self.config.sanitize_for_logging(arg) for arg in cmd]
        self.logger.debug(f"Executing command: {' '.join(sanitized_cmd)}")
        
        try:
            result = subprocess.run(cmd, **kwargs)
            
            if hasattr(result, 'returncode'):
                if result.returncode == 0:
                    self.logger.debug("Command completed successfully")
                else:
                    self.logger.warning(f"Command failed with exit code {result.returncode}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            raise


class SecureTempDir:
    """
    Context manager for secure temporary directory handling
    
    Features:
    - Automatic cleanup
    - Secure permissions
    - Logging of creation and cleanup
    """
    
    def __init__(self, prefix: str = "claude_manager_"):
        self.logger = get_logger(f"{__name__}.SecureTempDir")
        self.prefix = prefix
        self.temp_dir = None
    
    def __enter__(self) -> Path:
        """Create secure temporary directory"""
        self.temp_dir = tempfile.mkdtemp(prefix=self.prefix)
        temp_path = Path(self.temp_dir)
        
        # Set secure permissions (owner only)
        temp_path.chmod(0o700)
        
        self.logger.debug(f"Created secure temporary directory: {temp_path}")
        return temp_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory"""
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                self.logger.error(f"Failed to clean up temporary directory {self.temp_dir}: {e}")
        
        self.temp_dir = None


def validate_repo_name(repo_name: str) -> bool:
    """
    Validate repository name format for security
    
    Args:
        repo_name (str): Repository name in format "owner/repo"
        
    Returns:
        bool: True if valid, False otherwise
    """
    logger = get_logger(f"{__name__}.validate_repo_name")
    
    if not repo_name:
        logger.warning("Empty repository name")
        return False
    
    # Basic format validation
    if '/' not in repo_name:
        logger.warning(f"Invalid repository name format: {repo_name}")
        return False
    
    parts = repo_name.split('/')
    if len(parts) != 2:
        logger.warning(f"Invalid repository name format: {repo_name}")
        return False
    
    owner, repo = parts
    
    # Check for dangerous characters
    import re
    if not re.match(r'^[a-zA-Z0-9._-]+$', owner) or not re.match(r'^[a-zA-Z0-9._-]+$', repo):
        logger.warning(f"Repository name contains invalid characters: {repo_name}")
        return False
    
    # Check length limits (GitHub limits)
    if len(owner) > 39 or len(repo) > 100:
        logger.warning(f"Repository name exceeds length limits: {repo_name}")
        return False
    
    logger.debug(f"Repository name validated: {repo_name}")
    return True


def sanitize_issue_content(content: str) -> str:
    """
    Sanitize issue content for safe processing
    
    Args:
        content (str): Raw issue content
        
    Returns:
        str: Sanitized content
    """
    if not content:
        return ""
    
    # Basic sanitization - remove null bytes and control characters
    sanitized = content.replace('\x00', '').replace('\r', '\n')
    
    # Limit length to prevent DoS (configurable via environment variable)
    try:
        from config_env import get_env_config
        config = get_env_config()
        max_length = config.security_max_content_length
    except ImportError:
        max_length = int(os.getenv('SECURITY_MAX_CONTENT_LENGTH', '50000'))  # 50KB default limit
    if len(sanitized) > max_length:
        logger = get_logger(f"{__name__}.sanitize_issue_content")
        logger.warning(f"Issue content truncated from {len(sanitized)} to {max_length} characters")
        sanitized = sanitized[:max_length] + "\n\n[Content truncated for security]"
    
    return sanitized


# Enhanced Security Functions
def validate_token_enhanced(token: str, service: str = "github") -> bool:
    """
    Enhanced token validation with specific service patterns
    
    Args:
        token: Token to validate
        service: Service type (github, terragon)
        
    Returns:
        True if token is valid
        
    Raises:
        InvalidTokenFormatError: If token format is invalid
        WeakTokenError: If token is too weak
        ExpiredTokenError: If token appears expired
    """
    return validate_token_format(token, service)


def validate_token_format(token: str, service: str = "github") -> bool:
    """
    Validate token format using service-specific patterns
    
    Args:
        token: Token to validate
        service: Service type (github, terragon)
        
    Returns:
        True if token format is valid
        
    Raises:
        InvalidTokenFormatError: If token format is invalid
        WeakTokenError: If token is too weak
        ExpiredTokenError: If token appears expired
    """
    if not token:
        raise WeakTokenError("Empty token provided")
    
    # Check minimum length
    min_length = MIN_TOKEN_LENGTHS.get(service, 20)
    if len(token) < min_length:
        raise WeakTokenError(f"Token too short (minimum {min_length} characters for {service})")
    
    # Check against patterns
    if service in TOKEN_PATTERNS:
        patterns = TOKEN_PATTERNS[service]
        
        # Check if any pattern matches
        for pattern_name, pattern in patterns.items():
            if pattern.match(token):
                # Special handling for legacy tokens
                if pattern_name == "legacy_40_char":
                    logger.warning("Using legacy token format - consider upgrading")
                    raise ExpiredTokenError("Legacy token format detected - please upgrade")
                
                logger.debug(f"Token format validated: {pattern_name}")
                return True
        
        # No pattern matched
        raise InvalidTokenFormatError(f"Token format not recognized for {service}")
    
    # Unknown service - do basic validation
    logger.warning(f"Unknown service '{service}' - performing basic validation only")
    return len(token) >= min_length


def sanitize_repo_name(repo_name: str) -> str:
    """
    Sanitize repository name for security
    
    Args:
        repo_name: Repository name to sanitize
        
    Returns:
        Sanitized repository name
        
    Raises:
        PathTraversalError: If path traversal detected
        InputSanitizationError: If input cannot be sanitized
    """
    if not repo_name:
        raise InputSanitizationError("Empty repository name")
    
    # Check for path traversal
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if pattern.search(repo_name):
            raise PathTraversalError(f"Path traversal detected in repository name: {repo_name}")
    
    # URL decode to catch encoded attacks
    decoded = urllib.parse.unquote(repo_name)
    if decoded != repo_name:
        # Check decoded version too
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if pattern.search(decoded):
                raise PathTraversalError(f"Path traversal detected in decoded repository name: {decoded}")
    
    # Basic sanitization
    sanitized = re.sub(r'[^a-zA-Z0-9._/-]', '', repo_name)
    
    # Validate format
    if not validate_repo_name(sanitized):
        raise InputSanitizationError(f"Repository name failed validation after sanitization: {sanitized}")
    
    return sanitized


def sanitize_issue_content_enhanced(content: str) -> str:
    """
    Enhanced issue content sanitization with security focus
    
    Args:
        content: Content to sanitize
        
    Returns:
        Sanitized content
        
    Raises:
        InputSanitizationError: If content cannot be safely sanitized
    """
    if not content:
        return ""
    
    # First, run basic sanitization
    sanitized = sanitize_issue_content(content)
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(sanitized):
            logger.warning("Dangerous pattern detected in issue content - removing")
            sanitized = pattern.sub('[REMOVED FOR SECURITY]', sanitized)
    
    # HTML escape any remaining HTML-like content
    sanitized = html.escape(sanitized, quote=False)
    
    # Additional security checks
    if '\x00' in sanitized:
        raise InputSanitizationError("Null bytes detected in content")
    
    return sanitized


def validate_file_path(file_path: str, allowed_base_paths: List[str] = None) -> bool:
    """
    Validate file path for security (path traversal prevention)
    
    Args:
        file_path: File path to validate
        allowed_base_paths: List of allowed base paths
        
    Returns:
        True if path is safe
        
    Raises:
        PathTraversalError: If path traversal detected
    """
    if not file_path:
        raise PathTraversalError("Empty file path")
    
    # Check for path traversal patterns
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if pattern.search(file_path):
            raise PathTraversalError(f"Path traversal detected: {file_path}")
    
    # Resolve path and check against allowed bases
    try:
        resolved_path = Path(file_path).resolve()
    except (OSError, ValueError) as e:
        raise PathTraversalError(f"Invalid path: {file_path} ({e})")
    
    if allowed_base_paths:
        allowed = False
        for base_path in allowed_base_paths:
            try:
                base_resolved = Path(base_path).resolve()
                if resolved_path.is_relative_to(base_resolved):
                    allowed = True
                    break
            except (OSError, ValueError):
                continue
        
        if not allowed:
            raise PathTraversalError(f"Path outside allowed directories: {file_path}")
    
    return True


def generate_secure_hash(data: str, salt: str = None) -> str:
    """
    Generate secure hash for data
    
    Args:
        data: Data to hash
        salt: Optional salt (will generate if not provided)
        
    Returns:
        Hex-encoded hash
    """
    if salt is None:
        salt = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    hash_input = f"{salt}{data}".encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()


def mask_sensitive_data(text: str, mask_char: str = '*', preserve_chars: int = 4) -> str:
    """
    Mask sensitive data in text while preserving some characters for identification
    
    Args:
        text: Text to mask
        mask_char: Character to use for masking
        preserve_chars: Number of characters to preserve at start/end
        
    Returns:
        Masked text
    """
    if not text or len(text) <= preserve_chars * 2:
        return mask_char * len(text) if text else ""
    
    start = text[:preserve_chars]
    end = text[-preserve_chars:]
    middle_length = len(text) - (preserve_chars * 2)
    
    return f"{start}{mask_char * middle_length}{end}"


# Global instances for easy access (lazy initialization)
_secure_config = None
_secure_subprocess = None

def get_secure_config():
    """Get global SecureConfig instance with lazy initialization"""
    global _secure_config
    if _secure_config is None:
        _secure_config = SecureConfig()
    return _secure_config

def get_secure_subprocess():
    """Get global SecureSubprocess instance with lazy initialization"""
    global _secure_subprocess
    if _secure_subprocess is None:
        _secure_subprocess = SecureSubprocess()
    return _secure_subprocess


# Example usage and testing
if __name__ == "__main__":
    # Test the security utilities
    config = SecureConfig()
    
    # Test token sanitization
    test_text = "git clone https://x-access-token:ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx@github.com/test/repo.git"
    sanitized = config.sanitize_for_logging(test_text)
    print(f"Original: {test_text}")
    print(f"Sanitized: {sanitized}")
    
    # Test repository validation
    print(f"Valid repo: {validate_repo_name('owner/repo')}")
    print(f"Invalid repo: {validate_repo_name('invalid')}")
    
    # Test secure temp directory
    with SecureTempDir() as temp_dir:
        print(f"Created temp dir: {temp_dir}")
        print(f"Permissions: {oct(temp_dir.stat().st_mode)[-3:]}")
    
    print("Security utilities test completed")