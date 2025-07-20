"""
Security utilities for Claude Manager Service

This module provides secure handling of sensitive data like tokens,
environment variables, and subprocess execution.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from logger import get_logger

logger = get_logger(__name__)


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
    
    # Limit length to prevent DoS
    max_length = 50000  # 50KB limit
    if len(sanitized) > max_length:
        logger = get_logger(f"{__name__}.sanitize_issue_content")
        logger.warning(f"Issue content truncated from {len(sanitized)} to {max_length} characters")
        sanitized = sanitized[:max_length] + "\n\n[Content truncated for security]"
    
    return sanitized


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