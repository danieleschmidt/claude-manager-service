"""
Enhanced Security System for Generation 2 Robustness

This module provides comprehensive security features:
- Secure credential management
- Rate limiting and throttling
- Authentication and authorization
- Input sanitization and validation
- Security headers and CSRF protection
- Audit logging and monitoring
"""

import os
import time
import hashlib
import hmac
import secrets
import base64
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import re

from .enhanced_logger import get_enhanced_logger, log_security_event, SecurityEvent
from .error_handler import AuthenticationError, RateLimitError, EnhancedError


@dataclass
class Credential:
    """Secure credential representation"""
    name: str
    value: str
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_encrypted: bool = False


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    password_min_length: int = 12
    require_mfa: bool = False
    session_timeout_minutes: int = 480  # 8 hours
    rate_limit_requests_per_minute: int = 60
    allowed_origins: List[str] = None
    csrf_protection_enabled: bool = True


@dataclass
class SessionData:
    """User session data"""
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    ip_address: str
    user_agent: str
    permissions: Set[str]
    is_authenticated: bool = True


class CredentialManager:
    """Secure credential management system"""
    
    def __init__(self):
        self.logger = get_enhanced_logger(__name__)
        self.credentials: Dict[str, Credential] = {}
        self._encryption_key = None
        self._lock = threading.Lock()
        
        # Initialize encryption key
        self._initialize_encryption()
        
        # Load credentials from environment
        self._load_environment_credentials()
    
    def _initialize_encryption(self):
        """Initialize encryption key for credential storage"""
        # In production, this should come from a secure key management system
        key_env = os.getenv('CREDENTIAL_ENCRYPTION_KEY')
        if key_env:
            self._encryption_key = base64.b64decode(key_env)
        else:
            # Generate a new key (should be persisted securely in production)
            self._encryption_key = secrets.token_bytes(32)
            self.logger.warning("Using generated encryption key for credentials - should be persisted securely")
    
    def _load_environment_credentials(self):
        """Load credentials from environment variables"""
        # GitHub token
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token:
            self.store_credential('github_token', github_token)
        
        # Database credentials
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            self.store_credential('database_url', db_url)
        
        # Other API keys
        for env_var in os.environ:
            if env_var.endswith('_API_KEY') or env_var.endswith('_SECRET'):
                self.store_credential(env_var.lower(), os.getenv(env_var))
    
    def store_credential(self, name: str, value: str, expires_at: Optional[datetime] = None) -> bool:
        """
        Store a credential securely
        
        Args:
            name: Credential name
            value: Credential value
            expires_at: Optional expiration time
            
        Returns:
            True if stored successfully
        """
        try:
            with self._lock:
                # Encrypt the credential value
                encrypted_value = self._encrypt_value(value)
                
                credential = Credential(
                    name=name,
                    value=encrypted_value,
                    expires_at=expires_at,
                    is_encrypted=True
                )
                
                self.credentials[name] = credential
                
                # Log security event (without the actual value)
                log_security_event(
                    event_type='credential_management',
                    severity='low',
                    description=f'Credential stored: {name}',
                    action='store',
                    result='success'
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store credential {name}: {e}")
            log_security_event(
                event_type='credential_management',
                severity='medium',
                description=f'Failed to store credential: {name}',
                action='store',
                result='failure'
            )
            return False
    
    def get_credential(self, name: str) -> Optional[str]:
        """
        Retrieve a credential securely
        
        Args:
            name: Credential name
            
        Returns:
            Decrypted credential value or None if not found
        """
        try:
            with self._lock:
                if name not in self.credentials:
                    return None
                
                credential = self.credentials[name]
                
                # Check if credential has expired
                if credential.expires_at and datetime.now() > credential.expires_at:
                    del self.credentials[name]
                    log_security_event(
                        event_type='credential_management',
                        severity='low',
                        description=f'Expired credential removed: {name}',
                        action='cleanup',
                        result='success'
                    )
                    return None
                
                # Decrypt the value
                decrypted_value = self._decrypt_value(credential.value)
                
                # Update usage statistics
                credential.last_used = datetime.now()
                credential.usage_count += 1
                
                return decrypted_value
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve credential {name}: {e}")
            log_security_event(
                event_type='credential_management',
                severity='medium',
                description=f'Failed to retrieve credential: {name}',
                action='retrieve',
                result='failure'
            )
            return None
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a credential value"""
        from cryptography.fernet import Fernet
        
        # Create Fernet key from our encryption key
        fernet_key = base64.urlsafe_b64encode(self._encryption_key)
        fernet = Fernet(fernet_key)
        
        encrypted = fernet.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a credential value"""
        from cryptography.fernet import Fernet
        
        # Create Fernet key from our encryption key
        fernet_key = base64.urlsafe_b64encode(self._encryption_key)
        fernet = Fernet(fernet_key)
        
        encrypted_bytes = base64.b64decode(encrypted_value.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def remove_credential(self, name: str) -> bool:
        """Remove a credential"""
        try:
            with self._lock:
                if name in self.credentials:
                    del self.credentials[name]
                    log_security_event(
                        event_type='credential_management',
                        severity='low',
                        description=f'Credential removed: {name}',
                        action='remove',
                        result='success'
                    )
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove credential {name}: {e}")
            return False
    
    def list_credentials(self) -> List[str]:
        """List credential names (not values)"""
        with self._lock:
            return list(self.credentials.keys())
    
    def cleanup_expired(self) -> int:
        """Remove expired credentials and return count removed"""
        removed_count = 0
        current_time = datetime.now()
        
        with self._lock:
            expired_names = [
                name for name, cred in self.credentials.items()
                if cred.expires_at and current_time > cred.expires_at
            ]
            
            for name in expired_names:
                del self.credentials[name]
                removed_count += 1
        
        if removed_count > 0:
            log_security_event(
                event_type='credential_management',
                severity='low',
                description=f'Cleaned up {removed_count} expired credentials',
                action='cleanup',
                result='success'
            )
        
        return removed_count


class RateLimitManager:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = get_enhanced_logger(__name__)
        
        # Request tracking
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_clients: Dict[str, datetime] = {}
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        
        self._lock = threading.Lock()
    
    def check_rate_limit(self, client_id: str, operation: str = "default") -> bool:
        """
        Check if client is within rate limits
        
        Args:
            client_id: Client identifier (IP, user ID, etc.)
            operation: Operation being performed
            
        Returns:
            True if within limits, False if rate limited
        """
        current_time = time.time()
        rate_key = f"{client_id}:{operation}"
        
        with self._lock:
            # Check if client is currently blocked
            if self._is_client_blocked(client_id):
                log_security_event(
                    event_type='rate_limiting',
                    severity='medium',
                    description=f'Blocked client attempted request: {client_id}',
                    source_ip=client_id if self._is_ip_address(client_id) else None,
                    action=operation,
                    result='blocked'
                )
                return False
            
            # Get request history for this client/operation
            history = self.request_history[rate_key]
            
            # Remove old requests (older than 1 minute)
            cutoff_time = current_time - 60
            while history and history[0] < cutoff_time:
                history.popleft()
            
            # Check rate limit
            if len(history) >= self.policy.rate_limit_requests_per_minute:
                # Rate limit exceeded
                self._handle_rate_limit_violation(client_id, operation)
                return False
            
            # Add this request to history
            history.append(current_time)
            
            return True
    
    def _is_client_blocked(self, client_id: str) -> bool:
        """Check if client is currently blocked"""
        if client_id in self.blocked_clients:
            block_time = self.blocked_clients[client_id]
            if datetime.now() - block_time < timedelta(minutes=self.policy.lockout_duration_minutes):
                return True
            else:
                # Unblock client
                del self.blocked_clients[client_id]
        
        return False
    
    def _handle_rate_limit_violation(self, client_id: str, operation: str):
        """Handle rate limit violation"""
        self.suspicious_patterns[client_id] += 1
        
        # Block client after multiple violations
        if self.suspicious_patterns[client_id] >= 3:
            self.blocked_clients[client_id] = datetime.now()
            
            log_security_event(
                event_type='rate_limiting',
                severity='high',
                description=f'Client blocked due to repeated rate limit violations: {client_id}',
                source_ip=client_id if self._is_ip_address(client_id) else None,
                action=operation,
                result='blocked'
            )
        else:
            log_security_event(
                event_type='rate_limiting',
                severity='medium',
                description=f'Rate limit exceeded for client: {client_id}',
                source_ip=client_id if self._is_ip_address(client_id) else None,
                action=operation,
                result='rate_limited'
            )
    
    def _is_ip_address(self, address: str) -> bool:
        """Check if string looks like an IP address"""
        import ipaddress
        try:
            ipaddress.ip_address(address)
            return True
        except ValueError:
            return False
    
    def unblock_client(self, client_id: str):
        """Manually unblock a client"""
        with self._lock:
            if client_id in self.blocked_clients:
                del self.blocked_clients[client_id]
                if client_id in self.suspicious_patterns:
                    del self.suspicious_patterns[client_id]
                
                log_security_event(
                    event_type='rate_limiting',
                    severity='low',
                    description=f'Client manually unblocked: {client_id}',
                    action='unblock',
                    result='success'
                )
    
    def get_client_status(self, client_id: str) -> Dict[str, Any]:
        """Get rate limiting status for client"""
        with self._lock:
            is_blocked = self._is_client_blocked(client_id)
            suspicious_count = self.suspicious_patterns.get(client_id, 0)
            
            # Count recent requests
            current_time = time.time()
            recent_requests = 0
            for key, history in self.request_history.items():
                if key.startswith(f"{client_id}:"):
                    cutoff_time = current_time - 60
                    recent_requests += sum(1 for t in history if t > cutoff_time)
            
            return {
                'is_blocked': is_blocked,
                'suspicious_count': suspicious_count,
                'recent_requests': recent_requests,
                'rate_limit': self.policy.rate_limit_requests_per_minute
            }


class InputValidator:
    """Enhanced input validation and sanitization"""
    
    def __init__(self):
        self.logger = get_enhanced_logger(__name__)
        
        # Dangerous patterns to detect
        self.xss_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*='
        ]
        
        self.sql_injection_patterns = [
            r'(\bunion\b.*\bselect\b)|(\bselect\b.*\bunion\b)',
            r'\b(drop|delete|truncate|alter)\b.*\btable\b',
            r'\binsert\b.*\binto\b',
            r';\s*(drop|delete|truncate|alter)',
            r'(\bor\b|\|\|).*[\'"].*[\'"].*(\bor\b|\|\|)',
            r'(\band\b|&&).*[\'"].*[\'"].*(\band\b|&&)'
        ]
        
        self.command_injection_patterns = [
            r'[;&|`$()]',
            r'\b(cat|ls|pwd|whoami|id|ps|netstat|curl|wget)\b',
            r'\.\./',
            r'/etc/',
            r'/proc/'
        ]
    
    def validate_github_repo(self, repo_name: str) -> bool:
        """Validate GitHub repository name format"""
        if not isinstance(repo_name, str):
            return False
        
        # GitHub repo format: owner/repo
        if not re.match(r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$', repo_name):
            return False
        
        parts = repo_name.split('/')
        if len(parts) != 2:
            return False
        
        owner, repo = parts
        
        # Length limits
        if len(owner) > 39 or len(repo) > 100:
            return False
        
        # Check for suspicious patterns
        if self._contains_suspicious_patterns(repo_name):
            return False
        
        return True
    
    def validate_issue_title(self, title: str) -> bool:
        """Validate GitHub issue title"""
        if not isinstance(title, str):
            return False
        
        # Length check
        if len(title.strip()) == 0 or len(title) > 256:
            return False
        
        # Check for malicious patterns
        if self._contains_xss_patterns(title):
            return False
        
        return True
    
    def validate_issue_body(self, body: str) -> bool:
        """Validate GitHub issue body"""
        if not isinstance(body, str):
            return False
        
        # Length check (GitHub's limit)
        if len(body) > 65536:
            return False
        
        # Check for malicious patterns
        if self._contains_xss_patterns(body):
            return False
        
        if self._contains_command_injection_patterns(body):
            return False
        
        return True
    
    def sanitize_text_input(self, text: str) -> str:
        """Sanitize text input by removing dangerous content"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove XSS patterns
        for pattern in self.xss_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove null bytes and control characters
        text = text.replace('\x00', '')
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        return text.strip()
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check for general suspicious patterns"""
        text_lower = text.lower()
        
        suspicious_keywords = [
            'admin', 'root', 'administrator', 'system', 'config',
            'passwd', 'shadow', 'etc', 'var', 'tmp', 'proc'
        ]
        
        return any(keyword in text_lower for keyword in suspicious_keywords)
    
    def _contains_xss_patterns(self, text: str) -> bool:
        """Check for XSS patterns"""
        for pattern in self.xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                log_security_event(
                    event_type='input_validation',
                    severity='high',
                    description=f'XSS pattern detected in input',
                    action='validate_input',
                    result='blocked'
                )
                return True
        return False
    
    def _contains_sql_injection_patterns(self, text: str) -> bool:
        """Check for SQL injection patterns"""
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                log_security_event(
                    event_type='input_validation',
                    severity='high',
                    description=f'SQL injection pattern detected in input',
                    action='validate_input',
                    result='blocked'
                )
                return True
        return False
    
    def _contains_command_injection_patterns(self, text: str) -> bool:
        """Check for command injection patterns"""
        for pattern in self.command_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                log_security_event(
                    event_type='input_validation',
                    severity='high',
                    description=f'Command injection pattern detected in input',
                    action='validate_input',
                    result='blocked'
                )
                return True
        return False


class SessionManager:
    """Secure session management"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = get_enhanced_logger(__name__)
        self.sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str, permissions: Set[str]) -> str:
        """Create a new user session"""
        session_id = secrets.token_urlsafe(32)
        
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=permissions
        )
        
        with self._lock:
            self.sessions[session_id] = session
        
        log_security_event(
            event_type='authentication',
            severity='low',
            description=f'Session created for user: {user_id}',
            user_id=user_id,
            source_ip=ip_address,
            action='create_session',
            result='success'
        )
        
        return session_id
    
    def validate_session(self, session_id: str, ip_address: str = None) -> Optional[SessionData]:
        """Validate and refresh session"""
        if not session_id:
            return None
        
        with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            current_time = datetime.now()
            
            # Check if session has expired
            if current_time - session.last_accessed > timedelta(minutes=self.policy.session_timeout_minutes):
                del self.sessions[session_id]
                log_security_event(
                    event_type='authentication',
                    severity='low',
                    description=f'Session expired for user: {session.user_id}',
                    user_id=session.user_id,
                    action='session_expired',
                    result='expired'
                )
                return None
            
            # Check IP address if provided (optional for additional security)
            if ip_address and session.ip_address != ip_address:
                log_security_event(
                    event_type='authentication',
                    severity='medium',
                    description=f'Session IP mismatch for user: {session.user_id}',
                    user_id=session.user_id,
                    source_ip=ip_address,
                    action='ip_check',
                    result='mismatch'
                )
                # Optionally invalidate session on IP mismatch
                # del self.sessions[session_id]
                # return None
            
            # Update last accessed time
            session.last_accessed = current_time
            
            return session
    
    def invalidate_session(self, session_id: str):
        """Invalidate a session"""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                del self.sessions[session_id]
                
                log_security_event(
                    event_type='authentication',
                    severity='low',
                    description=f'Session invalidated for user: {session.user_id}',
                    user_id=session.user_id,
                    action='invalidate_session',
                    result='success'
                )
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_count = 0
        
        with self._lock:
            expired_sessions = [
                sid for sid, session in self.sessions.items()
                if current_time - session.last_accessed > timedelta(minutes=self.policy.session_timeout_minutes)
            ]
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
                expired_count += 1
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired sessions")
        
        return expired_count


class SecurityManager:
    """Comprehensive security management system"""
    
    def __init__(self, policy: SecurityPolicy = None):
        self.policy = policy or SecurityPolicy()
        self.logger = get_enhanced_logger(__name__)
        
        # Initialize security components
        self.credential_manager = CredentialManager()
        self.rate_limiter = RateLimitManager(self.policy)
        self.input_validator = InputValidator()
        self.session_manager = SessionManager(self.policy)
        
        # CSRF token management
        self.csrf_tokens: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        
        self.logger.info("SecurityManager initialized with enhanced features")
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        if not self.policy.csrf_protection_enabled:
            return ""
        
        token = secrets.token_urlsafe(32)
        
        with self._lock:
            self.csrf_tokens[f"{session_id}:{token}"] = datetime.now()
        
        return token
    
    def validate_csrf_token(self, session_id: str, token: str) -> bool:
        """Validate CSRF token"""
        if not self.policy.csrf_protection_enabled:
            return True
        
        if not token:
            return False
        
        token_key = f"{session_id}:{token}"
        
        with self._lock:
            if token_key in self.csrf_tokens:
                # Check if token is not expired (1 hour lifetime)
                if datetime.now() - self.csrf_tokens[token_key] < timedelta(hours=1):
                    del self.csrf_tokens[token_key]  # Single use
                    return True
                else:
                    del self.csrf_tokens[token_key]  # Remove expired token
        
        log_security_event(
            event_type='authorization',
            severity='medium',
            description=f'Invalid CSRF token for session: {session_id}',
            action='csrf_validation',
            result='blocked'
        )
        
        return False
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses"""
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
        
        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "connect-src 'self'",
            "font-src 'self'",
            "object-src 'none'",
            "frame-ancestors 'none'"
        ]
        
        headers['Content-Security-Policy'] = '; '.join(csp_directives)
        
        return headers
    
    def check_origin(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if not self.policy.allowed_origins:
            return True  # Allow all origins if not configured
        
        return origin in self.policy.allowed_origins
    
    def audit_security_events(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security audit report"""
        # This would integrate with the enhanced logger to analyze security events
        # For now, return a placeholder structure
        
        return {
            'audit_period_hours': hours,
            'total_events': 0,
            'events_by_type': {},
            'events_by_severity': {},
            'blocked_attempts': 0,
            'failed_authentications': 0,
            'rate_limit_violations': 0,
            'suspicious_activities': []
        }
    
    async def perform_security_check(self, client_id: str, operation: str, input_data: Dict[str, Any]) -> bool:
        """Comprehensive security check"""
        try:
            # Rate limiting check
            if not self.rate_limiter.check_rate_limit(client_id, operation):
                raise RateLimitError(f"Rate limit exceeded for {operation}", operation)
            
            # Input validation
            for key, value in input_data.items():
                if isinstance(value, str):
                    if key in ['title', 'body', 'comment']:
                        if key == 'title' and not self.input_validator.validate_issue_title(value):
                            raise AuthenticationError(f"Invalid {key} format", operation)
                        elif key in ['body', 'comment'] and not self.input_validator.validate_issue_body(value):
                            raise AuthenticationError(f"Invalid {key} format", operation)
                    elif key == 'repo_name':
                        if not self.input_validator.validate_github_repo(value):
                            raise AuthenticationError("Invalid repository name", operation)
            
            return True
            
        except Exception as e:
            log_security_event(
                event_type='authorization',
                severity='medium',
                description=f'Security check failed for operation: {operation}',
                source_ip=client_id if self.rate_limiter._is_ip_address(client_id) else None,
                action=operation,
                result='blocked'
            )
            raise


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager(policy: SecurityPolicy = None) -> SecurityManager:
    """Get global security manager instance"""
    global _security_manager
    
    if _security_manager is None:
        _security_manager = SecurityManager(policy)
    
    return _security_manager


def secure_operation(operation: str):
    """Decorator for securing operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            security_manager = get_security_manager()
            client_id = kwargs.get('client_id', 'unknown')
            
            # Extract input data from kwargs
            input_data = {k: v for k, v in kwargs.items() if k not in ['client_id']}
            
            # Perform security check
            security_manager.perform_security_check(client_id, operation, input_data)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator