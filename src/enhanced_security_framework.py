#!/usr/bin/env python3
"""
ENHANCED SECURITY FRAMEWORK - Generation 2
Comprehensive security measures, validation, and threat protection
"""

import hashlib
import hmac
import os
import re
import secrets
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from src.logger import get_logger

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security level classifications"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    event_type: str
    threat_level: ThreatLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved: bool = False


@dataclass
class SecurityConfig:
    """Security configuration parameters"""
    token_expiry_hours: int = 24
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    password_min_length: int = 12
    require_2fa: bool = False
    encryption_enabled: bool = True
    audit_logging_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    allowed_origins: List[str] = field(default_factory=list)


class EnhancedTokenValidator:
    """Enhanced token validation with multiple token types"""
    
    GITHUB_TOKEN_PATTERNS = {
        'personal_access_token': re.compile(r'^ghp_[A-Za-z0-9_]{36}$'),
        'oauth_token': re.compile(r'^gho_[A-Za-z0-9_]{36}$'),
        'app_token': re.compile(r'^(ghs_[A-Za-z0-9_]{36}|ghr_[A-Za-z0-9_]{76})$'),
        'refresh_token': re.compile(r'^ghr_[A-Za-z0-9_]{76}$')
    }
    
    def __init__(self):
        self.revoked_tokens: Set[str] = set()
        self.token_metadata: Dict[str, Dict[str, Any]] = {}
    
    def validate_github_token(self, token: str) -> Dict[str, Any]:
        """Validate GitHub token format and properties"""
        if not token or len(token.strip()) == 0:
            return {"valid": False, "error": "Empty token"}
        
        token = token.strip()
        
        # Check if token is revoked
        if self._is_token_revoked(token):
            return {"valid": False, "error": "Token has been revoked"}
        
        # Validate format
        token_type = None
        for type_name, pattern in self.GITHUB_TOKEN_PATTERNS.items():
            if pattern.match(token):
                token_type = type_name
                break
        
        if not token_type:
            return {"valid": False, "error": "Invalid GitHub token format"}
        
        # Additional security checks
        security_score = self._calculate_token_security_score(token)
        
        # Check for common weak patterns
        weak_patterns = self._check_weak_patterns(token)
        if weak_patterns:
            return {
                "valid": False, 
                "error": f"Token contains weak patterns: {', '.join(weak_patterns)}"
            }
        
        return {
            "valid": True,
            "token_type": token_type,
            "security_score": security_score,
            "expires_soon": False,  # Would need API call to determine actual expiry
            "scopes": []  # Would need API call to determine scopes
        }
    
    def _is_token_revoked(self, token: str) -> bool:
        """Check if token is in revocation list"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return token_hash in self.revoked_tokens
    
    def _calculate_token_security_score(self, token: str) -> float:
        """Calculate security score for token (0-100)"""
        score = 100.0
        
        # Length check
        if len(token) < 40:
            score -= 20
        
        # Character diversity
        has_upper = any(c.isupper() for c in token)
        has_lower = any(c.islower() for c in token)
        has_digit = any(c.isdigit() for c in token)
        has_special = any(c in '_-' for c in token)
        
        diversity_score = sum([has_upper, has_lower, has_digit, has_special])
        if diversity_score < 3:
            score -= (4 - diversity_score) * 10
        
        return max(score, 0.0)
    
    def _check_weak_patterns(self, token: str) -> List[str]:
        """Check for weak patterns in token"""
        weak_patterns = []
        
        # Check for repeated characters
        if re.search(r'(.)\1{3,}', token):
            weak_patterns.append("repeated_characters")
        
        # Check for sequential patterns
        if re.search(r'(abc|123|xyz)', token.lower()):
            weak_patterns.append("sequential_patterns")
        
        return weak_patterns
    
    def revoke_token(self, token: str) -> None:
        """Add token to revocation list"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self.revoked_tokens.add(token_hash)
        logger.info(f"Token revoked (hash: {token_hash[:8]}...)")


class SecureInputValidator:
    """Comprehensive input validation and sanitization"""
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        re.compile(r'<script[^>]*>', re.IGNORECASE),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'(\||;|&|`|\$\(|\${)', re.IGNORECASE),
        re.compile(r'\.\.\/'),  # Path traversal
        re.compile(r'\/etc\/passwd', re.IGNORECASE),
        re.compile(r'cmd\.exe', re.IGNORECASE),
        re.compile(r'powershell', re.IGNORECASE),
    ]
    
    # File path validation
    ALLOWED_FILE_EXTENSIONS = {'.py', '.js', '.md', '.json', '.yaml', '.yml', '.txt', '.csv'}
    BLOCKED_PATHS = {'/etc/', '/proc/', '/sys/', 'C:\\Windows\\', 'C:\\System32\\'}
    
    def validate_and_sanitize_input(self, input_data: str, input_type: str = "general") -> Dict[str, Any]:
        """Validate and sanitize user input"""
        if not isinstance(input_data, str):
            return {"valid": False, "error": "Input must be string", "sanitized": ""}
        
        original_length = len(input_data)
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(input_data):
                logger.warning(f"Dangerous pattern detected: {pattern.pattern}")
                return {
                    "valid": False,
                    "error": "Input contains potentially dangerous content",
                    "sanitized": ""
                }
        
        # Sanitize based on input type
        if input_type == "filename":
            sanitized = self._sanitize_filename(input_data)
        elif input_type == "filepath":
            sanitized = self._sanitize_filepath(input_data)
        elif input_type == "email":
            sanitized = self._sanitize_email(input_data)
        elif input_type == "url":
            sanitized = self._sanitize_url(input_data)
        else:
            sanitized = self._sanitize_general(input_data)
        
        # Length validation
        if len(sanitized) > 10000:  # Prevent DoS via large inputs
            return {
                "valid": False,
                "error": "Input too long",
                "sanitized": sanitized[:1000]
            }
        
        return {
            "valid": True,
            "sanitized": sanitized,
            "original_length": original_length,
            "sanitized_length": len(sanitized)
        }
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename input"""
        # Remove path separators and special characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        sanitized = sanitized.strip('. ')
        
        # Check extension
        file_path = Path(sanitized)
        if file_path.suffix.lower() not in self.ALLOWED_FILE_EXTENSIONS:
            logger.warning(f"Potentially unsafe file extension: {file_path.suffix}")
        
        return sanitized
    
    def _sanitize_filepath(self, filepath: str) -> str:
        """Sanitize file path input"""
        # Normalize path
        try:
            sanitized = str(Path(filepath).resolve())
        except Exception:
            return ""
        
        # Check for blocked paths
        for blocked in self.BLOCKED_PATHS:
            if blocked.lower() in sanitized.lower():
                logger.warning(f"Blocked path accessed: {sanitized}")
                return ""
        
        return sanitized
    
    def _sanitize_email(self, email: str) -> str:
        """Sanitize email input"""
        email = email.strip().lower()
        
        # Basic email validation
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not email_pattern.match(email):
            return ""
        
        return email
    
    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL input"""
        url = url.strip()
        
        # Allow only HTTP/HTTPS
        if not url.startswith(('http://', 'https://')):
            return ""
        
        # Block dangerous protocols
        dangerous_protocols = ['file://', 'javascript:', 'data:', 'vbscript:']
        for protocol in dangerous_protocols:
            if url.lower().startswith(protocol):
                return ""
        
        return url
    
    def _sanitize_general(self, text: str) -> str:
        """General text sanitization"""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # HTML encode dangerous characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
        
        return text


class SecureDataEncryption:
    """Secure data encryption and decryption"""
    
    def __init__(self, key: Optional[bytes] = None):
        if key:
            self.fernet = Fernet(key)
        else:
            self.fernet = Fernet(Fernet.generate_key())
        self.key_created_at = datetime.now(timezone.utc)
    
    @classmethod
    def derive_key_from_password(cls, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def is_key_expired(self, max_age_days: int = 90) -> bool:
        """Check if encryption key should be rotated"""
        age = datetime.now(timezone.utc) - self.key_created_at
        return age.days > max_age_days


class SecurityAuditLogger:
    """Security event logging and auditing"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = Path(log_file)
        self.security_events: List[SecurityEvent] = []
        self.ensure_log_file_exists()
    
    def ensure_log_file_exists(self):
        """Ensure log file exists with proper permissions"""
        if not self.log_file.exists():
            self.log_file.touch(mode=0o600)  # Owner read/write only
    
    def log_security_event(self, event: SecurityEvent):
        """Log a security event"""
        self.security_events.append(event)
        
        # Write to audit log
        log_entry = {
            "timestamp": event.timestamp,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "threat_level": event.threat_level.value,
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "details": event.details
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")
        except Exception as e:
            logger.error(f"Failed to write security audit log: {e}")
    
    def log_authentication_attempt(self, user_id: str, success: bool, source_ip: Optional[str] = None):
        """Log authentication attempt"""
        event = SecurityEvent(
            event_id=secrets.token_hex(8),
            event_type="authentication_attempt",
            threat_level=ThreatLevel.LOW if success else ThreatLevel.MEDIUM,
            source_ip=source_ip,
            user_id=user_id,
            details={"success": success}
        )
        self.log_security_event(event)
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any], source_ip: Optional[str] = None):
        """Log security violation"""
        event = SecurityEvent(
            event_id=secrets.token_hex(8),
            event_type="security_violation",
            threat_level=ThreatLevel.HIGH,
            source_ip=source_ip,
            details={"violation_type": violation_type, **details}
        )
        self.log_security_event(event)
    
    def get_recent_events(self, hours: int = 24) -> List[SecurityEvent]:
        """Get recent security events"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            event for event in self.security_events
            if datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')) > cutoff_time
        ]


class EnhancedSecurityFramework:
    """Main security framework coordinator"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.token_validator = EnhancedTokenValidator()
        self.input_validator = SecureInputValidator()
        self.audit_logger = SecurityAuditLogger()
        self.encryption = SecureDataEncryption()
        self.failed_attempts: Dict[str, List[datetime]] = {}
    
    def validate_request_security(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive request security validation"""
        security_result = {
            "valid": True,
            "violations": [],
            "sanitized_data": {},
            "risk_score": 0.0
        }
        
        # Validate each field
        for field, value in request_data.items():
            if isinstance(value, str):
                validation = self.input_validator.validate_and_sanitize_input(value, field)
                if not validation["valid"]:
                    security_result["valid"] = False
                    security_result["violations"].append(f"{field}: {validation['error']}")
                    security_result["risk_score"] += 25.0
                else:
                    security_result["sanitized_data"][field] = validation["sanitized"]
            else:
                security_result["sanitized_data"][field] = value
        
        # Check for rate limiting
        source_ip = request_data.get("source_ip", "unknown")
        if not self._check_rate_limit(source_ip):
            security_result["valid"] = False
            security_result["violations"].append("Rate limit exceeded")
            security_result["risk_score"] += 50.0
        
        # Normalize risk score
        security_result["risk_score"] = min(security_result["risk_score"], 100.0)
        
        return security_result
    
    def _check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits"""
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(minutes=1)
        
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        # Clean old attempts
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff_time
        ]
        
        # Check limit
        if len(self.failed_attempts[identifier]) >= self.config.rate_limit_requests_per_minute:
            return False
        
        self.failed_attempts[identifier].append(current_time)
        return True
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        recent_events = self.audit_logger.get_recent_events(24)
        
        # Categorize events
        event_categories = {}
        threat_levels = {level.value: 0 for level in ThreatLevel}
        
        for event in recent_events:
            event_type = event.event_type
            if event_type not in event_categories:
                event_categories[event_type] = 0
            event_categories[event_type] += 1
            threat_levels[event.threat_level.value] += 1
        
        # Calculate security score
        total_events = len(recent_events)
        critical_events = threat_levels[ThreatLevel.CRITICAL.value]
        high_events = threat_levels[ThreatLevel.HIGH.value]
        
        # Base security score calculation
        if total_events == 0:
            security_score = 100.0
        else:
            # Penalize based on threat levels
            penalty = (critical_events * 50) + (high_events * 25)
            security_score = max(100.0 - (penalty / total_events * 100), 0.0)
        
        return {
            "security_score": security_score,
            "total_events_24h": total_events,
            "event_categories": event_categories,
            "threat_levels": threat_levels,
            "revoked_tokens": len(self.token_validator.revoked_tokens),
            "encryption_key_age_days": (datetime.now(timezone.utc) - self.encryption.key_created_at).days,
            "rate_limited_ips": len([k for k, v in self.failed_attempts.items() if len(v) >= self.config.rate_limit_requests_per_minute]),
            "report_generated_at": datetime.now(timezone.utc).isoformat()
        }


# Factory function
def create_security_framework(config: Optional[SecurityConfig] = None) -> EnhancedSecurityFramework:
    """Create enhanced security framework"""
    return EnhancedSecurityFramework(config)


# Demo and testing
async def run_security_framework_demo():
    """Demonstration of security framework"""
    logger.info("Starting enhanced security framework demo")
    
    # Create security framework
    security_config = SecurityConfig(
        max_login_attempts=3,
        rate_limit_requests_per_minute=50
    )
    
    framework = create_security_framework(security_config)
    
    # Test token validation
    test_tokens = [
        "ghp_1234567890123456789012345678901234567890",  # Invalid format
        "ghp_abcdefghijk1234567890abcdefghijk1234567890",  # Valid format
        "malicious_token",  # Invalid
    ]
    
    for token in test_tokens:
        result = framework.token_validator.validate_github_token(token)
        logger.info(f"Token validation: {result['valid']} - {result.get('error', 'OK')}")
    
    # Test input validation
    test_inputs = [
        ("test_file.py", "filename"),
        ("<script>alert('xss')</script>", "general"),
        ("../../../etc/passwd", "filepath"),
        ("user@example.com", "email")
    ]
    
    for input_data, input_type in test_inputs:
        result = framework.input_validator.validate_and_sanitize_input(input_data, input_type)
        logger.info(f"Input validation ({input_type}): {result['valid']}")
    
    # Test request validation
    test_request = {
        "username": "test_user",
        "filename": "test.py",
        "content": "print('hello world')",
        "source_ip": "127.0.0.1"
    }
    
    security_result = framework.validate_request_security(test_request)
    logger.info(f"Request security: {security_result['valid']} (risk: {security_result['risk_score']:.1f}%)")
    
    # Generate security report
    report = framework.generate_security_report()
    logger.info(f"Security score: {report['security_score']:.1f}%")
    
    logger.info("Enhanced security framework demo completed")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_security_framework_demo())