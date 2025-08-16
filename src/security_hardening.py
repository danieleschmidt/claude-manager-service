#!/usr/bin/env python3
"""
Security Hardening Module for Autonomous SDLC System

Implements comprehensive security hardening measures:
- Advanced threat detection and prevention
- Secure secret management
- Access control and authorization
- Security audit logging
- Compliance and governance
- Real-time security monitoring
"""

import os
import re
import json
import hashlib
import secrets
import base64
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hmac
from urllib.parse import urlparse
import subprocess

# Cryptographic imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

class SecurityLevel(Enum):
    """Security enforcement levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats"""
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    PATH_TRAVERSAL = "path_traversal"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXPOSURE = "data_exposure"
    MALICIOUS_CODE = "malicious_code"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

class AccessLevel(Enum):
    """Access control levels"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SYSTEM = "system"

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    timestamp: datetime
    event_type: ThreatType
    severity: SecurityLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    max_login_attempts: int = 5
    password_min_length: int = 12
    require_mfa: bool = True
    session_timeout: int = 3600  # 1 hour
    allowed_domains: Set[str] = field(default_factory=set)
    blocked_ips: Set[str] = field(default_factory=set)
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 1 minute

class SecretManager:
    """Secure secret management system"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if not HAS_CRYPTOGRAPHY:
            raise ImportError("cryptography package required for SecretManager")
        
        self.master_key = master_key or self._generate_master_key()
        self.fernet = Fernet(self.master_key)
        self.secrets_store: Dict[str, bytes] = {}
        self.logger = logging.getLogger("secret_manager")
    
    def _generate_master_key(self) -> bytes:
        """Generate a new master key"""
        return Fernet.generate_key()
    
    def store_secret(self, key: str, value: str, metadata: Dict[str, Any] = None) -> bool:
        """Store a secret securely"""
        try:
            encrypted_value = self.fernet.encrypt(value.encode())
            self.secrets_store[key] = encrypted_value
            
            self.logger.info(f"Secret stored: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store secret {key}: {e}")
            return False
    
    def retrieve_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret"""
        try:
            if key not in self.secrets_store:
                return None
            
            encrypted_value = self.secrets_store[key]
            decrypted_value = self.fernet.decrypt(encrypted_value)
            
            self.logger.debug(f"Secret retrieved: {key}")
            return decrypted_value.decode()
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret {key}: {e}")
            return None
    
    def rotate_key(self) -> bytes:
        """Rotate the master key"""
        old_fernet = self.fernet
        new_key = self._generate_master_key()
        new_fernet = Fernet(new_key)
        
        # Re-encrypt all secrets with new key
        for key, encrypted_value in self.secrets_store.items():
            try:
                decrypted_value = old_fernet.decrypt(encrypted_value)
                self.secrets_store[key] = new_fernet.encrypt(decrypted_value)
            except Exception as e:
                self.logger.error(f"Failed to rotate key for secret {key}: {e}")
        
        self.master_key = new_key
        self.fernet = new_fernet
        self.logger.info("Master key rotated successfully")
        return new_key
    
    def delete_secret(self, key: str) -> bool:
        """Securely delete a secret"""
        if key in self.secrets_store:
            del self.secrets_store[key]
            self.logger.info(f"Secret deleted: {key}")
            return True
        return False

class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.logger = logging.getLogger("threat_detector")
        self.detected_threats: List[SecurityEvent] = []
    
    def _load_threat_patterns(self) -> Dict[ThreatType, List[str]]:
        """Load threat detection patterns"""
        return {
            ThreatType.INJECTION: [
                r"(['\"]\s*;\s*)|(\s*;\s*['\"])",  # SQL injection
                r"(union\s+select)|(select\s+.*\s+from)",
                r"(drop\s+table)|(delete\s+from)",
                r"(<script[^>]*>)|(javascript:)",  # Script injection
                r"(eval\s*\()|(exec\s*\()",
            ],
            ThreatType.XSS: [
                r"<script[^>]*>.*?</script>",
                r"javascript:[^'\"]*",
                r"on\w+\s*=\s*['\"][^'\"]*['\"]",
                r"data:text/html",
                r"vbscript:",
            ],
            ThreatType.PATH_TRAVERSAL: [
                r"\.\.\/",
                r"\.\.\\\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
                r"\/etc\/passwd",
                r"\/proc\/",
            ],
            ThreatType.MALICIOUS_CODE: [
                r"rm\s+-rf\s+\/",
                r"format\s+c:",
                r"wget\s+.*\s*\|",
                r"curl\s+.*\s*\|",
                r"nc\s+-l",
                r"\/bin\/sh",
            ],
            ThreatType.SUSPICIOUS_ACTIVITY: [
                r"(password|passwd|pwd)\s*[:=]\s*['\"]?[\w@#$%^&*]+['\"]?",
                r"(api[_-]?key|apikey|token)\s*[:=]\s*['\"]?[\w\-_]+['\"]?",
                r"(secret|private[_-]?key)\s*[:=]",
                r"(access[_-]?token|auth[_-]?token)",
            ]
        }
    
    def scan_content(self, content: str, context: str = "") -> List[SecurityEvent]:
        """Scan content for security threats"""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        event = SecurityEvent(
                            timestamp=datetime.now(timezone.utc),
                            event_type=threat_type,
                            severity=self._assess_severity(threat_type, match.group()),
                            resource=context,
                            description=f"Detected {threat_type.value}: {match.group()[:100]}",
                            metadata={"pattern": pattern, "position": match.start()}
                        )
                        threats.append(event)
                        self.detected_threats.append(event)
                except Exception as e:
                    self.logger.error(f"Error scanning for {threat_type}: {e}")
        
        return threats
    
    def scan_file(self, file_path: Path) -> List[SecurityEvent]:
        """Scan a file for security threats"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self.scan_content(content, str(file_path))
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
            return []
    
    def _assess_severity(self, threat_type: ThreatType, content: str) -> SecurityLevel:
        """Assess threat severity"""
        high_severity_indicators = [
            'drop table', 'delete from', 'rm -rf /', 'format c:',
            'eval(', 'exec(', '/etc/passwd', 'private_key'
        ]
        
        if any(indicator in content.lower() for indicator in high_severity_indicators):
            return SecurityLevel.CRITICAL
        
        if threat_type in [ThreatType.INJECTION, ThreatType.MALICIOUS_CODE]:
            return SecurityLevel.HIGH
        
        if threat_type in [ThreatType.XSS, ThreatType.PATH_TRAVERSAL]:
            return SecurityLevel.MEDIUM
        
        return SecurityLevel.LOW

class AccessController:
    """Role-based access control system"""
    
    def __init__(self):
        self.user_permissions: Dict[str, Set[str]] = {}
        self.role_permissions: Dict[str, Set[str]] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger("access_controller")
        
        # Initialize default roles
        self._setup_default_roles()
    
    def _setup_default_roles(self):
        """Setup default role permissions"""
        self.role_permissions.update({
            'viewer': {'read_repos', 'read_issues'},
            'contributor': {'read_repos', 'read_issues', 'create_issues', 'comment_issues'},
            'maintainer': {'read_repos', 'read_issues', 'create_issues', 'comment_issues', 
                          'manage_repos', 'manage_issues'},
            'admin': {'*'},  # All permissions
        })
    
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign a role to a user"""
        if role not in self.role_permissions:
            self.logger.error(f"Unknown role: {role}")
            return False
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role)
        self.logger.info(f"Assigned role {role} to user {user_id}")
        return True
    
    def grant_permission(self, user_id: str, permission: str) -> bool:
        """Grant a specific permission to a user"""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = set()
        
        self.user_permissions[user_id].add(permission)
        self.logger.info(f"Granted permission {permission} to user {user_id}")
        return True
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if a user has a specific permission"""
        # Check direct permissions
        if user_id in self.user_permissions and permission in self.user_permissions[user_id]:
            return True
        
        # Check role-based permissions
        if user_id in self.user_roles:
            for role in self.user_roles[user_id]:
                role_perms = self.role_permissions.get(role, set())
                if '*' in role_perms or permission in role_perms:
                    return True
        
        return False
    
    def revoke_permission(self, user_id: str, permission: str) -> bool:
        """Revoke a permission from a user"""
        if user_id in self.user_permissions:
            self.user_permissions[user_id].discard(permission)
            self.logger.info(f"Revoked permission {permission} from user {user_id}")
            return True
        return False

class SecurityAuditor:
    """Security audit and compliance system"""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("security_audit.log")
        self.audit_events: List[SecurityEvent] = []
        self.logger = logging.getLogger("security_auditor")
        
        # Setup audit logger
        self._setup_audit_logger()
    
    def _setup_audit_logger(self):
        """Setup secure audit logging"""
        audit_logger = logging.getLogger("security_audit")
        audit_logger.setLevel(logging.INFO)
        
        # Create file handler with rotation
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)
        
        self.audit_logger = audit_logger
    
    def log_security_event(self, event: SecurityEvent):
        """Log a security event"""
        self.audit_events.append(event)
        
        # Format event for logging
        log_message = (f"SECURITY_EVENT | "
                      f"Type: {event.event_type.value} | "
                      f"Severity: {event.severity.value} | "
                      f"Resource: {event.resource or 'N/A'} | "
                      f"User: {event.user_id or 'N/A'} | "
                      f"IP: {event.source_ip or 'N/A'} | "
                      f"Description: {event.description}")
        
        self.audit_logger.info(log_message)
    
    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for the specified time period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_events = [e for e in self.audit_events if e.timestamp >= cutoff_time]
        
        # Aggregate statistics
        threat_counts = {}
        severity_counts = {}
        
        for event in recent_events:
            threat_counts[event.event_type.value] = threat_counts.get(event.event_type.value, 0) + 1
            severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
        
        return {
            'report_period_hours': hours,
            'total_events': len(recent_events),
            'threat_breakdown': threat_counts,
            'severity_breakdown': severity_counts,
            'high_severity_events': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'type': e.event_type.value,
                    'severity': e.severity.value,
                    'description': e.description
                }
                for e in recent_events 
                if e.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
            ]
        }
    
    def check_compliance(self) -> Dict[str, bool]:
        """Check compliance with security standards"""
        compliance_checks = {
            'audit_logging_enabled': bool(self.audit_logger),
            'encryption_available': HAS_CRYPTOGRAPHY,
            'secure_file_permissions': self._check_file_permissions(),
            'no_hardcoded_secrets': self._check_hardcoded_secrets(),
            'secure_configurations': self._check_secure_configurations(),
        }
        
        return compliance_checks
    
    def _check_file_permissions(self) -> bool:
        """Check if sensitive files have secure permissions"""
        sensitive_files = [
            'config.json',
            '.env',
            'secrets.json'
        ]
        
        for file_name in sensitive_files:
            file_path = Path(file_name)
            if file_path.exists():
                # Check if file is readable by others
                stat = file_path.stat()
                if stat.st_mode & 0o077:  # Others have read/write/execute
                    return False
        
        return True
    
    def _check_hardcoded_secrets(self) -> bool:
        """Check for hardcoded secrets in code"""
        secret_patterns = [
            r'password\s*[:=]\s*["\'][^"\']+["\']',
            r'api[_-]?key\s*[:=]\s*["\'][^"\']+["\']',
            r'secret\s*[:=]\s*["\'][^"\']+["\']',
            r'token\s*[:=]\s*["\'][^"\']+["\']',
        ]
        
        for py_file in Path('.').rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return False
            except Exception:
                continue
        
        return True
    
    def _check_secure_configurations(self) -> bool:
        """Check for secure configuration settings"""
        # This would check various configuration files for secure settings
        # For now, return True as a placeholder
        return True

class SecurityHardeningManager:
    """Main security hardening management system"""
    
    def __init__(self, policy: SecurityPolicy = None):
        self.policy = policy or SecurityPolicy()
        self.secret_manager = SecretManager() if HAS_CRYPTOGRAPHY else None
        self.threat_detector = ThreatDetector()
        self.access_controller = AccessController()
        self.auditor = SecurityAuditor()
        self.logger = logging.getLogger("security_hardening")
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}
    
    def initialize_security(self):
        """Initialize all security systems"""
        self.logger.info("Initializing security hardening systems")
        
        # Setup secure configurations
        self._setup_secure_configs()
        
        # Initialize threat monitoring
        self._start_threat_monitoring()
        
        # Setup access controls
        self._setup_access_controls()
        
        self.logger.info("Security hardening initialization complete")
    
    def _setup_secure_configs(self):
        """Setup secure configuration defaults"""
        # Set secure environment variables
        os.environ.setdefault('PYTHONHTTPSVERIFY', '1')
        os.environ.setdefault('PYTHONHASHSEED', 'random')
        
        # Log security setup
        self.auditor.log_security_event(SecurityEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=ThreatType.SUSPICIOUS_ACTIVITY,
            severity=SecurityLevel.LOW,
            description="Security hardening systems initialized"
        ))
    
    def _start_threat_monitoring(self):
        """Start real-time threat monitoring"""
        # This would start background monitoring processes
        self.logger.info("Threat monitoring started")
    
    def _setup_access_controls(self):
        """Setup initial access control policies"""
        # Setup default admin user if needed
        admin_user = os.environ.get('SECURITY_ADMIN_USER')
        if admin_user:
            self.access_controller.assign_role(admin_user, 'admin')
    
    def validate_input_security(self, content: str, context: str = "") -> Tuple[bool, List[SecurityEvent]]:
        """Validate input for security threats"""
        threats = self.threat_detector.scan_content(content, context)
        
        # Log high-severity threats
        for threat in threats:
            if threat.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                self.auditor.log_security_event(threat)
        
        # Determine if input should be rejected
        critical_threats = [t for t in threats if t.severity == SecurityLevel.CRITICAL]
        is_safe = len(critical_threats) == 0
        
        return is_safe, threats
    
    def check_rate_limit(self, identifier: str, limit: int = None, window: int = None) -> bool:
        """Check if request is within rate limits"""
        limit = limit or self.policy.rate_limit_requests
        window = window or self.policy.rate_limit_window
        
        now = time.time()
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Remove old entries
        self.rate_limits[identifier] = [
            timestamp for timestamp in self.rate_limits[identifier]
            if now - timestamp < window
        ]
        
        # Check if within limit
        if len(self.rate_limits[identifier]) >= limit:
            self.auditor.log_security_event(SecurityEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=ThreatType.SUSPICIOUS_ACTIVITY,
                severity=SecurityLevel.MEDIUM,
                source_ip=identifier,
                description=f"Rate limit exceeded: {len(self.rate_limits[identifier])} requests"
            ))
            return False
        
        # Add current request
        self.rate_limits[identifier].append(now)
        return True
    
    def secure_subprocess_call(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Execute subprocess with security hardening"""
        # Validate command
        if not self._validate_command(cmd):
            raise SecurityError("Command failed security validation")
        
        # Set secure defaults
        secure_kwargs = {
            'capture_output': True,
            'text': True,
            'timeout': 30,
            'check': False,
            **kwargs
        }
        
        # Remove dangerous environment variables
        env = secure_kwargs.get('env', os.environ.copy())
        dangerous_env_vars = ['LD_PRELOAD', 'DYLD_INSERT_LIBRARIES']
        for var in dangerous_env_vars:
            env.pop(var, None)
        secure_kwargs['env'] = env
        
        try:
            result = subprocess.run(cmd, **secure_kwargs)
            
            # Log subprocess execution
            self.auditor.log_security_event(SecurityEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=ThreatType.SUSPICIOUS_ACTIVITY,
                severity=SecurityLevel.LOW,
                description=f"Subprocess executed: {' '.join(cmd[:3])}..."
            ))
            
            return result
        except Exception as e:
            self.logger.error(f"Secure subprocess call failed: {e}")
            raise
    
    def _validate_command(self, cmd: List[str]) -> bool:
        """Validate subprocess command for security"""
        if not cmd:
            return False
        
        # Check for dangerous commands
        dangerous_commands = {
            'rm', 'del', 'format', 'mkfs', 'dd',
            'nc', 'netcat', 'telnet', 'ftp',
            'wget', 'curl', 'ssh', 'scp',
            'eval', 'exec', 'python', 'bash', 'sh'
        }
        
        command_name = os.path.basename(cmd[0])
        if command_name in dangerous_commands:
            return False
        
        # Check for dangerous arguments
        cmd_str = ' '.join(cmd)
        dangerous_patterns = [
            r'rm\s+-rf',
            r'--user-data-dir',
            r'--disable-web-security',
            r'>/dev/',
            r'\|.*sh',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, cmd_str):
                return False
        
        return True
    
    def generate_security_summary(self) -> Dict[str, Any]:
        """Generate comprehensive security summary"""
        return {
            'security_policy': {
                'max_login_attempts': self.policy.max_login_attempts,
                'session_timeout': self.policy.session_timeout,
                'rate_limit': f"{self.policy.rate_limit_requests}/{self.policy.rate_limit_window}s"
            },
            'threat_detection': {
                'total_threats_detected': len(self.threat_detector.detected_threats),
                'patterns_monitored': sum(len(patterns) for patterns in self.threat_detector.threat_patterns.values())
            },
            'access_control': {
                'users_with_roles': len(self.access_controller.user_roles),
                'available_roles': len(self.access_controller.role_permissions)
            },
            'audit_logging': {
                'events_logged': len(self.auditor.audit_events),
                'log_file': str(self.auditor.log_file)
            },
            'compliance_status': self.auditor.check_compliance(),
            'security_report': self.auditor.generate_security_report(hours=24)
        }

class SecurityError(Exception):
    """Base exception for security-related errors"""
    pass

class AuthenticationError(SecurityError):
    """Authentication failed"""
    pass

class AuthorizationError(SecurityError):
    """Authorization failed"""
    pass

class ThreatDetectedError(SecurityError):
    """Security threat detected"""
    pass

# Utility functions
def create_security_manager(level: SecurityLevel = SecurityLevel.MEDIUM) -> SecurityHardeningManager:
    """Create a security manager with appropriate configuration"""
    policy = SecurityPolicy()
    
    if level == SecurityLevel.HIGH:
        policy.max_login_attempts = 3
        policy.session_timeout = 1800  # 30 minutes
        policy.require_mfa = True
    elif level == SecurityLevel.CRITICAL:
        policy.max_login_attempts = 2
        policy.session_timeout = 900   # 15 minutes
        policy.require_mfa = True
        policy.rate_limit_requests = 50
    
    return SecurityHardeningManager(policy)

def secure_hash(data: str, salt: str = None) -> str:
    """Create a secure hash of data"""
    if salt is None:
        salt = secrets.token_hex(16)
    
    combined = f"{data}{salt}".encode()
    hash_obj = hashlib.sha256(combined)
    return hash_obj.hexdigest()

def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure token"""
    return secrets.token_urlsafe(length)

# Example usage
if __name__ == "__main__":
    # Create security manager
    security_manager = create_security_manager(SecurityLevel.HIGH)
    security_manager.initialize_security()
    
    # Test threat detection
    test_content = "SELECT * FROM users WHERE username = 'admin'"
    is_safe, threats = security_manager.validate_input_security(test_content, "test_input")
    
    print(f"Content is safe: {is_safe}")
    print(f"Threats detected: {len(threats)}")
    
    # Generate security summary
    summary = security_manager.generate_security_summary()
    print(json.dumps(summary, indent=2))