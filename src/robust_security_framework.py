#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - ROBUST SECURITY FRAMEWORK
Comprehensive security, validation, and resilience implementation
"""

import asyncio
import hashlib
import hmac
import json
import re
import secrets
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import wraps

import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog


class SecurityLevel(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message: str, field: str = None, code: str = None):
        super().__init__(message)
        self.field = field
        self.code = code
        self.timestamp = datetime.now(timezone.utc)


class SecurityViolation(Exception):
    """Security violation exception"""
    def __init__(self, message: str, severity: str = "high", action: str = None):
        super().__init__(message)
        self.severity = severity
        self.action = action
        self.timestamp = datetime.now(timezone.utc)


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    session_id: str
    permissions: List[str]
    security_level: SecurityLevel
    ip_address: str
    user_agent: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if security context is expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions or "admin" in self.permissions


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    max_requests: int
    time_window: int  # seconds
    burst_allowance: int = 0
    
    
@dataclass
class SecurityMetrics:
    """Security monitoring metrics"""
    failed_authentications: int = 0
    security_violations: int = 0
    rate_limit_hits: int = 0
    suspicious_activities: int = 0
    last_security_scan: Optional[datetime] = None
    vulnerability_count: int = 0


class SecureValidator:
    """Comprehensive input validation with security focus"""
    
    # Dangerous patterns that should be blocked
    DANGEROUS_PATTERNS = [
        r'<script.*?>.*?</script>',  # XSS
        r'javascript:',              # JavaScript injection
        r'on\w+\s*=',               # Event handler injection
        r'(\b(union|select|insert|update|delete|drop|create|alter)\b)',  # SQL injection
        r'\.\./',                    # Path traversal
        r'\\x[0-9a-fA-F]{2}',       # Hex encoding
        r'%[0-9a-fA-F]{2}',         # URL encoding
        r'eval\s*\(',               # Code injection
        r'exec\s*\(',               # Code execution
        r'import\s+\w+',            # Import injection
        r'__\w+__',                 # Python magic methods
    ]
    
    def __init__(self):
        self.logger = structlog.get_logger("SecureValidator")
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS]
    
    def validate_input(self, value: Any, field_name: str, rules: Dict[str, Any]) -> Any:
        """Comprehensive input validation"""
        
        if value is None and rules.get('required', False):
            raise ValidationError(f"{field_name} is required", field_name, "required")
        
        if value is None:
            return None
        
        # Convert to string for pattern checking
        str_value = str(value)
        
        # Check for dangerous patterns
        self._check_security_patterns(str_value, field_name)
        
        # Type validation
        if 'type' in rules:
            value = self._validate_type(value, rules['type'], field_name)
        
        # Length validation
        if 'min_length' in rules and len(str_value) < rules['min_length']:
            raise ValidationError(f"{field_name} must be at least {rules['min_length']} characters", 
                                field_name, "min_length")
        
        if 'max_length' in rules and len(str_value) > rules['max_length']:
            raise ValidationError(f"{field_name} must not exceed {rules['max_length']} characters", 
                                field_name, "max_length")
        
        # Pattern validation
        if 'pattern' in rules:
            if not re.match(rules['pattern'], str_value):
                raise ValidationError(f"{field_name} format is invalid", field_name, "pattern")
        
        # Custom validation
        if 'validator' in rules:
            if not rules['validator'](value):
                raise ValidationError(f"{field_name} failed custom validation", field_name, "custom")
        
        return value
    
    def _check_security_patterns(self, value: str, field_name: str):
        """Check for dangerous security patterns"""
        
        for pattern in self.compiled_patterns:
            if pattern.search(value):
                self.logger.warning("Dangerous pattern detected", 
                                  field=field_name, 
                                  pattern=pattern.pattern,
                                  value=value[:100])  # Log only first 100 chars
                raise SecurityViolation(f"Dangerous pattern detected in {field_name}", 
                                      severity="high", 
                                      action="input_rejected")
    
    def _validate_type(self, value: Any, expected_type: str, field_name: str) -> Any:
        """Validate and convert types safely"""
        
        try:
            if expected_type == 'int':
                return int(value)
            elif expected_type == 'float':
                return float(value)
            elif expected_type == 'bool':
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            elif expected_type == 'string':
                return str(value)
            elif expected_type == 'email':
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, str(value)):
                    raise ValidationError(f"Invalid email format", field_name, "email")
                return str(value)
            elif expected_type == 'url':
                url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
                if not re.match(url_pattern, str(value)):
                    raise ValidationError(f"Invalid URL format", field_name, "url")
                return str(value)
            else:
                return value
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid {expected_type} value for {field_name}", 
                                field_name, "type_conversion")


class RobustErrorHandler:
    """Comprehensive error handling and recovery"""
    
    def __init__(self):
        self.logger = structlog.get_logger("RobustErrorHandler")
        self.error_metrics = {
            'total_errors': 0,
            'error_types': {},
            'recovery_attempts': 0,
            'recovery_successes': 0
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle errors with appropriate recovery strategies"""
        
        self.error_metrics['total_errors'] += 1
        error_type = type(error).__name__
        self.error_metrics['error_types'][error_type] = self.error_metrics['error_types'].get(error_type, 0) + 1
        
        # Log error with context
        self.logger.error("Error occurred", 
                         error_type=error_type,
                         error_message=str(error),
                         context=context or {})
        
        # Determine error category and response
        if isinstance(error, ValidationError):
            return self._handle_validation_error(error)
        elif isinstance(error, SecurityViolation):
            return self._handle_security_violation(error)
        elif isinstance(error, ConnectionError):
            return self._handle_connection_error(error, context)
        elif isinstance(error, TimeoutError):
            return self._handle_timeout_error(error, context)
        elif isinstance(error, PermissionError):
            return self._handle_permission_error(error)
        else:
            return self._handle_generic_error(error)
    
    def _handle_validation_error(self, error: ValidationError) -> Dict[str, Any]:
        """Handle validation errors"""
        return {
            'error_type': 'validation',
            'message': str(error),
            'field': error.field,
            'code': error.code,
            'recoverable': True,
            'suggested_action': 'correct_input'
        }
    
    def _handle_security_violation(self, error: SecurityViolation) -> Dict[str, Any]:
        """Handle security violations"""
        # Log security incident
        self.logger.critical("Security violation detected",
                           severity=error.severity,
                           action=error.action,
                           message=str(error))
        
        return {
            'error_type': 'security_violation',
            'message': 'Security violation detected',
            'severity': error.severity,
            'recoverable': False,
            'suggested_action': 'review_security_logs'
        }
    
    def _handle_connection_error(self, error: ConnectionError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle connection errors with retry logic"""
        
        self.error_metrics['recovery_attempts'] += 1
        
        # Implement exponential backoff
        retry_count = context.get('retry_count', 0) if context else 0
        max_retries = 3
        
        if retry_count < max_retries:
            backoff_time = 2 ** retry_count
            return {
                'error_type': 'connection',
                'message': str(error),
                'recoverable': True,
                'suggested_action': 'retry_with_backoff',
                'retry_after': backoff_time,
                'retry_count': retry_count + 1
            }
        else:
            return {
                'error_type': 'connection',
                'message': 'Connection failed after maximum retries',
                'recoverable': False,
                'suggested_action': 'check_network_connectivity'
            }
    
    def _handle_timeout_error(self, error: TimeoutError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle timeout errors"""
        return {
            'error_type': 'timeout',
            'message': str(error),
            'recoverable': True,
            'suggested_action': 'increase_timeout_or_optimize'
        }
    
    def _handle_permission_error(self, error: PermissionError) -> Dict[str, Any]:
        """Handle permission errors"""
        return {
            'error_type': 'permission',
            'message': str(error),
            'recoverable': False,
            'suggested_action': 'check_permissions'
        }
    
    def _handle_generic_error(self, error: Exception) -> Dict[str, Any]:
        """Handle generic errors"""
        return {
            'error_type': 'generic',
            'message': str(error),
            'recoverable': True,
            'suggested_action': 'investigate_logs'
        }
    
    async def attempt_recovery(self, error_response: Dict[str, Any], recovery_function: Callable) -> Any:
        """Attempt to recover from an error"""
        
        if not error_response.get('recoverable', False):
            raise Exception(f"Error is not recoverable: {error_response.get('message')}")
        
        suggested_action = error_response.get('suggested_action')
        
        if suggested_action == 'retry_with_backoff':
            retry_after = error_response.get('retry_after', 1)
            await asyncio.sleep(retry_after)
            self.error_metrics['recovery_attempts'] += 1
            
            try:
                result = await recovery_function()
                self.error_metrics['recovery_successes'] += 1
                return result
            except Exception as e:
                self.logger.error("Recovery attempt failed", error=str(e))
                raise
        
        # Other recovery strategies can be implemented here
        raise Exception(f"Recovery strategy '{suggested_action}' not implemented")


class SecurityAuditLogger:
    """Security-focused audit logging"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = Path(log_file)
        self.logger = structlog.get_logger("SecurityAudit")
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str, 
                                 user_agent: str, reason: str = None):
        """Log authentication attempts"""
        
        event = {
            'event_type': 'authentication',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': user_id,
            'success': success,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'reason': reason
        }
        
        self._write_audit_log(event)
        
        if not success:
            self.logger.warning("Failed authentication attempt",
                              user_id=user_id,
                              ip_address=ip_address,
                              reason=reason)
    
    def log_authorization_check(self, user_id: str, resource: str, action: str, 
                              granted: bool, reason: str = None):
        """Log authorization checks"""
        
        event = {
            'event_type': 'authorization',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'granted': granted,
            'reason': reason
        }
        
        self._write_audit_log(event)
    
    def log_security_violation(self, violation: SecurityViolation, context: Dict[str, Any]):
        """Log security violations"""
        
        event = {
            'event_type': 'security_violation',
            'timestamp': violation.timestamp.isoformat(),
            'severity': violation.severity,
            'message': str(violation),
            'action': violation.action,
            'context': context
        }
        
        self._write_audit_log(event)
        
        self.logger.critical("Security violation logged",
                           severity=violation.severity,
                           message=str(violation))
    
    def log_data_access(self, user_id: str, resource: str, operation: str, 
                       classification: SecurityLevel, success: bool):
        """Log data access events"""
        
        event = {
            'event_type': 'data_access',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': user_id,
            'resource': resource,
            'operation': operation,
            'classification': classification.value,
            'success': success
        }
        
        self._write_audit_log(event)
    
    def _write_audit_log(self, event: Dict[str, Any]):
        """Write event to audit log file"""
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            self.logger.error("Failed to write audit log", error=str(e))


class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self):
        self.limits: Dict[str, RateLimitConfig] = {}
        self.usage: Dict[str, List[float]] = {}
        self.logger = structlog.get_logger("RateLimiter")
    
    def set_limit(self, key: str, config: RateLimitConfig):
        """Set rate limit for a specific key"""
        self.limits[key] = config
        if key not in self.usage:
            self.usage[key] = []
    
    async def check_rate_limit(self, key: str, weight: int = 1) -> bool:
        """Check if request is within rate limits"""
        
        if key not in self.limits:
            return True  # No limit set, allow request
        
        config = self.limits[key]
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - config.time_window
        self.usage[key] = [t for t in self.usage.setdefault(key, []) if t > cutoff_time]
        
        # Check current usage
        current_usage = len(self.usage[key])
        
        if current_usage + weight <= config.max_requests:
            # Add current request
            for _ in range(weight):
                self.usage[key].append(current_time)
            return True
        else:
            # Rate limit exceeded
            self.logger.warning("Rate limit exceeded",
                              key=key,
                              current_usage=current_usage,
                              limit=config.max_requests,
                              window=config.time_window)
            return False
    
    def get_rate_limit_status(self, key: str) -> Dict[str, Any]:
        """Get current rate limit status"""
        
        if key not in self.limits:
            return {'limited': False}
        
        config = self.limits[key]
        current_time = time.time()
        cutoff_time = current_time - config.time_window
        
        current_usage = len([t for t in self.usage.get(key, []) if t > cutoff_time])
        
        return {
            'limited': True,
            'current_usage': current_usage,
            'limit': config.max_requests,
            'window': config.time_window,
            'remaining': max(0, config.max_requests - current_usage),
            'reset_time': current_time + config.time_window
        }


class HealthMonitor:
    """Comprehensive health monitoring and alerting"""
    
    def __init__(self):
        self.logger = structlog.get_logger("HealthMonitor")
        self.metrics = {
            'system_health': 1.0,
            'last_check': datetime.now(timezone.utc),
            'checks_performed': 0,
            'checks_failed': 0,
            'alerts_sent': 0
        }
        self.health_checks: Dict[str, Callable] = {}
        self.alert_thresholds = {
            'error_rate': 0.1,      # 10% error rate
            'response_time': 5.0,    # 5 second response time
            'memory_usage': 0.85,    # 85% memory usage
            'cpu_usage': 0.90        # 90% CPU usage
        }
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_function
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        
        self.metrics['checks_performed'] += 1
        self.metrics['last_check'] = datetime.now(timezone.utc)
        
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_healthy': True,
            'checks': {}
        }
        
        for check_name, check_function in self.health_checks.items():
            try:
                check_result = await self._run_health_check(check_function)
                results['checks'][check_name] = check_result
                
                if not check_result.get('healthy', False):
                    results['overall_healthy'] = False
                    
            except Exception as e:
                self.metrics['checks_failed'] += 1
                self.logger.error("Health check failed", check=check_name, error=str(e))
                
                results['checks'][check_name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                results['overall_healthy'] = False
        
        # Update system health score
        healthy_checks = sum(1 for check in results['checks'].values() if check.get('healthy', False))
        total_checks = len(results['checks'])
        self.metrics['system_health'] = healthy_checks / max(total_checks, 1)
        
        # Send alerts if needed
        if not results['overall_healthy']:
            await self._send_health_alert(results)
        
        return results
    
    async def _run_health_check(self, check_function: Callable) -> Dict[str, Any]:
        """Run individual health check with timeout"""
        
        try:
            if asyncio.iscoroutinefunction(check_function):
                result = await asyncio.wait_for(check_function(), timeout=10.0)
            else:
                result = check_function()
            
            return {
                'healthy': True,
                'result': result,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                'healthy': False,
                'error': 'Health check timed out',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _send_health_alert(self, health_status: Dict[str, Any]):
        """Send health alert (placeholder for actual alerting system)"""
        
        self.metrics['alerts_sent'] += 1
        
        failed_checks = [name for name, result in health_status['checks'].items() 
                        if not result.get('healthy', False)]
        
        self.logger.critical("Health check alert",
                           failed_checks=failed_checks,
                           overall_healthy=health_status['overall_healthy'])
        
        # In a real implementation, this would send emails, Slack messages, etc.


def security_required(permissions: List[str] = None, security_level: SecurityLevel = SecurityLevel.INTERNAL):
    """Decorator for functions requiring security validation"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract security context from kwargs or context manager
            security_context = kwargs.pop('security_context', None)
            
            if not security_context or not isinstance(security_context, SecurityContext):
                raise SecurityViolation("No valid security context provided", severity="high")
            
            # Check if context is expired
            if security_context.is_expired():
                raise SecurityViolation("Security context expired", severity="medium")
            
            # Check permissions
            if permissions:
                for permission in permissions:
                    if not security_context.has_permission(permission):
                        raise SecurityViolation(f"Missing required permission: {permission}", severity="high")
            
            # Check security level
            context_level_priority = {
                SecurityLevel.PUBLIC: 0,
                SecurityLevel.INTERNAL: 1,
                SecurityLevel.CONFIDENTIAL: 2,
                SecurityLevel.RESTRICTED: 3
            }
            
            if context_level_priority[security_context.security_level] < context_level_priority[security_level]:
                raise SecurityViolation(f"Insufficient security clearance. Required: {security_level.value}, "
                                      f"Have: {security_context.security_level.value}", severity="high")
            
            # Call the original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limited(key_generator: Callable = None, max_requests: int = 100, time_window: int = 3600):
    """Decorator for rate limiting functions"""
    
    rate_limiter = RateLimiter()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate rate limit key
            if key_generator:
                key = key_generator(*args, **kwargs)
            else:
                key = f"{func.__name__}_default"
            
            # Set rate limit if not already set
            if key not in rate_limiter.limits:
                rate_limiter.set_limit(key, RateLimitConfig(max_requests, time_window))
            
            # Check rate limit
            if not await rate_limiter.check_rate_limit(key):
                status = rate_limiter.get_rate_limit_status(key)
                raise SecurityViolation(f"Rate limit exceeded. Try again after {status['reset_time']}", 
                                      severity="medium", action="rate_limited")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Example usage and testing
async def main():
    """Example usage of the robust security framework"""
    
    # Initialize components
    validator = SecureValidator()
    error_handler = RobustErrorHandler()
    audit_logger = SecurityAuditLogger()
    rate_limiter = RateLimiter()
    health_monitor = HealthMonitor()
    
    print("🛡️ Robust Security Framework Initialized")
    
    # Test validation
    try:
        valid_email = validator.validate_input("user@example.com", "email", {"type": "email", "required": True})
        print(f"✅ Valid email: {valid_email}")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    # Test security violation detection
    try:
        validator.validate_input("<script>alert('xss')</script>", "comment", {"type": "string"})
    except SecurityViolation as e:
        print(f"🚨 Security violation detected: {e}")
    
    # Test rate limiting
    rate_limiter.set_limit("test_endpoint", RateLimitConfig(max_requests=5, time_window=60))
    
    for i in range(7):
        allowed = await rate_limiter.check_rate_limit("test_endpoint")
        print(f"Request {i+1}: {'✅ Allowed' if allowed else '❌ Rate limited'}")
    
    # Test health monitoring
    async def sample_health_check():
        return {"status": "healthy", "uptime": 3600}
    
    health_monitor.register_health_check("sample", sample_health_check)
    health_status = await health_monitor.perform_health_check()
    print(f"🏥 Health status: {'✅ Healthy' if health_status['overall_healthy'] else '❌ Unhealthy'}")


if __name__ == "__main__":
    asyncio.run(main())