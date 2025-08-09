"""
Generation 2: MAKE IT ROBUST - Enhanced System Robustness Implementation

This module implements comprehensive robustness features for the Claude Manager Service:

1. Enhanced Error Handling & Recovery
2. Input Validation & Sanitization  
3. Comprehensive Logging & Monitoring
4. Security Hardening
5. Configuration Validation

This is the evolution of the system to enterprise-grade reliability and security.
"""

import asyncio
import time
import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import logging
import functools

from .error_handler import (
    EnhancedError, FileOperationError, NetworkError, RateLimitError,
    AuthenticationError, get_rate_limiter, get_circuit_breaker, 
    get_error_tracker, with_enhanced_error_handling,
    github_api_operation, network_operation, file_operation
)
from .validation import (
    ValidationError, ConfigurationValidationError, ParameterValidationError,
    validate_config_schema, validate_api_parameters, get_validator
)
from .logger import get_logger
from .services.configuration_service import ConfigurationService


@dataclass
class SystemHealth:
    """System health status representation"""
    overall_status: str
    components: Dict[str, Dict[str, Any]]
    timestamp: datetime
    uptime_seconds: float
    error_rate: float
    performance_metrics: Dict[str, float]


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: List[str] = None


class InputSanitizer:
    """Enhanced input sanitization and validation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.validator = get_validator()
    
    def sanitize_github_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize GitHub API inputs"""
        sanitized = {}
        
        # Repository name sanitization
        if 'repo_name' in input_data:
            repo_name = str(input_data['repo_name']).strip()
            # Remove potentially malicious characters
            repo_name = ''.join(c for c in repo_name if c.isalnum() or c in '._-/')
            if len(repo_name.split('/')) != 2:
                raise ParameterValidationError("Invalid repository name format")
            sanitized['repo_name'] = repo_name
        
        # Issue title sanitization
        if 'title' in input_data:
            title = str(input_data['title']).strip()
            # Remove HTML/script tags and limit length
            title = self._remove_html_tags(title)
            title = title[:256]  # GitHub limit
            sanitized['title'] = title
        
        # Issue body sanitization
        if 'body' in input_data:
            body = str(input_data['body']).strip()
            # Basic HTML sanitization but allow markdown
            body = self._sanitize_markdown(body)
            body = body[:65536]  # GitHub limit
            sanitized['body'] = body
        
        # Label sanitization
        if 'labels' in input_data:
            labels = input_data['labels']
            if isinstance(labels, list):
                sanitized_labels = []
                for label in labels[:100]:  # GitHub limit
                    label_str = str(label).strip()
                    label_str = ''.join(c for c in label_str if c.isalnum() or c in '._- ')
                    if label_str and len(label_str) <= 50:
                        sanitized_labels.append(label_str)
                sanitized['labels'] = sanitized_labels
        
        # Issue number sanitization
        if 'issue_number' in input_data:
            try:
                issue_num = int(input_data['issue_number'])
                if 1 <= issue_num <= 999999999:  # Reasonable range
                    sanitized['issue_number'] = issue_num
                else:
                    raise ParameterValidationError("Issue number out of range")
            except (ValueError, TypeError):
                raise ParameterValidationError("Invalid issue number")
        
        return sanitized
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        import re
        # Remove script tags and their content
        text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', text, flags=re.IGNORECASE)
        # Remove other HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def _sanitize_markdown(self, text: str) -> str:
        """Basic markdown sanitization"""
        # Remove script tags but allow other markdown
        import re
        text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'data:', '', text, flags=re.IGNORECASE)
        return text
    
    def validate_and_sanitize(self, data: Dict[str, Any], operation: str) -> Dict[str, Any]:
        """Combined validation and sanitization"""
        # First validate the structure
        validate_api_parameters(data, operation)
        
        # Then sanitize the content
        if operation in ['create_issue', 'add_comment', 'get_repository']:
            return self.sanitize_github_input(data)
        
        return data


class SecurityManager:
    """Comprehensive security management"""
    
    def __init__(self, config_service: ConfigurationService):
        self.config_service = config_service
        self.logger = get_logger(__name__)
        self.rate_limiter = get_rate_limiter()
        self.failed_attempts: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
    
    async def validate_request(self, context: SecurityContext, operation: str) -> bool:
        """Validate security context for operation"""
        try:
            # Check IP blocking
            if context.source_ip and self._is_ip_blocked(context.source_ip):
                raise AuthenticationError("IP address blocked", operation)
            
            # Check rate limiting
            rate_key = f"{context.source_ip or 'unknown'}:{operation}"
            if not self.rate_limiter.can_proceed(rate_key):
                self._record_failed_attempt(context.source_ip or "unknown")
                raise RateLimitError("Rate limit exceeded", operation)
            
            # Validate user agent
            if context.user_agent:
                if self._is_suspicious_user_agent(context.user_agent):
                    self.logger.warning(f"Suspicious user agent: {context.user_agent}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            raise
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked"""
        if ip in self.blocked_ips:
            block_time = self.blocked_ips[ip]
            if datetime.now() - block_time > timedelta(hours=1):
                del self.blocked_ips[ip]
                return False
            return True
        return False
    
    def _record_failed_attempt(self, identifier: str):
        """Record failed attempt and potentially block"""
        current_time = time.time()
        
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        attempts = self.failed_attempts[identifier]
        # Clean old attempts
        attempts[:] = [t for t in attempts if current_time - t < 3600]  # Last hour
        
        attempts.append(current_time)
        
        # Block after 10 failed attempts in an hour
        if len(attempts) >= 10:
            self.blocked_ips[identifier] = datetime.now()
            self.logger.warning(f"Blocked identifier due to repeated failures: {identifier}")
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check for suspicious user agent patterns"""
        suspicious_patterns = [
            'bot', 'crawler', 'spider', 'scraper', 'python-requests',
            'curl', 'wget', 'nikto', 'sqlmap', 'havij'
        ]
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)
    
    async def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for responses"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        }


class PerformanceMonitor:
    """Enhanced performance monitoring and metrics"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.start_time = time.time()
    
    @asynccontextmanager
    async def monitor_operation(self, operation: str):
        """Monitor operation performance"""
        start_time = time.time()
        
        try:
            yield
            # Record success
            duration = time.time() - start_time
            self._record_metric(f"{operation}_duration", duration)
            self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
            
        except Exception as e:
            # Record error
            duration = time.time() - start_time
            self._record_metric(f"{operation}_duration", duration)
            self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
            raise
    
    def _record_metric(self, metric_name: str, value: float):
        """Record performance metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        metrics_list = self.metrics[metric_name]
        metrics_list.append(value)
        
        # Keep only last 1000 measurements
        if len(metrics_list) > 1000:
            metrics_list.pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'uptime_seconds': time.time() - self.start_time,
            'total_operations': sum(self.operation_counts.values()),
            'total_errors': sum(self.error_counts.values()),
            'error_rate': 0.0,
            'operations': {},
            'metrics': {}
        }
        
        total_ops = summary['total_operations']
        if total_ops > 0:
            summary['error_rate'] = summary['total_errors'] / total_ops
        
        # Operation summaries
        for operation, count in self.operation_counts.items():
            errors = self.error_counts.get(operation, 0)
            summary['operations'][operation] = {
                'count': count,
                'errors': errors,
                'error_rate': errors / count if count > 0 else 0.0
            }
        
        # Metric summaries
        for metric_name, values in self.metrics.items():
            if values:
                summary['metrics'][metric_name] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return summary


class RobustSystem:
    """
    Generation 2 Robust System Implementation
    
    This class provides enterprise-grade robustness features:
    - Enhanced error handling and recovery
    - Comprehensive input validation and sanitization
    - Security hardening
    - Performance monitoring
    - Configuration validation
    """
    
    def __init__(self, config_service: ConfigurationService):
        self.config_service = config_service
        self.logger = get_logger(__name__)
        
        # Initialize robust components
        self.input_sanitizer = InputSanitizer()
        self.security_manager = SecurityManager(config_service)
        self.performance_monitor = PerformanceMonitor()
        self.error_tracker = get_error_tracker()
        self.circuit_breaker = get_circuit_breaker()
        
        # System state
        self.system_start_time = datetime.now()
        self.health_checks_enabled = True
        
        self.logger.info("RobustSystem initialized with Generation 2 features")
    
    async def initialize(self) -> None:
        """Initialize the robust system"""
        try:
            # Validate configuration
            config = await self.config_service.get_config()
            validate_config_schema(config)
            
            # Initialize security components
            await self._initialize_security()
            
            # Setup monitoring
            await self._initialize_monitoring()
            
            self.logger.info("RobustSystem initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"RobustSystem initialization failed: {e}")
            raise
    
    async def _initialize_security(self):
        """Initialize security components"""
        try:
            # Check for required environment variables
            required_env = ['GITHUB_TOKEN']
            missing_env = [var for var in required_env if not os.getenv(var)]
            if missing_env:
                self.logger.warning(f"Missing environment variables: {missing_env}")
            
            # Initialize credential validation
            await self._validate_credentials()
            
        except Exception as e:
            self.logger.error(f"Security initialization failed: {e}")
            raise
    
    async def _initialize_monitoring(self):
        """Initialize monitoring components"""
        try:
            # Setup performance monitoring
            self.logger.info("Performance monitoring initialized")
            
            # Setup health checks
            await self._setup_health_checks()
            
        except Exception as e:
            self.logger.error(f"Monitoring initialization failed: {e}")
            raise
    
    async def _validate_credentials(self):
        """Validate system credentials"""
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            raise AuthenticationError("GitHub token not configured", "credential_validation")
        
        if len(github_token) < 20:  # Basic sanity check
            raise AuthenticationError("Invalid GitHub token format", "credential_validation")
    
    async def _setup_health_checks(self):
        """Setup periodic health checks"""
        # This would be extended with actual health check implementation
        self.logger.info("Health checks configured")
    
    @with_enhanced_error_handling(
        operation="secure_api_call",
        use_rate_limiter=True,
        use_circuit_breaker=True
    )
    async def execute_secure_operation(
        self,
        operation: str,
        parameters: Dict[str, Any],
        context: SecurityContext = None
    ) -> Any:
        """
        Execute operation with full security and robustness features
        
        Args:
            operation: Operation name
            parameters: Operation parameters  
            context: Security context
            
        Returns:
            Operation result
        """
        context = context or SecurityContext()
        
        async with self.performance_monitor.monitor_operation(operation):
            try:
                # Security validation
                await self.security_manager.validate_request(context, operation)
                
                # Input validation and sanitization
                safe_parameters = self.input_sanitizer.validate_and_sanitize(parameters, operation)
                
                # Execute operation (would be implemented by specific services)
                result = await self._execute_operation(operation, safe_parameters)
                
                self.logger.info(f"Secure operation completed: {operation}")
                return result
                
            except Exception as e:
                # Record error for monitoring
                self.error_tracker.record_error(
                    module="robust_system",
                    function="execute_secure_operation",
                    error_type=e.__class__.__name__,
                    message=str(e)
                )
                raise
    
    async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute the actual operation (to be implemented by specific services)"""
        # This is a placeholder - actual implementation would delegate to appropriate services
        self.logger.info(f"Executing operation: {operation} with parameters: {list(parameters.keys())}")
        return {"status": "completed", "operation": operation}
    
    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        try:
            # Get performance metrics
            perf_summary = self.performance_monitor.get_performance_summary()
            
            # Get error statistics
            error_stats = self.error_tracker.get_error_statistics()
            
            # Component health checks
            components = {
                'configuration': await self._check_configuration_health(),
                'security': await self._check_security_health(),
                'performance': await self._check_performance_health(),
                'error_handling': await self._check_error_handling_health()
            }
            
            # Calculate overall status
            overall_status = "healthy"
            if any(comp.get("status") == "unhealthy" for comp in components.values()):
                overall_status = "unhealthy"
            elif any(comp.get("status") == "degraded" for comp in components.values()):
                overall_status = "degraded"
            
            return SystemHealth(
                overall_status=overall_status,
                components=components,
                timestamp=datetime.now(),
                uptime_seconds=perf_summary['uptime_seconds'],
                error_rate=perf_summary['error_rate'],
                performance_metrics=perf_summary.get('metrics', {})
            )
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return SystemHealth(
                overall_status="unhealthy",
                components={"error": {"status": "unhealthy", "message": str(e)}},
                timestamp=datetime.now(),
                uptime_seconds=0.0,
                error_rate=1.0,
                performance_metrics={}
            )
    
    async def _check_configuration_health(self) -> Dict[str, Any]:
        """Check configuration system health"""
        try:
            # Test configuration access
            config = await self.config_service.get_config('github')
            if not config:
                return {"status": "unhealthy", "message": "Configuration not accessible"}
            
            return {"status": "healthy", "message": "Configuration system operational"}
            
        except Exception as e:
            return {"status": "unhealthy", "message": f"Configuration error: {str(e)}"}
    
    async def _check_security_health(self) -> Dict[str, Any]:
        """Check security system health"""
        try:
            # Check credential availability
            if not os.getenv('GITHUB_TOKEN'):
                return {"status": "degraded", "message": "GitHub credentials not configured"}
            
            return {"status": "healthy", "message": "Security system operational"}
            
        except Exception as e:
            return {"status": "unhealthy", "message": f"Security error: {str(e)}"}
    
    async def _check_performance_health(self) -> Dict[str, Any]:
        """Check performance monitoring health"""
        try:
            summary = self.performance_monitor.get_performance_summary()
            
            # Check error rate
            if summary['error_rate'] > 0.1:  # 10% error rate threshold
                return {
                    "status": "degraded", 
                    "message": f"High error rate: {summary['error_rate']:.2%}"
                }
            
            return {"status": "healthy", "message": "Performance monitoring operational"}
            
        except Exception as e:
            return {"status": "unhealthy", "message": f"Performance monitoring error: {str(e)}"}
    
    async def _check_error_handling_health(self) -> Dict[str, Any]:
        """Check error handling system health"""
        try:
            recent_errors = self.error_tracker.get_recent_errors(10)
            
            if len(recent_errors) > 5:  # More than 5 recent errors
                return {
                    "status": "degraded", 
                    "message": f"High error count: {len(recent_errors)} recent errors"
                }
            
            return {"status": "healthy", "message": "Error handling system operational"}
            
        except Exception as e:
            return {"status": "unhealthy", "message": f"Error handling system error: {str(e)}"}
    
    async def export_metrics(self, export_path: str) -> None:
        """Export system metrics and health data"""
        try:
            health = await self.get_system_health()
            error_stats = self.error_tracker.get_error_statistics()
            recent_errors = self.error_tracker.get_recent_errors(100)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'system_health': asdict(health),
                'error_statistics': error_stats,
                'recent_errors': recent_errors,
                'configuration_summary': await self._get_config_summary()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to: {export_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            raise
    
    async def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for export"""
        try:
            config = await self.config_service.get_config()
            
            # Remove sensitive information
            summary = {}
            for key, value in config.items():
                if isinstance(value, dict):
                    summary[key] = {k: v for k, v in value.items() if 'token' not in k.lower() and 'password' not in k.lower()}
                else:
                    if 'token' not in key.lower() and 'password' not in key.lower():
                        summary[key] = value
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to get config summary: {str(e)}"}


# Factory function for easy initialization
async def create_robust_system(config_service: ConfigurationService) -> RobustSystem:
    """
    Create and initialize a RobustSystem instance
    
    Args:
        config_service: Initialized configuration service
        
    Returns:
        Initialized RobustSystem instance
    """
    robust_system = RobustSystem(config_service)
    await robust_system.initialize()
    return robust_system


# Decorator for making any function robust
def make_robust(operation_name: str):
    """Decorator to add robustness features to any function"""
    return with_enhanced_error_handling(
        operation=operation_name,
        use_rate_limiter=True,
        use_circuit_breaker=True
    )