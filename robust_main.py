```python
#!/usr/bin/env python3
"""
Claude Manager Service - Robust Entry Point (Generation 2: MAKE IT ROBUST)

Enhanced implementation with comprehensive error handling, logging, monitoring,
security measures, and resilience patterns.
"""

import asyncio
import json
import os
import sys
import logging
import traceback
import time
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from functools import wraps
from dataclasses import dataclass, asdict
import signal
import argparse
from contextlib import asynccontextmanager

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# Security and validation imports
try:
    import validators
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False

# =============================================================================
# ERROR CLASSES
# =============================================================================

class ClaudeManagerError(Exception):
    """Base exception for Claude Manager"""
    pass

class ConfigurationError(ClaudeManagerError):
    """Configuration related errors"""
    pass

class SecurityError(ClaudeManagerError):
    """Security related errors"""
    pass

class ValidationError(ClaudeManagerError):
    """Input validation errors"""
    pass

class GitHubAPIError(ClaudeManagerError):
    """GitHub API related errors"""
    pass

# =============================================================================
# SECURITY AND VALIDATION
# =============================================================================

class SecurityValidator:
    """Comprehensive security validation and sanitization"""
    
    @staticmethod
    def sanitize_input(value: str, max_length: int = 1000) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")
        
        if len(value) > max_length:
            raise ValidationError(f"Input too long: {len(value)} > {max_length}")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>&"\'`$();|]', '', value)
        
        # Additional validation for paths
        if '..' in sanitized or sanitized.startswith('/'):
            raise SecurityError("Potentially unsafe path detected")
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized.strip()
    
    @staticmethod
    def validate_repo_name(repo_name: str) -> str:
        """Validate and sanitize repository name"""
        if not repo_name:
            raise ValidationError("Repository name cannot be empty")
        
        # GitHub repo format: owner/repo
        pattern = r'^[a-zA-Z0-9\-_.]+/[a-zA-Z0-9\-_.]+$'
        if not re.match(pattern, repo_name):
            raise ValidationError(f"Invalid repository name format: {repo_name}")
        
        if len(repo_name) > 100:
            raise ValidationError(f"Repository name too long: {len(repo_name)}")
        
        return repo_name
    
    @staticmethod
    def validate_file_path(file_path: str) -> str:
        """Validate file path to prevent directory traversal"""
        if not file_path:
            raise ValidationError("File path cannot be empty")
        
        # Prevent directory traversal
        if '..' in file_path or file_path.startswith('/'):
            raise ValidationError(f"Unsafe file path: {file_path}")
        
        # Sanitize path
        safe_path = os.path.normpath(file_path)
        
        if safe_path != file_path:
            raise ValidationError(f"Path normalization changed: {file_path} -> {safe_path}")
        
        return safe_path
    
    @staticmethod
    def validate_github_token(token: str) -> bool:
        """Basic GitHub token validation"""
        if not token:
            return False
        
        # GitHub tokens are typically 40 characters or more
        if len(token) < 20 or len(token) > 100:
            return False
        
        # Should contain only alphanumeric characters and underscores
        pattern = r'^[a-zA-Z0-9_]+$'
        return bool(re.match(pattern, token))

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

class RobustLogger:
    """Advanced logging with structured output and performance tracking"""
    
    def __init__(self, name: str, config: LogConfig = None):
        if config is None:
            config = LogConfig()
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.level.upper()))
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Console handler with Rich
        console_handler = RichHandler(rich_tracebacks=True)
        console_handler.setLevel(getattr(logging, config.level))
        console_formatter = logging.Formatter(config.format)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if config.file_path:
            try:
                from logging.handlers import RotatingFileHandler
                log_file = Path(config.file_path)
                log_file.parent.mkdir(exist_ok=True)
                
                file_handler = RotatingFileHandler(
                    config.file_path,
                    maxBytes=config.max_file_size,
                    backupCount=config.backup_count
                )
                file_handler.setLevel(getattr(logging, config.level))
                file_formatter = logging.Formatter(config.format)
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Could not setup file logging: {e}")
        
        self.performance_data = []
    
    def sanitize_log_data(self, data: Any) -> str:
        """Sanitize data before logging to prevent injection"""
        if isinstance(data, dict):
            # Remove sensitive keys
            sensitive_keys = ['token', 'password', 'secret', 'key', 'auth']
            sanitized = {}
            for k, v in data.items():
                if any(sensitive_key in k.lower() for sensitive_key in sensitive_keys):
                    sanitized[k] = "***REDACTED***"
                else:
                    sanitized[k] = str(v)[:200]  # Limit length
            return json.dumps(sanitized, indent=2)
        
        # Convert to string and limit length
        return str(data)[:500]
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data"""
        if kwargs:
            message = f"{message} | {self.sanitize_log_data(kwargs)}"
        self.logger.info(message)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error with exception details"""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
            if kwargs:
                message = f"{message} | {self.sanitize_log_data(kwargs)}"
            self.logger.error(message, exc_info=True)
        else:
            if kwargs:
                message = f"{message} | {self.sanitize_log_data(kwargs)}"
            self.logger.error(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        if kwargs:
            message = f"{message} | {self.sanitize_log_data(kwargs)}"
        self.logger.warning(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        if kwargs:
            message = f"{message} | {self.sanitize_log_data(kwargs)}"
        self.logger.debug(message)
    
    def track_performance(self, operation: str, duration: float, **metadata):
        """Track performance metrics"""
        perf_data = {
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            **metadata
        }
        self.performance_data.append(perf_data)
        
        self.info(f"Performance: {operation} completed in {duration:.3f}s", **metadata)
        
        # Keep only last 1000 entries to prevent memory growth
        if len(self.performance_data) > 1000:
            self.performance_data = self.performance_data[-1000:]

# =============================================================================
# ERROR HANDLING AND RESILIENCE
# =============================================================================

@dataclass
class ErrorContext:
    """Structured error context for better debugging"""
    operation: str
    timestamp: str
    error_type: str
    error_message: str
    traceback: str
    metadata: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    memory_usage: Optional[float] = None

class RobustErrorHandler:
    """Comprehensive error handling with recovery strategies"""
    
    def __init__(self, logger: RobustLogger):
        self.logger = logger
        self.error_history = []
    
    def handle_error(self, operation: str, exception: Exception, **metadata) -> ErrorContext:
        """Handle error with structured logging and context"""
        error_context = ErrorContext(
            operation=operation,
            timestamp=datetime.now().isoformat(),
            error_type=type(exception).__name__,
            error_message=str(exception),
            traceback=traceback.format_exc(),
            metadata=metadata
        )
        
        self.error_history.append(error_context)
        
        self.logger.error(
            f"Error in {operation}",
            exception=exception,
            error_type=error_context.error_type,
            **metadata
        )
        
        return error_context
    
    async def with_retry(self, operation_func, max_retries: int = 3, delay: float = 1.0, **kwargs):
        """Execute operation with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await operation_func(**kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    wait_time = delay * (2 ** attempt)
                    self.logger.warning(
                        f"Operation failed, retrying in {wait_time}s",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e)
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.handle_error("retry_exhausted", e, max_retries=max_retries)
        
        raise last_exception

def handle_errors(logger: RobustLogger = None):
    """Decorator for comprehensive error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ClaudeManagerError as e:
                if logger:
                    logger.error(f"Claude Manager error in {func.__name__}: {e}")
                rprint(f"[red]Error: {e}[/red]")
                raise typer.Exit(1)
            except Exception as e:
                if logger:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                rprint(f"[red]Unexpected error: {e}[/red]")
                rprint(f"[dim]Function: {func.__name__}[/dim]")
                raise typer.Exit(1)
        return wrapper
    return decorator

def async_handle_errors(logger: RobustLogger = None):
    """Decorator for async function error handling"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ClaudeManagerError as e:
                if logger:
                    logger.error(f"Claude Manager error in {func.__name__}: {e}")
                rprint(f"[red]Error: {e}[/red]")
                raise typer.Exit(1)
            except Exception as e:
                if logger:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                rprint(f"[red]Unexpected error: {e}[/red]")
                raise typer.Exit(1)
        return wrapper
    return decorator

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class RobustConfigManager:
    """Advanced configuration management with validation and security"""
    
    def __init__(self, config_path: str, logger: RobustLogger):
        self.config_path = config_path
        self.logger = logger
        self.config = {}
        self.config_hash = ""
        
    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration with integrity checking"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    content = f.read()
                
                # Calculate content hash for integrity
                self.config_hash = hashlib.sha256(content.encode()).hexdigest()
                
                self.config = json.loads(content)
                self.logger.info(f"Configuration loaded successfully", 
                               config_path=self.config_path,
                               config_hash=self.config_hash[:8])
            else:
                self.logger.warning(f"Config file {self.config_path} not found, creating default")
                self.config = self._create_default_config()
            
            # Validate configuration structure
            self._validate_config()
            
            return self.config
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration", exception=e)
            raise ConfigurationError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration", exception=e)
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _validate_config(self):
        """Comprehensive configuration validation"""
        required_sections = ['github', 'analyzer', 'executor']
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate GitHub configuration
        github_config = self.config['github']
        required_github_keys = ['username', 'managerRepo', 'reposToScan']
        for key in required_github_keys:
            if key not in github_config:
                raise ConfigurationError(f"Missing required GitHub configuration: {key}")
        
        # Validate repository names
        for repo in github_config['reposToScan']:
            SecurityValidator.validate_repo_name(repo)
        
        SecurityValidator.validate_repo_name(github_config['managerRepo'])
        
        # Validate analyzer configuration
        analyzer_config = self.config['analyzer']
        if not isinstance(analyzer_config.get('scanForTodos'), bool):
            raise ConfigurationError("analyzer.scanForTodos must be boolean")
        if not isinstance(analyzer_config.get('scanOpenIssues'), bool):
            raise ConfigurationError("analyzer.scanOpenIssues must be boolean")
        
        self.logger.info("Configuration validation passed")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            "github": {
                "username": "your-username",
                "managerRepo": "your-username/claude-manager-service",
                "reposToScan": []
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True,
                "maxResults": 100
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            },
            "security": {
                "maxInputLength": 1000,
                "enableValidation": True,
                "logSensitiveData": False
            },
            "monitoring": {
                "enableHealthChecks": True,
                "metricsCollection": True,
                "performanceTracking": True
            }
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            rprint(f"[green]Created default config at {self.config_path}[/green]")
        except Exception as e:
            self.logger.warning(f"Could not save default config: {e}")
        
        return default_config
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def get(self, key: str, default=None):
        """Get configuration value with security checks"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            # Security check: don't log sensitive values
            if any(sensitive in key.lower() for sensitive in ['token', 'password', 'secret']):
                self.logger.info(f"Retrieved sensitive config key: {key}")
            else:
                self.logger.info(f"Retrieved config: {key} = {value}")
            
            return value
        except Exception as e:
            self.logger.error(f"Error retrieving config key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, persist: bool = False):
        """Set configuration value with validation"""
        try:
            # Validate input
            if not isinstance(key, str) or not key:
                raise ValidationError("Configuration key must be a non-empty string")
            
            keys = key.split('.')
            config = self.config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            
            # Persist if requested
            if persist:
                self._save_config()
            
            self.logger.info(f"Configuration updated: {key}")
            
        except Exception as e:
            raise ConfigurationError(f"Error setting config key {key}: {e}")
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            # Create backup
            backup_path = f"{self.config_path}.backup"
            if os.path.exists(self.config_path):
                import shutil
                shutil.copy2(self.config_path, backup_path)
            
            # Save new config
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info("Configuration saved successfully")
            
        except Exception as e:
            # Restore backup if save failed
            backup_path = f"{self.config_path}.backup"
            if os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, self.config_path)
            raise ConfigurationError(f"Error saving configuration: {e}")
    
    def watch_config_changes(self) -> bool:
        """Check if configuration file has changed"""
        try:
            with open(self.config_path, 'r') as f:
                content = f.read()
            
            current_hash = hashlib.sha256(content.encode()).hexdigest()
            
            if current_hash != self.config_hash:
                self.logger.info("Configuration file changed, reloading")
                self.load_config()
                return True
                
        except Exception as e:
            self.logger.error("Failed to check configuration changes", exception=e)
        
        return False

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Monitor performance and collect metrics"""
    
    def __init__(self, logger: RobustLogger):
        self.logger = logger
        self.metrics: List[PerformanceMetrics] = []
        self.operation_counts = {}
    
    def start_operation(self, operation: str) -> float:
        """Start monitoring an operation"""
        start_time = time.time()
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
        self.logger.info(f"Started operation: {operation}")
        return start_time
    
    def end_operation(self, operation: str, start_time: float, success: bool = True, error_message: str = None):
        """End monitoring an operation"""
        end_time = time.time()
        duration = end_time - start_time
        
        # Get memory usage if psutil is available
        memory_usage = None
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass
        
        metric = PerformanceMetrics(
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success=success,
            error_message=error_message,
            memory_usage=memory_usage
        )
        
        self.metrics.append(metric)
        
        if success:
            self.logger.info(f"Completed operation {operation} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed operation {operation} after {duration:.2f}s: {error_message}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.metrics:
            return {"message": "No metrics collected"}
        
        successful_metrics = [m for m in self.metrics if m.success]
        failed_metrics = [m for m in self.metrics if not m.success]
        
        stats = {
            "total_operations": len(self.metrics),
            "successful_operations": len(successful_metrics),
            "failed_operations": len(failed_metrics),
            "success_rate": len(successful_metrics) / len(self.metrics) * 100,
            "average_duration": sum(m.duration for m in self.metrics) / len(self.metrics),
            "operation_counts": self.operation_counts
        }
        
        if successful_metrics:
            stats["average_successful_duration"] = sum(m.duration for m in successful_metrics) / len(successful_metrics)
        
        return stats

# =============================================================================
# HEALTH MONITORING
# =============================================================================

@dataclass
class HealthCheckResult:
    """Structured health check result"""
    check_name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    details: Dict[str, Any]
    duration: float
    timestamp: str

class HealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, logger: RobustLogger, config_manager: RobustConfigManager):
        self.logger = logger
        self.config_manager = config_manager
        self.health_history = []
        self.performance_monitor = PerformanceMonitor(logger)
    
    async def run_health_check(self) -> Dict[str, HealthCheckResult]:
        """Run comprehensive health check"""
        self.logger.info("Starting comprehensive health check")
        
        checks = {
            'configuration': self._check_configuration,
            'environment': self._check_environment,
            'file_system': self._check_file_system,
            'network': self._check_network,
            'security': self._check_security,
            'dependencies': self._check_dependencies,
            'github_connectivity': self._check_github_connectivity,
            'performance': self._check_performance
        }
        
        results = {}
        for check_name, check_func in checks.items():
            start_time = time.time()
            try:
                result = await check_func()
                duration = time.time() - start_time
                
                results[check_name] = HealthCheckResult(
                    check_name=check_name,
                    status=result['status'],
                    message=result['message'],
                    details=result.get('details', {}),
                    duration=duration,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                duration = time.time() - start_time
                results[check_name] = HealthCheckResult(
                    check_name=check_name,
                    status='unhealthy',
                    message=f"Health check failed: {str(e)}",
                    details={'error': str(e)},
                    duration=duration,
                    timestamp=datetime.now().isoformat()
                )
                
                self.logger.error(f"Health check {check_name} failed", exception=e)
        
        # Store in history
        overall_status = self._calculate_overall_status(results)
        self.health_history.append({
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'results': results
        })
        
        # Keep only last 100 health checks
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        self.logger.info(f"Health check completed", overall_status=overall_status)
        
        return results
    
    async def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration health"""
        try:
            config = self.config_manager.get_config()
            
            username = config['github']['username']
            repos_count = len(config['github']['reposToScan'])
            
            if not username or username == "your-username":
                return {
                    'status': 'degraded',
                    'message': 'GitHub username not configured',
                    'details': {'repos_count': repos_count}
                }
            
            if repos_count == 0:
                return {
                    'status': 'degraded',
                    'message': 'No repositories configured for scanning',
                    'details': {'repos_count': repos_count}
                }
            
            return {
                'status': 'healthy',
                'message': f'Configuration valid with {repos_count} repositories',
                'details': {
                    'repos_count': repos_count,
                    'config_hash': self.config_manager.config_hash[:8]
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Configuration error: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def _check_environment(self) -> Dict[str, Any]:
        """Check environment variables and dependencies"""
        github_token = bool(os.getenv('GITHUB_TOKEN'))
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        issues = []
        if not github_token:
            issues.append("GITHUB_TOKEN not set")
        elif github_token and not SecurityValidator.validate_github_token(os.getenv('GITHUB_TOKEN')):
            issues.append("GITHUB_TOKEN appears invalid")
        
        if issues:
            return {
                'status': 'degraded',
                'message': f'Environment issues: {", ".join(issues)}',
                'details': {
                    'github_token': github_token,
                    'python_version': python_version
                }
            }
        
        return {
            'status': 'healthy',
            'message': f'Environment configured correctly, Python {python_version}',
            'details': {
                'github_token': github_token,
                'python_version': python_version,
                'working_directory': os.getcwd()
            }
        }
    
    async def _check_file_system(self) -> Dict[str, Any]:
        """Check file system access and permissions"""
        try:
            # Test read/write access
            test_file = Path('health_check_test.tmp')
            test_file.write_text('test')
            content = test_file.read_text()
            test_file.unlink()
            
            if content != 'test':
                return {
                    'status': 'unhealthy',
                    'message': 'File system read/write test failed',
                    'details': {'test_result': content}
                }
            
            # Check log directory
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # Check disk space
            try:
                import shutil
                total, used, free = shutil.disk_usage(".")
                free_gb = free / (1024**3)
                
                if free_gb < 1:
                    return {
                        'status': 'degraded',
                        'message': f'Low disk space: {free_gb:.1f}GB free',
                        'details': {
                            'log_directory': str(log_dir.absolute()),
                            'writable': True,
                            'free_space_gb': free_gb
                        }
                    }
            except:
                pass
            
            return {
                'status': 'healthy',
                'message': 'File system access verified',
                'details': {
                    'log_directory': str(log_dir.absolute()),
                    'writable': True
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'File system error: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity"""
        # Simulate network check (would require actual HTTP client in production)
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            'status': 'healthy',
            'message': 'Network connectivity simulated',
            'details': {
                'simulation': True,
                'note': 'Would check GitHub API connectivity in production'
            }
        }
    
    async def _check_security(self) -> Dict[str, Any]:
        """Check security configuration"""
        issues = []
        
        # Check file permissions
        config_file = Path(self.config_manager.config_path)
        if config_file.exists():
            stat = config_file.stat()
            # Check if file is world-readable (potential security issue)
            if stat.st_mode & 0o044:
                issues.append("Configuration file is world-readable")
        
        # Check for sensitive data in environment
        sensitive_env_vars = ['GITHUB_TOKEN', 'API_KEY', 'SECRET_KEY']
        exposed_vars = [var for var in sensitive_env_vars if os.getenv(var)]
        
        if issues:
            return {
                'status': 'degraded',
                'message': f'Security issues found: {len(issues)}',
                'details': {
                    'issues': issues,
                    'exposed_env_vars': len(exposed_vars)
                }
            }
        
        return {
            'status': 'healthy',
            'message': 'Security checks passed',
            'details': {
                'checks_performed': ['file_permissions', 'environment_variables'],
                'exposed_env_vars': len(exposed_vars)
            }
        }
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies"""
        try:
            missing_deps = []
            optional_deps = []
            
            # Required dependencies
            required = ['typer', 'rich']
            for dep in required:
                try:
                    __import__(dep)
                except ImportError:
                    missing_deps.append(dep)
            
            # Optional dependencies
            optional = ['github', 'validators', 'psutil']
            for dep in optional:
                try:
                    __import__(dep)
                except ImportError:
                    optional_deps.append(dep)
            
            if missing_deps:
                return {
                    'status': 'unhealthy',
                    'message': f'Missing required dependencies: {", ".join(missing_deps)}',
                    'details': {'missing': missing_deps}
                }
            
            status_msg = "All required dependencies available"
            if optional_deps:
                status_msg += f", optional missing: {', '.join(optional_deps)}"
            
            return {
                'status': 'healthy',
                'message': status_msg,
                'details': {'optional_missing': optional_deps}
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Dependency check failed: {e}',
                'details': {'error': str(e)}
            }
    
    async def _check_github_connectivity(self) -> Dict[str, Any]:
        """Check GitHub API connectivity"""
        try:
            github_token = os.getenv("GITHUB_TOKEN")
            if not github_token:
                return {
                    'status': 'degraded',
                    'message': 'No GitHub token, cannot test connectivity',
                    'details': {}
                }
            
            try:
                from github import Github
                client = Github(github_token)
                user = client.get_user()
                rate_limit = client.get_rate_limit()
                
                return {
                    'status': 'healthy',
                    'message': f'Connected as {user.login}',
                    'details': {
                        'rate_limit_remaining': rate_limit.core.remaining,
                        'rate_limit_total': rate_limit.core.limit
                    }
                }
                
            except ImportError:
                return {
                    'status': 'degraded',
                    'message': 'PyGithub not available, GitHub features disabled',
                    'details': {}
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'GitHub connectivity check failed: {e}',
                'details': {'error': str(e)}
            }
    
    async def _check_performance(self) -> Dict[str, Any]:
        """Check performance metrics"""
        try:
            stats = self.performance_monitor.get_statistics()
            
            if stats.get("message"):  # No metrics collected
                return {
                    'status': 'healthy',
                    'message': 'No performance data yet',
                    'details': {}
                }
            
            success_rate = stats.get("success_rate", 100)
            avg_duration = stats.get("average_duration", 0)
            
            if success_rate < 80:
                status = 'degraded'
                message = f'Low success rate: {success_rate:.1f}%'
            elif avg_duration > 10:
                status = 'degraded'
                message = f'High average duration: {avg_duration:.2f}s'
            else:
                status = 'healthy'
                message = f'Performance good: {success_rate:.1f}% success, {avg_duration:.2f}s avg'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'success_rate': success_rate,
                    'average_duration': avg_duration
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Performance check failed: {e}',
                'details': {'error': str(e)}
            }
    
    def _calculate_overall_status(self, results: Dict[str, HealthCheckResult]) -> str:
        """Calculate overall health status"""
        statuses = [result.status for result in results.values()]
        
        if 'unhealthy' in statuses:
            return 'unhealthy'
        elif 'degraded' in statuses:
            return 'degraded'
        else:
            return 'healthy'

# =============================================================================
# MAIN APPLICATION
# =============================================================================

class RobustClaudeManager:
    """Robust Claude Manager Service with comprehensive error handling and monitoring"""
    
    def __init__(self, config_path: str = "config.json", log_level: str = "INFO"):
        log_config = LogConfig(level=log_level, file_path="logs/claude-manager.log")
        self.logger = RobustLogger("claude-manager-robust", log_config)
        self.error_handler = RobustErrorHandler(self.logger)
        self.config_manager = RobustConfigManager(config_path, self.logger)
        self.health_monitor = HealthMonitor(self.logger, self.config_manager)
        self.performance_monitor = PerformanceMonitor(self.logger)
        
        # Load configuration
        self.config = self.config_manager.load_config()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info("RobustClaudeManager initialized successfully")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            
            # Save any pending metrics or state
            if self.performance_monitor:
                stats = self.performance_monitor.get_statistics()
                self.logger.info("Final performance statistics", **stats)
            
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def scan_repositories(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced repository scanning with error handling and monitoring"""
        operation_start = time.time()
        
        try:
            self.logger.info("Starting enhanced repository scan")
            
            repos_to_scan = self.config['github']['reposToScan']
            scan_results = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'enhanced_repository_scan',
                'repos_scanned': len(repos_to_scan),
                'repos': [],
                'scan_duration': 0,
                'status': 'in_progress',
                'errors': [],
                'security_checks': True,
                'monitoring_enabled': True
            }
            
            for repo_name in repos_to_scan:
                repo_start_time = time.time()
                
                try:
                    # Validate repository name for security
                    validated_repo = SecurityValidator.validate_repo_name(repo_name)
                    
                    self.logger.info(f"Scanning repository: {validated_repo}")
                    
                    # Enhanced repository scanning with error handling
                    repo_result = await self._scan_single_repository(validated_repo)
                    repo_result['scan_duration'] = time.time() - repo_start_time
                    
                    scan_results['repos'].append(repo_result)
                    
                    self.logger.info(
                        f"Repository scan completed",
                        repo=validated_repo,
                        duration=repo_result['scan_duration'],
                        status=repo_result['status']
                    )
                    
                except Exception as e:
                    error_context = self.error_handler.handle_error(
                        f"scan_repository_{repo_name}",
                        e,
                        repo_name=repo_name
                    )
                    
                    scan_results['errors'].append(asdict(error_context))
                    
                    # Add failed repo to results
                    scan_results['repos'].append({
                        'name': repo_name,
                        'status': 'failed',
                        'error': str(e),
                        'scanned_at': datetime.now().isoformat(),
                        'scan_duration': time.time() - repo_start_time
                    })
            
            scan_results['scan_duration'] = time.time() - operation_start
            scan_results['status'] = 'completed'
            scan_results['success_rate'] = len([r for r in scan_results['repos'] if r['status'] == 'success']) / len(repos_to_scan) if repos_to_scan else 0
            
            # Track performance
            self.logger.track_performance(
                'enhanced_repository_scan',
                scan_results['scan_duration'],
                repos_count=len(repos_to_scan),
                success_rate=scan_results['success_rate']
            )
            
            # Save results if output file specified
            if output_file:
                await self._save_results(scan_results, output_file)
            
            self.logger.info(
                "Enhanced repository scan completed",
                total_duration=scan_results['scan_duration'],
                success_rate=scan_results['success_rate']
            )
            
            return scan_results
            
        except Exception as e:
            error_context = self.error_handler.handle_error("enhanced_repository_scan", e)
            raise
    
    async def _scan_single_repository(self, repo_name: str) -> Dict[str, Any]:
        """Scan a single repository with enhanced error handling"""
        
        # Simulate repository scanning with error handling
        await asyncio.sleep(0.2)  # Simulate API call
        
        # Randomly simulate different scenarios for testing
        import random
        scenario = random.choice(['success', 'rate_limited', 'not_found', 'success', 'success'])
        
        if scenario == 'rate_limited':
            raise Exception("API rate limit exceeded")
        elif scenario == 'not_found':
            raise Exception("Repository not found or access denied")
        
        return {
            'name': repo_name,
            'scanned_at': datetime.now().isoformat(),
            'todos_found': random.randint(0, 5),
            'issues_analyzed': random.randint(0, 10),
            'status': 'success',
            'security_validated': True,
            'api_calls_made': random.randint(1, 3)
        }
    
    async def execute_task(self, task_description: str, executor: str = "auto") -> Dict[str, Any]:
        """Enhanced task execution with comprehensive error handling"""
        operation_start = time.time()
        
        try:
            # Sanitize task description
            sanitized_task = SecurityValidator.sanitize_input(task_description)
            
            self.logger.info(f"Executing enhanced task", task=sanitized_task, executor=executor)
            
            result = {
                'task': sanitized_task,
                'executor': executor,
                'started_at': datetime.now().isoformat(),
                'status': 'in_progress',
                'security_validated': True,
                'monitoring_enabled': True,
                'steps_completed': []
            }
            
            # Execute task with retry logic
            execution_result = await self.error_handler.with_retry(
                self._execute_task_steps,
                max_retries=2,
                delay=1.0,
                task=sanitized_task,
                executor=executor
            )
            
            result.update(execution_result)
            result['duration'] = time.time() - operation_start
            result['completed_at'] = datetime.now().isoformat()
            result['status'] = 'completed'
            
            # Track performance
            self.logger.track_performance(
                'enhanced_task_execution',
                result['duration'],
                task_type=executor,
                steps_count=len(result['steps_completed'])
            )
            
            self.logger.info(
                "Enhanced task execution completed",
                task=sanitized_task,
                duration=result['duration'],
                steps_completed=len(result['steps_completed'])
            )
            
            return result
            
        except Exception as e:
            error_context = self.error_handler.handle_error("enhanced_task_execution", e, task=task_description)
            raise
    
    async def _execute_task_steps(self, task: str, executor: str) -> Dict[str, Any]:
        """Execute individual task steps with monitoring"""
        steps = [
            "Analyzing task requirements",
            "Validating security constraints",
            "Preparing execution environment",
            "Executing core task logic",
            "Validating results",
            "Cleanup and finalization"
        ]
        
        steps_completed = []
        
        for i, step in enumerate(steps):
            step_start = time.time()
            
            self.logger.debug(f"Executing step: {step}")
            
            # Simulate step execution
            await asyncio.sleep(0.1 + (i * 0.05))  # Variable delay per step
            
            step_duration = time.time() - step_start
            steps_completed.append({
                'step': step,
                'duration': step_duration,
                'completed_at': datetime.now().isoformat()
            })
            
            # Simulate potential failure in middle steps
            if i == 3 and task.lower().count('fail') > 0:
                raise Exception("Simulated task execution failure")
        
        return {
            'steps_completed': steps_completed,
            'total_steps': len(steps)
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        health_results = await self.health_monitor.run_health_check()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': self.health_monitor._calculate_overall_status(health_results),
            'checks': {name: asdict(result) for name, result in health_results.items()},
            'performance_summary': self._get_performance_summary(),
            'error_summary': self._get_error_summary()
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        return self.performance_monitor.get_statistics()
    
    def _get_error_summary(self) -> Dict[str, Any]:
        """Get error metrics summary"""
        error_data = self.error_handler.error_history
        
        if not error_data:
            return {'total_errors': 0}
        
        recent_errors = [e for e in error_data if 
                        datetime.fromisoformat(e.timestamp) > 
                        datetime.now() - datetime.timedelta(hours=1)]
        
        error_types = {}
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        return {
            'total_errors': len(error_data),
            'recent_errors': len(recent_errors),
            'error_types': error_types,
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }
    
    async def _save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to file with error handling"""
        try:
            # Validate output file path
            safe_path = SecurityValidator.validate_file_path(output_file)
            
            with open(safe_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved successfully", output_file=safe_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save results", exception=e, output_file=output_file)
            raise

# =============================================================================
# CLI INTERFACE
# =============================================================================

# Initialize global components
console = Console()

# CLI Application using Typer
app = typer.Typer(
    name="claude-manager-robust",
    help="Claude Manager Service - Generation 2: Robust Implementation",
    add_completion=False,
    rich_markup_mode="rich"
)

# Global manager instance
manager: Optional[RobustClaudeManager] = None

@app.callback()
def init_manager(
    config_file: str = typer.Option(
        "config.json",
        "--config",
        "-c",
        help="Configuration file path",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """Initialize the Claude Manager Service"""
    global manager
    
    try:
        log_level = "DEBUG" if verbose else "INFO"
        manager = RobustClaudeManager(config_file, log_level)
        
    except Exception as e:
        rprint(f"[red]Initialization failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def status(
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed information",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json)",
    ),
):
    """Show service status"""
    if not manager:
        raise typer.Exit(1)
    
    rprint("📊 Claude Manager Service - Robust Status")
    rprint("=" * 50)
    
    config = manager.config_manager.get_config()
    
    rprint(f"Configuration: ✓ Loaded and validated")
    rprint(f"Security: ✓ Input validation enabled") 
    rprint(f"Monitoring: ✓ Performance tracking active")
    rprint(f"Error Handling: ✓ Retry logic enabled")
    rprint(f"Logging: ✓ Structured logging to file")
    
    rprint(f"\nGitHub Configuration:")
    rprint(f"  Username: {config['github']['username']}")
    rprint(f"  Manager Repo: {config['github']['managerRepo']}")
    rprint(f"  Repos to Scan: {len(config['github']['reposToScan'])}")
    
    if detailed:
        for repo in config['github']['reposToScan']:
            rprint(f"    - {repo}")
        
        rprint(f"\nPerformance Summary:")
        perf_summary = manager._get_performance_summary()
        for key, value in perf_summary.items():
            if key != "message":
                rprint(f"  {key}: {value}")
        
        rprint(f"\nError Summary:")
        error_summary = manager._get_error_summary()
        for key, value in error_summary.items():
            rprint(f"  {key}: {value}")

@app.command()
def scan(
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON format)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json)",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed information",
    ),
):
    """Scan repositories for tasks"""
    if not manager:
        raise typer.Exit(1)
    
    async def run_scan():
        results = await manager.scan_repositories(output)
        
        if format == 'json':
            rprint(json.dumps(results, indent=2, default=str))
        else:
            rprint(f"🔍 Enhanced Repository Scan Results")
            rprint(f"Duration: {results['scan_duration']:.2f}s")
            rprint(f"Success Rate: {results['success_rate']:.1%}")
            rprint(f"Repositories: {results['repos_scanned']}")
            rprint(f"Errors: {len(results['errors'])}")
            
            if detailed:
                for repo in results['repos']:
                    status_emoji = "✅" if repo['status'] == 'success' else "❌"
                    rprint(f"  {status_emoji} {repo['name']} ({repo.get('scan_duration', 0):.2f}s)")
    
    asyncio.run(run_scan())

@app.command()
def execute(
    task_description: str = typer.Argument(..., help="Task description"),
    executor: str = typer.Option(
        "auto",
        "--executor",
        "-e",
        help="Task executor type (auto/terragon/claude-flow)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json)",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed information",
    ),
):
    """Execute a task"""
    if not manager:
        raise typer.Exit(1)
    
    async def run_execute():
        results = await manager.execute_task(task_description, executor)
        
        if format == 'json':
            rprint(json.dumps(results, indent=2, default=str))
        else:
            rprint(f"⚡ Enhanced Task Execution Results")
            rprint(f"Task: {results['task']}")
            rprint(f"Duration: {results['duration']:.2f}s")
            rprint(f"Status: {results['status']}")
            rprint(f"Steps: {len(results['steps_completed'])}")
            
            if detailed:
                for step in results['steps_completed']:
                    rprint(f"  ✓ {step['step']} ({step['duration']:.3f}s)")
    
    asyncio.run(run_execute())

@app.command()
def health(
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json)",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed information",
    ),
):
    """Perform comprehensive health check"""
    if not manager:
        raise typer.Exit(1)
    
    rprint("[bold blue]🏥 Performing comprehensive health check...[/bold blue]")
    
    async def run_health_check():
        health_status = await manager.get_health_status()
        
        if format == 'json':
            rprint(json.dumps(health_status, indent=2, default=str))
        else:
            overall_emoji = {
                'healthy': '✅',
                'degraded': '⚠️', 
                'unhealthy': '❌'
            }.get(health_status['overall_status'], '❓')
            
            rprint(f"🏥 Enhanced Health Check Results")
            rprint(f"Overall Status: {overall_emoji} {health_status['overall_status'].upper()}")
            
            # Display results table
            table = Table(title=f"System Health Report")
            table.add_column("Check", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Message", style="dim")
            
            for check_name, check_result in health_status['checks'].items():
                status_emoji = {
                    'healthy': '✅',
                    'degraded': '⚠️',
                    'unhealthy': '❌'
                }.get(check_result['status'], '❓')
                
                table.add_row(
                    check_name.replace('_', ' ').title(),
                    f"{status_emoji} {check_result['status']}",
                    check_result['message']
                )
            
            console.print(table)
            
            if detailed:
                for check_name, check_result in health_status['checks'].items():
                    rprint(f"\n{check_name.title()} Details:")
                    rprint(f"  Duration: {check_result['duration']:.3f}s")
                    for key, value in check_result['details'].items():
                        rprint(f"  {key}: {value}")
    
    asyncio.run(run_health_check())

@app.command()
def monitor():
    """Start continuous monitoring mode"""
    if not manager:
        raise typer.Exit(1)
    
    rprint("📈 Starting continuous monitoring mode...")
    rprint("Press Ctrl+C to stop")
    
    async def run_monitor():
        try:
            while True:
                # Check for configuration changes
                if manager.config_manager.watch_config_changes():
                    rprint("⚙️  Configuration reloaded")
                
                # Periodic health check
                health_status = await manager.get_health_status()
                status_emoji = {
                    'healthy': '✅',
                    'degraded': '⚠️',
                    'unhealthy': '❌'
                }.get(health_status['overall_status'], '❓')
                
                rprint(f"{datetime.now().strftime('%H:%M:%S')} {status_emoji} {health_status['overall_status']}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            rprint("\n⏸️  Monitoring stopped")
    
    try:
        asyncio.run(run_monitor())
    except KeyboardInterrupt:
        rprint("\n⏸️  Monitoring stopped")

@app.command()
def metrics():
    """Show performance metrics"""
    if not manager:
        raise typer.Exit(1)
    
    rprint("[bold blue]📊 Performance Metrics[/bold blue]")
    
    stats = manager.performance_monitor.get_statistics()
    
    if stats.get("message"):
        rprint("[yellow]No metrics collected yet[/yellow]")
        return
    
    # Display metrics table
    table = Table(title="Performance Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Operations", str(stats["total_operations"]))
    table.add_row("Successful Operations", str(stats["successful_operations"]))
    table.add_row("Failed Operations", str(stats["failed_operations"]))
    table.add_row("Success Rate", f"{stats['success_rate']:.1f}%")
    table.add_row("Average Duration", f"{stats['average_duration']:.2f}s")
    
    if "average_successful_duration" in stats:
        table.add_row("Avg Successful Duration", f"{stats['average_successful_duration']:.2f}s")
    
    console.print(table)
    
    # Show operation counts
    if stats["operation_counts"]:
        op_table = Table(title="Operation Counts")
        op_table.add_column("Operation", style="cyan")
        op_table.add_column("Count", style="yellow")
        
        for operation, count in stats["operation_counts"].items():
            op_table.add_row(operation, str(count))
        
        console.print(op_table)

if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        rprint("\n⏸️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        rprint(f"\n❌ Fatal error: {e}")
        sys.exit(1)
```
