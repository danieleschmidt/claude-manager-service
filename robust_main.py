#!/usr/bin/env python3
"""
Claude Manager Service - Generation 2: MAKE IT ROBUST
Enhanced with comprehensive error handling, security, and monitoring
"""

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
import asyncio
import signal

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

# Enhanced error classes
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

# Security utilities
class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def sanitize_input(input_str: str, max_length: int = 1000) -> str:
        """Sanitize user input"""
        if not isinstance(input_str, str):
            raise ValidationError("Input must be a string")
        
        # Limit length
        if len(input_str) > max_length:
            raise ValidationError(f"Input too long (max {max_length} characters)")
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>&"\'`$();|]', '', input_str)
        
        # Additional validation for paths
        if '..' in sanitized or sanitized.startswith('/'):
            raise SecurityError("Potentially unsafe path detected")
        
        return sanitized.strip()
    
    @staticmethod
    def validate_repo_name(repo_name: str) -> bool:
        """Validate GitHub repository name format"""
        pattern = r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$'
        return bool(re.match(pattern, repo_name))
    
    @staticmethod
    def validate_github_token(token: str) -> bool:
        """Basic GitHub token validation"""
        if not token:
            return False
        
        # GitHub tokens are typically 40 characters
        if len(token) < 20 or len(token) > 100:
            return False
        
        # Should contain only alphanumeric characters and underscores
        pattern = r'^[a-zA-Z0-9_]+$'
        return bool(re.match(pattern, token))

# Logging configuration
@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

class EnhancedLogger:
    """Enhanced logging with security considerations"""
    
    def __init__(self, name: str, config: LogConfig):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.level))
        
        # Clear existing handlers
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
    
    def info(self, message: str, data: Any = None):
        if data:
            self.logger.info(f"{message}\nData: {self.sanitize_log_data(data)}")
        else:
            self.logger.info(message)
    
    def error(self, message: str, data: Any = None, exc_info: bool = True):
        if data:
            self.logger.error(f"{message}\nData: {self.sanitize_log_data(data)}", exc_info=exc_info)
        else:
            self.logger.error(message, exc_info=exc_info)
    
    def warning(self, message: str, data: Any = None):
        if data:
            self.logger.warning(f"{message}\nData: {self.sanitize_log_data(data)}")
        else:
            self.logger.warning(message)

# Error handling decorators
def handle_errors(logger: EnhancedLogger = None):
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

def async_handle_errors(logger: EnhancedLogger = None):
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

# Enhanced configuration with validation
class RobustConfig:
    """Robust configuration management with validation and security"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.logger = EnhancedLogger("RobustConfig", LogConfig())
        self.config = self._load_and_validate_config()
    
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self._validate_config(config)
                self.logger.info(f"Configuration loaded from {self.config_path}")
                return config
            else:
                self.logger.warning(f"Config file {self.config_path} not found, creating default")
                return self._create_default_config()
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure and values"""
        required_sections = ['github', 'analyzer', 'executor']
        
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required config section: {section}")
        
        # Validate GitHub configuration
        github_config = config.get('github', {})
        if 'username' not in github_config:
            raise ConfigurationError("Missing github.username in configuration")
        
        if 'reposToScan' in github_config:
            for repo in github_config['reposToScan']:
                if not SecurityValidator.validate_repo_name(repo):
                    raise ConfigurationError(f"Invalid repository name format: {repo}")
        
        # Validate other sections
        analyzer_config = config.get('analyzer', {})
        if not isinstance(analyzer_config.get('scanForTodos', True), bool):
            raise ConfigurationError("analyzer.scanForTodos must be boolean")
        
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

# Performance monitoring
@dataclass
class PerformanceMetrics:
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    memory_usage: Optional[float] = None

class PerformanceMonitor:
    """Monitor performance and collect metrics"""
    
    def __init__(self, logger: EnhancedLogger):
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

# Enhanced health checker
class RobustHealthCheck:
    """Comprehensive health checking with monitoring"""
    
    def __init__(self, config: RobustConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.performance_monitor = PerformanceMonitor(logger)
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        start_time = self.performance_monitor.start_operation("system_health_check")
        
        try:
            checks = {}
            
            # Configuration check
            checks["configuration"] = self._check_configuration()
            
            # Environment check
            checks["environment"] = self._check_environment()
            
            # Dependencies check
            checks["dependencies"] = self._check_dependencies()
            
            # GitHub connectivity check
            checks["github_connectivity"] = self._check_github_connectivity()
            
            # File system check
            checks["file_system"] = self._check_file_system()
            
            # Performance check
            checks["performance"] = self._check_performance()
            
            # Calculate overall health
            healthy_checks = sum(1 for check in checks.values() if check.get("status") == "OK")
            total_checks = len(checks)
            health_percentage = (healthy_checks / total_checks) * 100
            
            overall_health = {
                "overall_status": "HEALTHY" if health_percentage >= 80 else "DEGRADED" if health_percentage >= 60 else "UNHEALTHY",
                "health_percentage": health_percentage,
                "checks": checks,
                "timestamp": datetime.now().isoformat()
            }
            
            self.performance_monitor.end_operation("system_health_check", start_time, True)
            return overall_health
            
        except Exception as e:
            self.performance_monitor.end_operation("system_health_check", start_time, False, str(e))
            raise
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration health"""
        try:
            # Test configuration access
            username = self.config.get("github.username")
            repos = self.config.get("github.reposToScan", [])
            
            if not username or username == "your-username":
                return {
                    "status": "WARNING",
                    "message": "GitHub username not configured"
                }
            
            if not repos:
                return {
                    "status": "WARNING", 
                    "message": "No repositories configured for scanning"
                }
            
            return {
                "status": "OK",
                "message": f"Configuration valid, {len(repos)} repositories configured"
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Configuration check failed: {e}"
            }
    
    def _check_environment(self) -> Dict[str, Any]:
        """Check environment health"""
        try:
            github_token = os.getenv("GITHUB_TOKEN")
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            issues = []
            if not github_token:
                issues.append("GITHUB_TOKEN not set")
            elif not SecurityValidator.validate_github_token(github_token):
                issues.append("GITHUB_TOKEN appears invalid")
            
            if issues:
                return {
                    "status": "WARNING",
                    "message": f"Environment issues: {', '.join(issues)}",
                    "python_version": python_version
                }
            
            return {
                "status": "OK",
                "message": f"Environment healthy, Python {python_version}",
                "python_version": python_version
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Environment check failed: {e}"
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
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
                    "status": "ERROR",
                    "message": f"Missing required dependencies: {', '.join(missing_deps)}"
                }
            
            status_msg = "All required dependencies available"
            if optional_deps:
                status_msg += f", optional missing: {', '.join(optional_deps)}"
            
            return {
                "status": "OK",
                "message": status_msg
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Dependency check failed: {e}"
            }
    
    def _check_github_connectivity(self) -> Dict[str, Any]:
        """Check GitHub API connectivity"""
        try:
            github_token = os.getenv("GITHUB_TOKEN")
            if not github_token:
                return {
                    "status": "WARNING",
                    "message": "No GitHub token, cannot test connectivity"
                }
            
            try:
                from github import Github
                client = Github(github_token)
                user = client.get_user()
                rate_limit = client.get_rate_limit()
                
                return {
                    "status": "OK",
                    "message": f"Connected as {user.login}",
                    "rate_limit_remaining": rate_limit.core.remaining,
                    "rate_limit_total": rate_limit.core.limit
                }
                
            except ImportError:
                return {
                    "status": "WARNING",
                    "message": "PyGithub not available, GitHub features disabled"
                }
                
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"GitHub connectivity check failed: {e}"
            }
    
    def _check_file_system(self) -> Dict[str, Any]:
        """Check file system health"""
        try:
            import tempfile
            
            # Test write access
            with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                tmp_file.write(b"test")
                tmp_file.flush()
            
            # Check disk space
            try:
                import shutil
                total, used, free = shutil.disk_usage(".")
                free_gb = free / (1024**3)
                
                if free_gb < 1:
                    return {
                        "status": "WARNING",
                        "message": f"Low disk space: {free_gb:.1f}GB free"
                    }
                
                return {
                    "status": "OK",
                    "message": f"File system healthy, {free_gb:.1f}GB free"
                }
                
            except Exception:
                return {
                    "status": "OK",
                    "message": "File system writable (disk space check unavailable)"
                }
                
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"File system check failed: {e}"
            }
    
    def _check_performance(self) -> Dict[str, Any]:
        """Check performance metrics"""
        try:
            stats = self.performance_monitor.get_statistics()
            
            if stats.get("message"):  # No metrics collected
                return {
                    "status": "OK",
                    "message": "No performance data yet"
                }
            
            success_rate = stats.get("success_rate", 100)
            avg_duration = stats.get("average_duration", 0)
            
            if success_rate < 80:
                status = "WARNING"
                message = f"Low success rate: {success_rate:.1f}%"
            elif avg_duration > 10:
                status = "WARNING"
                message = f"High average duration: {avg_duration:.2f}s"
            else:
                status = "OK"
                message = f"Performance good: {success_rate:.1f}% success, {avg_duration:.2f}s avg"
            
            return {
                "status": status,
                "message": message,
                "success_rate": success_rate,
                "average_duration": avg_duration
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Performance check failed: {e}"
            }

# Initialize global components
logger = EnhancedLogger("RobustClaudeManager", LogConfig(level="INFO"))
config: Optional[RobustConfig] = None
health_checker: Optional[RobustHealthCheck] = None
performance_monitor: Optional[PerformanceMonitor] = None

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    rprint("\n[yellow]Graceful shutdown initiated...[/yellow]")
    
    # Save any pending metrics or state
    if performance_monitor:
        stats = performance_monitor.get_statistics()
        logger.info("Final performance statistics", stats)
    
    rprint("[green]Shutdown complete[/green]")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# CLI Application
app = typer.Typer(
    name="claude-manager-robust",
    help="Claude Manager Service - Generation 2: Robust Implementation",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()

@app.callback()
def main(
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
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        help="Log file path",
    ),
):
    """Claude Manager Service - Generation 2: Robust Implementation"""
    global config, health_checker, performance_monitor, logger
    
    try:
        # Update logger configuration
        log_config = LogConfig(
            level="DEBUG" if verbose else "INFO",
            file_path=log_file
        )
        logger = EnhancedLogger("RobustClaudeManager", log_config)
        
        # Initialize configuration
        config = RobustConfig(config_file)
        
        # Initialize health checker and performance monitor
        health_checker = RobustHealthCheck(config, logger)
        performance_monitor = PerformanceMonitor(logger)
        
        logger.info("Claude Manager initialized successfully")
        
    except Exception as e:
        rprint(f"[red]Initialization failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
@handle_errors(logger)
def health():
    """Perform comprehensive health check"""
    if not health_checker:
        raise ClaudeManagerError("Health checker not initialized")
    
    rprint("[bold blue]🏥 Performing comprehensive health check...[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running health checks...", total=None)
        
        health_report = health_checker.check_system_health()
        
        progress.update(task, completed=True)
    
    # Display results
    table = Table(title=f"System Health Report - {health_report['overall_status']}")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Message", style="dim")
    
    status_colors = {
        "OK": "green",
        "WARNING": "yellow", 
        "ERROR": "red"
    }
    
    for check_name, result in health_report["checks"].items():
        status = result["status"]
        color = status_colors.get(status, "white")
        
        table.add_row(
            check_name.replace("_", " ").title(),
            f"[{color}]{status}[/{color}]",
            result["message"]
        )
    
    console.print(table)
    
    # Show overall health
    overall_color = status_colors.get(health_report["overall_status"].split()[0], "white")
    rprint(f"\n[{overall_color}]Overall Health: {health_report['overall_status']} ({health_report['health_percentage']:.1f}%)[/{overall_color}]")

@app.command()
@handle_errors(logger)
def scan(
    path: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="Path to scan",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o", 
        help="Output file for results",
    ),
    max_results: int = typer.Option(
        100,
        "--max-results",
        "-m",
        help="Maximum number of results",
    ),
):
    """Scan for tasks with robust error handling"""
    start_time = performance_monitor.start_operation("scan_operation")
    
    try:
        # Validate inputs
        path = SecurityValidator.sanitize_input(path, 200)
        if output:
            output = SecurityValidator.sanitize_input(output, 200)
        
        if max_results <= 0 or max_results > 10000:
            raise ValidationError("max_results must be between 1 and 10000")
        
        rprint("[bold blue]🔍 Scanning for tasks with robust validation...[/bold blue]")
        
        # Import and use the enhanced task analyzer
        from simple_main import SimpleTaskAnalyzer
        analyzer = SimpleTaskAnalyzer(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning repository...", total=None)
            
            tasks = analyzer.scan_repository(path)
            
            # Limit results
            if len(tasks) > max_results:
                logger.warning(f"Limiting results from {len(tasks)} to {max_results}")
                tasks = tasks[:max_results]
            
            progress.update(task, completed=True)
        
        if tasks:
            # Display results with enhanced formatting
            table = Table(title=f"Found {len(tasks)} tasks (secure scan)")
            table.add_column("Priority", style="yellow", width=8)
            table.add_column("Type", style="cyan", width=15)
            table.add_column("Title", style="green")
            table.add_column("File", style="dim", width=30)
            
            for task in sorted(tasks, key=lambda x: x.get("priority", 0), reverse=True):
                # Sanitize display data
                title = task["title"][:50] + "..." if len(task["title"]) > 50 else task["title"]
                file_path = task.get("file_path", "N/A")
                if len(file_path) > 30:
                    file_path = "..." + file_path[-27:]
                
                table.add_row(
                    str(task.get("priority", 0)),
                    task.get("type", "unknown"),
                    title,
                    file_path
                )
            
            console.print(table)
            
            # Save to file with security validation
            if output:
                try:
                    # Validate output path
                    output_path = Path(output)
                    if not output_path.parent.exists():
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_path, 'w') as f:
                        json.dump(tasks, f, indent=2, default=str)
                    
                    logger.info(f"Results saved securely to {output}")
                    rprint(f"[green]✓[/green] Results saved to {output}")
                    
                except Exception as e:
                    logger.error(f"Failed to save results: {e}")
                    raise ValidationError(f"Could not save to {output}: {e}")
        else:
            rprint("[green]✓[/green] No issues found!")
        
        performance_monitor.end_operation("scan_operation", start_time, True)
        
    except Exception as e:
        performance_monitor.end_operation("scan_operation", start_time, False, str(e))
        raise

@app.command()
@handle_errors(logger)
def metrics():
    """Show performance metrics"""
    if not performance_monitor:
        raise ClaudeManagerError("Performance monitor not initialized")
    
    rprint("[bold blue]📊 Performance Metrics[/bold blue]")
    
    stats = performance_monitor.get_statistics()
    
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

@app.command()
@handle_errors(logger)
def validate_config():
    """Validate current configuration"""
    if not config:
        raise ClaudeManagerError("Configuration not initialized")
    
    rprint("[bold blue]🔧 Validating Configuration[/bold blue]")
    
    try:
        # Re-validate configuration
        config._validate_config(config.config)
        
        # Show configuration summary
        table = Table(title="Configuration Validation")
        table.add_column("Section", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        # GitHub section
        github_config = config.get("github", {})
        username = github_config.get("username", "")
        repos = github_config.get("reposToScan", [])
        
        if username and username != "your-username":
            github_status = "✓ Configured"
            github_details = f"User: {username}, Repos: {len(repos)}"
        else:
            github_status = "⚠ Needs Setup"
            github_details = "Username not configured"
        
        table.add_row("GitHub", github_status, github_details)
        
        # Analyzer section
        analyzer_config = config.get("analyzer", {})
        scan_todos = analyzer_config.get("scanForTodos", False)
        scan_issues = analyzer_config.get("scanOpenIssues", False)
        
        analyzer_status = "✓ Configured" if scan_todos or scan_issues else "⚠ Disabled"
        analyzer_details = f"TODOs: {scan_todos}, Issues: {scan_issues}"
        
        table.add_row("Analyzer", analyzer_status, analyzer_details)
        
        # Security section
        security_config = config.get("security", {})
        validation_enabled = security_config.get("enableValidation", False)
        
        security_status = "✓ Enabled" if validation_enabled else "⚠ Disabled"
        security_details = f"Validation: {validation_enabled}"
        
        table.add_row("Security", security_status, security_details)
        
        console.print(table)
        rprint("[green]✓ Configuration validation passed[/green]")
        
    except ConfigurationError as e:
        rprint(f"[red]✗ Configuration validation failed: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()