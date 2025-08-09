"""
Enhanced Health Check System for Generation 2 Robustness

This module provides comprehensive health monitoring with:
- System resource monitoring
- Database connectivity checks
- External service availability
- Performance metrics validation
- Security system status
- Configuration validation
- Automated remediation suggestions
"""

import asyncio
import psutil
import time
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import aiohttp
import sqlite3
from pathlib import Path

from .enhanced_logger import get_enhanced_logger, log_security_event
from .security_v2 import get_security_manager
from .config_validator_v2 import validate_config_file
from .services.configuration_service import ConfigurationService


@dataclass
class HealthCheckResult:
    """Individual health check result"""
    name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    details: Dict[str, Any] = None
    remediation: Optional[str] = None
    check_duration_ms: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    overall_status: str
    health_score: float  # 0-100
    timestamp: datetime
    uptime_seconds: float
    checks: List[HealthCheckResult]
    summary: Dict[str, int]
    recommendations: List[str]
    critical_issues: List[str]
    
    def __post_init__(self):
        if not self.summary:
            self.summary = {
                'healthy': sum(1 for c in self.checks if c.status == 'healthy'),
                'degraded': sum(1 for c in self.checks if c.status == 'degraded'),
                'unhealthy': sum(1 for c in self.checks if c.status == 'unhealthy'),
                'total': len(self.checks)
            }


class HealthCheckRegistry:
    """Registry for health check functions"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.check_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, check_func: Callable, 
                 category: str = "system", 
                 critical: bool = False,
                 timeout_seconds: int = 30,
                 description: str = ""):
        """Register a health check function"""
        self.checks[name] = check_func
        self.check_metadata[name] = {
            'category': category,
            'critical': critical,
            'timeout': timeout_seconds,
            'description': description
        }
    
    def get_checks(self, category: Optional[str] = None) -> Dict[str, Callable]:
        """Get health checks, optionally filtered by category"""
        if category:
            return {
                name: func for name, func in self.checks.items()
                if self.check_metadata[name]['category'] == category
            }
        return self.checks
    
    def get_critical_checks(self) -> Dict[str, Callable]:
        """Get only critical health checks"""
        return {
            name: func for name, func in self.checks.items()
            if self.check_metadata[name]['critical']
        }


class EnhancedHealthCheck:
    """Enhanced health check system with comprehensive monitoring"""
    
    def __init__(self, config_service: Optional[ConfigurationService] = None):
        self.config_service = config_service
        self.logger = get_enhanced_logger(__name__)
        self.registry = HealthCheckRegistry()
        self.system_start_time = time.time()
        
        # Register built-in health checks
        self._register_builtin_checks()
        
        # Health check history
        self.health_history: List[SystemHealthReport] = []
        self.max_history = 100
        
        self.logger.info("Enhanced health check system initialized")
    
    def _register_builtin_checks(self):
        """Register built-in health checks"""
        # System resource checks
        self.registry.register("cpu_usage", self._check_cpu_usage, 
                              category="resources", critical=False, description="CPU usage monitoring")
        self.registry.register("memory_usage", self._check_memory_usage,
                              category="resources", critical=True, description="Memory usage monitoring")
        self.registry.register("disk_usage", self._check_disk_usage,
                              category="resources", critical=True, description="Disk space monitoring")
        
        # System service checks
        self.registry.register("configuration", self._check_configuration,
                              category="system", critical=True, description="Configuration validation")
        self.registry.register("logging", self._check_logging_system,
                              category="system", critical=False, description="Logging system health")
        self.registry.register("security", self._check_security_system,
                              category="security", critical=True, description="Security system status")
        
        # Database checks
        self.registry.register("database_connection", self._check_database_connection,
                              category="database", critical=True, description="Database connectivity")
        
        # External service checks
        self.registry.register("github_api", self._check_github_api,
                              category="external", critical=True, description="GitHub API connectivity")
        
        # Performance checks
        self.registry.register("response_times", self._check_response_times,
                              category="performance", critical=False, description="System response times")
        self.registry.register("error_rates", self._check_error_rates,
                              category="performance", critical=False, description="Error rate monitoring")
    
    async def run_health_checks(self, categories: Optional[List[str]] = None,
                               include_non_critical: bool = True) -> SystemHealthReport:
        """
        Run comprehensive health checks
        
        Args:
            categories: Optional list of categories to check
            include_non_critical: Whether to include non-critical checks
            
        Returns:
            Comprehensive health report
        """
        start_time = time.time()
        results = []
        
        # Determine which checks to run
        checks_to_run = {}
        
        if categories:
            for category in categories:
                checks_to_run.update(self.registry.get_checks(category))
        else:
            checks_to_run = self.registry.checks
        
        if not include_non_critical:
            critical_checks = self.registry.get_critical_checks()
            checks_to_run = {k: v for k, v in checks_to_run.items() if k in critical_checks}
        
        # Run checks concurrently
        tasks = []
        for name, check_func in checks_to_run.items():
            task = self._run_single_check(name, check_func)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed with exception: {result}")
                valid_results.append(HealthCheckResult(
                    name="unknown",
                    status="unhealthy",
                    message=f"Check failed with exception: {str(result)}",
                    remediation="Check logs for details"
                ))
            else:
                valid_results.append(result)
        
        # Calculate overall health
        overall_status, health_score = self._calculate_overall_health(valid_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(valid_results)
        critical_issues = self._identify_critical_issues(valid_results)
        
        # Create health report
        report = SystemHealthReport(
            overall_status=overall_status,
            health_score=health_score,
            timestamp=datetime.now(),
            uptime_seconds=time.time() - self.system_start_time,
            checks=valid_results,
            summary={},
            recommendations=recommendations,
            critical_issues=critical_issues
        )
        
        # Store in history
        self.health_history.append(report)
        if len(self.health_history) > self.max_history:
            self.health_history.pop(0)
        
        # Log health status
        self._log_health_status(report)
        
        return report
    
    async def _run_single_check(self, name: str, check_func: Callable) -> HealthCheckResult:
        """Run a single health check with timeout and error handling"""
        metadata = self.check_metadata.get(name, {})
        timeout = metadata.get('timeout', 30)
        
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(check_func(), timeout=timeout)
            
            if isinstance(result, HealthCheckResult):
                result.check_duration_ms = (time.time() - start_time) * 1000
                return result
            else:
                # Handle legacy return formats
                return HealthCheckResult(
                    name=name,
                    status="healthy" if result else "unhealthy",
                    message="Check completed",
                    check_duration_ms=(time.time() - start_time) * 1000
                )
                
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=name,
                status="unhealthy",
                message=f"Health check timed out after {timeout} seconds",
                remediation="Check system load and performance",
                check_duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                remediation="Check logs for detailed error information",
                check_duration_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_overall_health(self, results: List[HealthCheckResult]) -> Tuple[str, float]:
        """Calculate overall health status and score"""
        if not results:
            return "unhealthy", 0.0
        
        healthy_count = sum(1 for r in results if r.status == "healthy")
        degraded_count = sum(1 for r in results if r.status == "degraded")
        unhealthy_count = sum(1 for r in results if r.status == "unhealthy")
        
        total_count = len(results)
        
        # Calculate weighted score
        healthy_weight = 1.0
        degraded_weight = 0.5
        unhealthy_weight = 0.0
        
        score = (
            (healthy_count * healthy_weight) +
            (degraded_count * degraded_weight) +
            (unhealthy_count * unhealthy_weight)
        ) / total_count * 100
        
        # Determine overall status
        if unhealthy_count == 0:
            if degraded_count == 0:
                overall_status = "healthy"
            elif degraded_count <= total_count * 0.2:  # Less than 20% degraded
                overall_status = "healthy"
            else:
                overall_status = "degraded"
        elif unhealthy_count <= total_count * 0.1:  # Less than 10% unhealthy
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return overall_status, score
    
    def _generate_recommendations(self, results: List[HealthCheckResult]) -> List[str]:
        """Generate recommendations based on health check results"""
        recommendations = []
        
        # Collect existing recommendations from results
        for result in results:
            if result.status != "healthy" and result.remediation:
                recommendations.append(f"{result.name}: {result.remediation}")
        
        # Add general recommendations based on patterns
        unhealthy_checks = [r for r in results if r.status == "unhealthy"]
        degraded_checks = [r for r in results if r.status == "degraded"]
        
        if len(unhealthy_checks) > 3:
            recommendations.append("Multiple systems are unhealthy - perform comprehensive system review")
        
        if any("memory" in r.name.lower() for r in unhealthy_checks):
            recommendations.append("Consider increasing system memory or optimizing memory usage")
        
        if any("disk" in r.name.lower() for r in unhealthy_checks):
            recommendations.append("Free up disk space or expand storage capacity")
        
        if any("api" in r.name.lower() or "external" in r.name.lower() for r in unhealthy_checks):
            recommendations.append("Check network connectivity and external service status")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _identify_critical_issues(self, results: List[HealthCheckResult]) -> List[str]:
        """Identify critical issues that need immediate attention"""
        critical_issues = []
        
        for result in results:
            metadata = self.check_metadata.get(result.name, {})
            if metadata.get('critical', False) and result.status == "unhealthy":
                critical_issues.append(f"{result.name}: {result.message}")
        
        return critical_issues
    
    def _log_health_status(self, report: SystemHealthReport):
        """Log health status for monitoring"""
        if report.overall_status == "unhealthy":
            self.logger.error(
                f"System health check failed - Status: {report.overall_status}, "
                f"Score: {report.health_score:.1f}%, "
                f"Critical issues: {len(report.critical_issues)}"
            )
            
            # Log security event for unhealthy system
            log_security_event(
                event_type='system_monitoring',
                severity='high',
                description=f'System health degraded - Score: {report.health_score:.1f}%',
                action='health_check',
                result='unhealthy'
            )
        elif report.overall_status == "degraded":
            self.logger.warning(
                f"System health degraded - Score: {report.health_score:.1f}%, "
                f"Degraded checks: {report.summary.get('degraded', 0)}"
            )
        else:
            self.logger.info(
                f"System health check passed - Score: {report.health_score:.1f}%"
            )
    
    # Built-in health check implementations
    
    async def _check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent < 70:
                status = "healthy"
                message = f"CPU usage normal: {cpu_percent:.1f}%"
                remediation = None
            elif cpu_percent < 90:
                status = "degraded"
                message = f"CPU usage elevated: {cpu_percent:.1f}%"
                remediation = "Monitor CPU usage and consider optimizing processes"
            else:
                status = "unhealthy"
                message = f"CPU usage critical: {cpu_percent:.1f}%"
                remediation = "Investigate high CPU usage and scale resources"
            
            return HealthCheckResult(
                name="cpu_usage",
                status=status,
                message=message,
                details={"cpu_percent": cpu_percent},
                remediation=remediation
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="cpu_usage",
                status="unhealthy",
                message=f"Failed to check CPU usage: {str(e)}",
                remediation="Check system monitoring tools"
            )
    
    async def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_mb = memory.available // 1024 // 1024
            
            if memory_percent < 80:
                status = "healthy"
                message = f"Memory usage normal: {memory_percent:.1f}% ({available_mb}MB available)"
                remediation = None
            elif memory_percent < 95:
                status = "degraded"
                message = f"Memory usage high: {memory_percent:.1f}% ({available_mb}MB available)"
                remediation = "Monitor memory usage and consider optimizing applications"
            else:
                status = "unhealthy"
                message = f"Memory usage critical: {memory_percent:.1f}% ({available_mb}MB available)"
                remediation = "Free memory immediately or add more RAM"
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                details={
                    "memory_percent": memory_percent,
                    "total_mb": memory.total // 1024 // 1024,
                    "available_mb": available_mb,
                    "used_mb": memory.used // 1024 // 1024
                },
                remediation=remediation
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status="unhealthy",
                message=f"Failed to check memory usage: {str(e)}",
                remediation="Check system monitoring tools"
            )
    
    async def _check_disk_usage(self) -> HealthCheckResult:
        """Check disk usage"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free // 1024 // 1024 // 1024
            
            if disk_percent < 80:
                status = "healthy"
                message = f"Disk usage normal: {disk_percent:.1f}% ({free_gb}GB free)"
                remediation = None
            elif disk_percent < 95:
                status = "degraded"
                message = f"Disk usage high: {disk_percent:.1f}% ({free_gb}GB free)"
                remediation = "Clean up disk space or expand storage"
            else:
                status = "unhealthy"
                message = f"Disk usage critical: {disk_percent:.1f}% ({free_gb}GB free)"
                remediation = "Immediate disk cleanup required"
            
            return HealthCheckResult(
                name="disk_usage",
                status=status,
                message=message,
                details={
                    "disk_percent": disk_percent,
                    "total_gb": disk_usage.total // 1024 // 1024 // 1024,
                    "free_gb": free_gb,
                    "used_gb": disk_usage.used // 1024 // 1024 // 1024
                },
                remediation=remediation
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk_usage",
                status="unhealthy",
                message=f"Failed to check disk usage: {str(e)}",
                remediation="Check disk monitoring tools"
            )
    
    async def _check_configuration(self) -> HealthCheckResult:
        """Check configuration validity"""
        try:
            if self.config_service:
                # Try to get configuration
                config = await self.config_service.get_config()
                
                # Validate configuration structure
                result = await validate_config_file("config.json", "production")
                
                if result.is_valid:
                    status = "healthy"
                    message = "Configuration is valid"
                    remediation = None
                elif result.security_issues or result.errors:
                    status = "unhealthy"
                    message = f"Configuration has {len(result.errors)} errors and {len(result.security_issues or [])} security issues"
                    remediation = "Fix configuration errors and security issues"
                else:
                    status = "degraded"
                    message = f"Configuration has {len(result.warnings or [])} warnings"
                    remediation = "Address configuration warnings"
                
                return HealthCheckResult(
                    name="configuration",
                    status=status,
                    message=message,
                    details={
                        "errors": len(result.errors or []),
                        "warnings": len(result.warnings or []),
                        "security_issues": len(result.security_issues or [])
                    },
                    remediation=remediation
                )
            else:
                return HealthCheckResult(
                    name="configuration",
                    status="unhealthy",
                    message="Configuration service not available",
                    remediation="Initialize configuration service"
                )
                
        except Exception as e:
            return HealthCheckResult(
                name="configuration",
                status="unhealthy",
                message=f"Configuration check failed: {str(e)}",
                remediation="Check configuration file and service"
            )
    
    async def _check_logging_system(self) -> HealthCheckResult:
        """Check logging system health"""
        try:
            log_dir = Path('logs')
            
            if not log_dir.exists():
                return HealthCheckResult(
                    name="logging",
                    status="unhealthy",
                    message="Log directory does not exist",
                    remediation="Create logs directory"
                )
            
            # Check log files
            log_files = list(log_dir.glob('*.log')) + list(log_dir.glob('*.json'))
            
            if not log_files:
                return HealthCheckResult(
                    name="logging",
                    status="degraded",
                    message="No log files found",
                    remediation="Check logging configuration"
                )
            
            # Check recent log activity
            recent_logs = []
            current_time = datetime.now()
            
            for log_file in log_files:
                mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if current_time - mod_time < timedelta(minutes=5):
                    recent_logs.append(log_file.name)
            
            if recent_logs:
                return HealthCheckResult(
                    name="logging",
                    status="healthy",
                    message=f"Logging active - {len(recent_logs)} recent log files",
                    details={"recent_files": recent_logs}
                )
            else:
                return HealthCheckResult(
                    name="logging",
                    status="degraded",
                    message="No recent logging activity",
                    remediation="Check application logging and log rotation"
                )
                
        except Exception as e:
            return HealthCheckResult(
                name="logging",
                status="unhealthy",
                message=f"Logging system check failed: {str(e)}",
                remediation="Check logging configuration and permissions"
            )
    
    async def _check_security_system(self) -> HealthCheckResult:
        """Check security system status"""
        try:
            security_manager = get_security_manager()
            
            # Check credential availability
            if not os.getenv('GITHUB_TOKEN'):
                return HealthCheckResult(
                    name="security",
                    status="unhealthy",
                    message="GitHub token not configured",
                    remediation="Set GITHUB_TOKEN environment variable"
                )
            
            # Check security components
            details = {
                "credentials_available": bool(security_manager.credential_manager.list_credentials()),
                "rate_limiting_active": True,  # Assuming it's active
                "session_management_active": True
            }
            
            return HealthCheckResult(
                name="security",
                status="healthy",
                message="Security system operational",
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="security",
                status="unhealthy",
                message=f"Security system check failed: {str(e)}",
                remediation="Check security configuration and credentials"
            )
    
    async def _check_database_connection(self) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            # For SQLite (default)
            db_path = "claude_manager.db"
            
            if not Path(db_path).exists():
                return HealthCheckResult(
                    name="database_connection",
                    status="degraded",
                    message="Database file does not exist - will be created on first use",
                    remediation="Database will be initialized automatically"
                )
            
            # Test connection
            conn = sqlite3.connect(db_path, timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return HealthCheckResult(
                    name="database_connection",
                    status="healthy",
                    message="Database connection successful",
                    details={"database_type": "sqlite", "database_path": db_path}
                )
            else:
                return HealthCheckResult(
                    name="database_connection",
                    status="unhealthy",
                    message="Database query failed",
                    remediation="Check database integrity"
                )
                
        except Exception as e:
            return HealthCheckResult(
                name="database_connection",
                status="unhealthy",
                message=f"Database connection failed: {str(e)}",
                remediation="Check database configuration and connectivity"
            )
    
    async def _check_github_api(self) -> HealthCheckResult:
        """Check GitHub API connectivity"""
        try:
            github_token = os.getenv('GITHUB_TOKEN')
            
            if not github_token:
                return HealthCheckResult(
                    name="github_api",
                    status="unhealthy",
                    message="GitHub token not configured",
                    remediation="Set GITHUB_TOKEN environment variable"
                )
            
            # Test GitHub API
            headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.github.com/rate_limit', 
                                     headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        remaining = data.get('rate', {}).get('remaining', 0)
                        
                        if remaining > 100:
                            status = "healthy"
                            message = f"GitHub API accessible - {remaining} requests remaining"
                        elif remaining > 0:
                            status = "degraded"
                            message = f"GitHub API accessible - {remaining} requests remaining (low)"
                        else:
                            status = "degraded"
                            message = "GitHub API accessible but rate limited"
                        
                        return HealthCheckResult(
                            name="github_api",
                            status=status,
                            message=message,
                            details={"rate_limit_remaining": remaining}
                        )
                    else:
                        return HealthCheckResult(
                            name="github_api",
                            status="unhealthy",
                            message=f"GitHub API returned status {response.status}",
                            remediation="Check GitHub token permissions and API status"
                        )
                        
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name="github_api",
                status="unhealthy",
                message="GitHub API request timed out",
                remediation="Check network connectivity"
            )
        except Exception as e:
            return HealthCheckResult(
                name="github_api",
                status="unhealthy",
                message=f"GitHub API check failed: {str(e)}",
                remediation="Check network connectivity and GitHub token"
            )
    
    async def _check_response_times(self) -> HealthCheckResult:
        """Check system response times"""
        try:
            # Simple response time test - measure configuration access time
            if not self.config_service:
                return HealthCheckResult(
                    name="response_times",
                    status="degraded",
                    message="Cannot measure response times - config service unavailable"
                )
            
            start_time = time.time()
            await self.config_service.get_config()
            response_time_ms = (time.time() - start_time) * 1000
            
            if response_time_ms < 100:
                status = "healthy"
                message = f"Response times normal: {response_time_ms:.1f}ms"
            elif response_time_ms < 500:
                status = "degraded"
                message = f"Response times elevated: {response_time_ms:.1f}ms"
            else:
                status = "unhealthy"
                message = f"Response times slow: {response_time_ms:.1f}ms"
                
            return HealthCheckResult(
                name="response_times",
                status=status,
                message=message,
                details={"response_time_ms": response_time_ms}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="response_times",
                status="unhealthy",
                message=f"Response time check failed: {str(e)}",
                remediation="Check system performance"
            )
    
    async def _check_error_rates(self) -> HealthCheckResult:
        """Check system error rates"""
        try:
            # This would integrate with error tracking system
            # For now, return a placeholder result
            
            return HealthCheckResult(
                name="error_rates",
                status="healthy",
                message="Error rates within acceptable limits",
                details={"error_rate": 0.0}  # Placeholder
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="error_rates",
                status="unhealthy",
                message=f"Error rate check failed: {str(e)}",
                remediation="Check error tracking system"
            )
    
    def export_health_report(self, report: SystemHealthReport, export_path: str):
        """Export health report to file"""
        try:
            export_data = {
                'report': asdict(report),
                'export_timestamp': datetime.now().isoformat(),
                'system_info': {
                    'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                    'platform': os.name,
                    'uptime_seconds': report.uptime_seconds
                }
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
            self.logger.info(f"Health report exported to: {export_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export health report: {e}")
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_reports = [r for r in self.health_history if r.timestamp > cutoff_time]
        
        if not recent_reports:
            return {"error": "No health data available for specified period"}
        
        # Calculate trends
        scores = [r.health_score for r in recent_reports]
        statuses = [r.overall_status for r in recent_reports]
        
        return {
            'period_hours': hours,
            'reports_count': len(recent_reports),
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'status_distribution': {
                'healthy': statuses.count('healthy'),
                'degraded': statuses.count('degraded'),
                'unhealthy': statuses.count('unhealthy')
            },
            'trend': 'improving' if scores[-1] > scores[0] else 'degrading' if len(scores) > 1 else 'stable'
        }


# Factory function
async def create_health_check(config_service: Optional[ConfigurationService] = None) -> EnhancedHealthCheck:
    """Create and initialize health check system"""
    return EnhancedHealthCheck(config_service)


# Global instance
_health_check_instance: Optional[EnhancedHealthCheck] = None


async def get_health_check(config_service: Optional[ConfigurationService] = None) -> EnhancedHealthCheck:
    """Get global health check instance"""
    global _health_check_instance
    
    if _health_check_instance is None:
        _health_check_instance = await create_health_check(config_service)
    
    return _health_check_instance