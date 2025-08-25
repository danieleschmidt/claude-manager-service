#!/usr/bin/env python3
"""
ROBUST HEALTH MONITORING SYSTEM - Generation 2
Comprehensive health checks, system monitoring, and alerting
"""

import asyncio
import json
import os
import psutil
import time
import aiohttp
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum

from src.logger import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    response_time_ms: float = 0.0


@dataclass 
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_connections: int
    uptime_seconds: float
    load_average: List[float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RobustHealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.health_checks = {}
        self.metrics_history = []
        self.alert_thresholds = {
            "cpu_threshold": self.config.get("cpu_threshold", 85.0),
            "memory_threshold": self.config.get("memory_threshold", 90.0),
            "disk_threshold": self.config.get("disk_threshold", 85.0),
            "response_time_threshold": self.config.get("response_time_threshold", 5000.0)
        }
        self.start_time = time.time()
        
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register a custom health check"""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def perform_comprehensive_health_check(self) -> Dict[str, HealthCheckResult]:
        """Perform all registered health checks"""
        results = {}
        
        # Core system checks
        results.update(await self._check_system_resources())
        results.update(await self._check_disk_space())
        results.update(await self._check_network_connectivity())
        results.update(await self._check_database_connectivity())
        results.update(await self._check_external_services())
        
        # Custom health checks
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = await check_func()
                response_time = (time.time() - start_time) * 1000
                
                if isinstance(result, HealthCheckResult):
                    result.response_time_ms = response_time
                    results[name] = result
                else:
                    # Convert simple boolean or dict result
                    status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                    results[name] = HealthCheckResult(
                        name=name,
                        status=status,
                        message="Custom health check completed",
                        response_time_ms=response_time
                    )
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}"
                )
        
        return results
    
    async def _check_system_resources(self) -> Dict[str, HealthCheckResult]:
        """Check system resource utilization"""
        results = {}
        
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = HealthStatus.HEALTHY
            if cpu_percent > self.alert_thresholds["cpu_threshold"]:
                cpu_status = HealthStatus.CRITICAL
            elif cpu_percent > self.alert_thresholds["cpu_threshold"] * 0.8:
                cpu_status = HealthStatus.DEGRADED
            
            results["cpu_usage"] = HealthCheckResult(
                name="cpu_usage",
                status=cpu_status,
                message=f"CPU usage: {cpu_percent:.1f}%",
                details={"cpu_percent": cpu_percent, "threshold": self.alert_thresholds["cpu_threshold"]}
            )
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_status = HealthStatus.HEALTHY
            if memory.percent > self.alert_thresholds["memory_threshold"]:
                memory_status = HealthStatus.CRITICAL
            elif memory.percent > self.alert_thresholds["memory_threshold"] * 0.8:
                memory_status = HealthStatus.DEGRADED
            
            results["memory_usage"] = HealthCheckResult(
                name="memory_usage",
                status=memory_status,
                message=f"Memory usage: {memory.percent:.1f}%",
                details={
                    "memory_percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "threshold": self.alert_thresholds["memory_threshold"]
                }
            )
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            results["system_resources"] = HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}"
            )
        
        return results
    
    async def _check_disk_space(self) -> Dict[str, HealthCheckResult]:
        """Check disk space availability"""
        results = {}
        
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            disk_status = HealthStatus.HEALTHY
            if disk_percent > self.alert_thresholds["disk_threshold"]:
                disk_status = HealthStatus.CRITICAL
            elif disk_percent > self.alert_thresholds["disk_threshold"] * 0.8:
                disk_status = HealthStatus.DEGRADED
            
            results["disk_space"] = HealthCheckResult(
                name="disk_space",
                status=disk_status,
                message=f"Disk usage: {disk_percent:.1f}%",
                details={
                    "disk_percent": disk_percent,
                    "free_gb": disk_usage.free / (1024**3),
                    "total_gb": disk_usage.total / (1024**3),
                    "threshold": self.alert_thresholds["disk_threshold"]
                }
            )
            
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            results["disk_space"] = HealthCheckResult(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}"
            )
        
        return results
    
    async def _check_network_connectivity(self) -> Dict[str, HealthCheckResult]:
        """Check network connectivity"""
        results = {}
        
        try:
            # Test DNS resolution and connectivity
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get('https://api.github.com/') as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        message = "Network connectivity healthy"
                    else:
                        status = HealthStatus.DEGRADED
                        message = f"Network connectivity degraded (HTTP {response.status})"
                    
                    results["network_connectivity"] = HealthCheckResult(
                        name="network_connectivity", 
                        status=status,
                        message=message,
                        details={
                            "response_time_ms": response_time,
                            "status_code": response.status
                        },
                        response_time_ms=response_time
                    )
                    
        except asyncio.TimeoutError:
            results["network_connectivity"] = HealthCheckResult(
                name="network_connectivity",
                status=HealthStatus.CRITICAL,
                message="Network connectivity timeout"
            )
        except Exception as e:
            results["network_connectivity"] = HealthCheckResult(
                name="network_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Network connectivity failed: {str(e)}"
            )
        
        return results
    
    async def _check_database_connectivity(self) -> Dict[str, HealthCheckResult]:
        """Check database connectivity"""
        results = {}
        
        try:
            # For SQLite, check if file is accessible
            db_path = Path("tasks.db")
            if db_path.exists():
                # Simple file access test
                start_time = time.time()
                with open(db_path, 'rb') as f:
                    f.read(1024)  # Read first 1KB
                response_time = (time.time() - start_time) * 1000
                
                results["database_connectivity"] = HealthCheckResult(
                    name="database_connectivity",
                    status=HealthStatus.HEALTHY,
                    message="Database accessible",
                    details={"database_path": str(db_path)},
                    response_time_ms=response_time
                )
            else:
                results["database_connectivity"] = HealthCheckResult(
                    name="database_connectivity",
                    status=HealthStatus.DEGRADED,
                    message="Database file not found (will be created on first use)",
                    details={"database_path": str(db_path)}
                )
                
        except Exception as e:
            results["database_connectivity"] = HealthCheckResult(
                name="database_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Database connectivity failed: {str(e)}"
            )
        
        return results
    
    async def _check_external_services(self) -> Dict[str, HealthCheckResult]:
        """Check external service dependencies"""
        results = {}
        
        # Check GitHub API (without authentication)
        try:
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get('https://api.github.com/rate_limit') as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        remaining_requests = data.get('rate', {}).get('remaining', 0)
                        
                        if remaining_requests > 100:
                            status = HealthStatus.HEALTHY
                            message = "GitHub API healthy"
                        else:
                            status = HealthStatus.DEGRADED
                            message = f"GitHub API rate limit low ({remaining_requests} remaining)"
                        
                        results["github_api"] = HealthCheckResult(
                            name="github_api",
                            status=status,
                            message=message,
                            details={
                                "remaining_requests": remaining_requests,
                                "response_time_ms": response_time
                            },
                            response_time_ms=response_time
                        )
                    else:
                        results["github_api"] = HealthCheckResult(
                            name="github_api",
                            status=HealthStatus.DEGRADED,
                            message=f"GitHub API returned HTTP {response.status}"
                        )
                        
        except Exception as e:
            results["github_api"] = HealthCheckResult(
                name="github_api",
                status=HealthStatus.CRITICAL,
                message=f"GitHub API check failed: {str(e)}"
            )
        
        return results
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network_connections = len(psutil.net_connections())
            uptime_seconds = time.time() - self.start_time
            
            # Load average (Unix-like systems)
            try:
                load_average = list(os.getloadavg())
            except (OSError, AttributeError):
                # Windows or other systems without load average
                load_average = [0.0, 0.0, 0.0]
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                network_connections=network_connections,
                uptime_seconds=uptime_seconds,
                load_average=load_average
            )
            
            # Store metrics history (keep last 100 measurements)
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_connections=0,
                uptime_seconds=0.0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def get_health_summary(self, results: Dict[str, HealthCheckResult]) -> Dict[str, Any]:
        """Generate health summary from check results"""
        total_checks = len(results)
        healthy_count = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
        critical_count = sum(1 for r in results.values() if r.status == HealthStatus.CRITICAL)
        
        # Overall system health
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall_status.value,
            "total_checks": total_checks,
            "healthy": healthy_count,
            "degraded": degraded_count,
            "unhealthy": unhealthy_count,
            "critical": critical_count,
            "health_score": (healthy_count / total_checks * 100) if total_checks > 0 else 0,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        health_results = await self.perform_comprehensive_health_check()
        system_metrics = self.collect_system_metrics()
        health_summary = self.get_health_summary(health_results)
        
        return {
            "health_summary": health_summary,
            "health_checks": {name: asdict(result) for name, result in health_results.items()},
            "system_metrics": asdict(system_metrics),
            "metrics_history_count": len(self.metrics_history),
            "uptime_hours": system_metrics.uptime_seconds / 3600,
            "report_generated_at": datetime.now(timezone.utc).isoformat()
        }


# Health monitoring factory and utilities
def create_robust_health_monitor(config: Optional[Dict[str, Any]] = None) -> RobustHealthMonitor:
    """Create robust health monitoring system"""
    return RobustHealthMonitor(config)


async def run_health_check_demo():
    """Demonstration of health monitoring system"""
    logger.info("Starting robust health monitoring demo")
    
    # Create health monitor
    monitor = create_robust_health_monitor({
        "cpu_threshold": 80.0,
        "memory_threshold": 85.0,
        "disk_threshold": 90.0
    })
    
    # Register custom health check
    async def custom_config_check():
        """Custom check for configuration file"""
        config_path = Path("config.json")
        if config_path.exists():
            return HealthCheckResult(
                name="configuration_file",
                status=HealthStatus.HEALTHY,
                message="Configuration file exists and is accessible"
            )
        else:
            return HealthCheckResult(
                name="configuration_file",
                status=HealthStatus.DEGRADED,
                message="Configuration file not found"
            )
    
    monitor.register_health_check("config_file", custom_config_check)
    
    # Generate health report
    report = await monitor.generate_health_report()
    
    # Display summary
    summary = report["health_summary"]
    logger.info(f"Health Status: {summary['overall_status']}")
    logger.info(f"Health Score: {summary['health_score']:.1f}%")
    logger.info(f"Checks: {summary['healthy']}/{summary['total_checks']} healthy")
    
    # Display failed checks
    for name, result in report["health_checks"].items():
        if result["status"] != "healthy":
            logger.warning(f"Check {name}: {result['status']} - {result['message']}")
    
    logger.info("Robust health monitoring demo completed")


if __name__ == "__main__":
    asyncio.run(run_health_check_demo())