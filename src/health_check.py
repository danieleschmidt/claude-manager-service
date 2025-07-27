"""
Health check module for Claude Code Manager.
Provides comprehensive health monitoring endpoints and system status checks.
"""

import asyncio
import json
import os
import psutil
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

import requests
from flask import Flask, jsonify, Response


class HealthChecker:
    """Comprehensive health checker for all system components."""
    
    def __init__(self):
        self.start_time = time.time()
        self.check_history: List[Dict[str, Any]] = []
        
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "checks": {}
        }
        
        # System resource checks
        health_status["checks"]["system"] = self._check_system_resources()
        
        # Database connectivity
        health_status["checks"]["database"] = self._check_database()
        
        # External service connectivity
        health_status["checks"]["github_api"] = self._check_github_api()
        
        # File system checks
        health_status["checks"]["filesystem"] = self._check_filesystem()
        
        # Application-specific checks
        health_status["checks"]["application"] = self._check_application()
        
        # Performance metrics
        health_status["checks"]["performance"] = self._check_performance()
        
        # Determine overall status
        failed_checks = [
            name for name, check in health_status["checks"].items()
            if check.get("status") != "healthy"
        ]
        
        if failed_checks:
            health_status["status"] = "degraded" if len(failed_checks) <= 2 else "unhealthy"
            health_status["failed_checks"] = failed_checks
        
        # Store in history
        self.check_history.append(health_status)
        if len(self.check_history) > 100:  # Keep last 100 checks
            self.check_history.pop(0)
        
        return health_status
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            issues = []
            
            if cpu_percent > 90:
                status = "degraded"
                issues.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 90:
                status = "degraded"
                issues.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                status = "degraded"
                issues.append(f"High disk usage: {disk.percent}%")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "issues": issues
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and health."""
        try:
            from src.services.database_service import DatabaseService
            
            db = DatabaseService()
            
            # Simple connectivity test
            start_time = time.time()
            result = db.execute_query("SELECT 1")
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "connection_pool_size": getattr(db, 'pool_size', 'unknown')
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _check_github_api(self) -> Dict[str, Any]:
        """Check GitHub API connectivity and rate limits."""
        try:
            github_token = os.getenv('GITHUB_TOKEN')
            if not github_token:
                return {
                    "status": "degraded",
                    "error": "No GitHub token configured"
                }
            
            headers = {'Authorization': f'token {github_token}'}
            start_time = time.time()
            
            response = requests.get(
                'https://api.github.com/rate_limit',
                headers=headers,
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                rate_limit_data = response.json()
                remaining = rate_limit_data['rate']['remaining']
                
                status = "healthy"
                if remaining < 100:
                    status = "degraded"
                
                return {
                    "status": status,
                    "response_time_ms": round(response_time * 1000, 2),
                    "rate_limit_remaining": remaining,
                    "rate_limit_reset": rate_limit_data['rate']['reset']
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"GitHub API returned {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _check_filesystem(self) -> Dict[str, Any]:
        """Check filesystem health and required directories."""
        try:
            required_dirs = ['data', 'logs', 'temp', 'backups']
            issues = []
            
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    issues.append(f"Created missing directory: {dir_name}")
                
                # Check write permissions
                test_file = dir_path / '.health_check_test'
                try:
                    test_file.write_text('test')
                    test_file.unlink()
                except Exception:
                    issues.append(f"No write permission for directory: {dir_name}")
            
            return {
                "status": "healthy" if not issues else "degraded",
                "required_directories": required_dirs,
                "issues": issues
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _check_application(self) -> Dict[str, Any]:
        """Check application-specific health metrics."""
        try:
            from src.services.task_service import TaskService
            
            task_service = TaskService()
            
            # Get recent task statistics
            pending_tasks = task_service.count_tasks_by_status('pending')
            running_tasks = task_service.count_tasks_by_status('running')
            failed_tasks = task_service.count_tasks_by_status('failed')
            
            status = "healthy"
            issues = []
            
            if failed_tasks > 10:
                status = "degraded"
                issues.append(f"High number of failed tasks: {failed_tasks}")
            
            if running_tasks > 50:
                status = "degraded"
                issues.append(f"High number of running tasks: {running_tasks}")
            
            return {
                "status": status,
                "pending_tasks": pending_tasks,
                "running_tasks": running_tasks,
                "failed_tasks": failed_tasks,
                "issues": issues
            }
        except Exception as e:
            return {
                "status": "degraded",
                "error": str(e),
                "note": "Application services not fully initialized"
            }
    
    def _check_performance(self) -> Dict[str, Any]:
        """Check performance metrics and thresholds."""
        try:
            from src.performance_monitor import PerformanceMonitor
            
            monitor = PerformanceMonitor()
            metrics = monitor.get_current_metrics()
            
            status = "healthy"
            issues = []
            
            # Check response time thresholds
            avg_response_time = metrics.get('avg_response_time', 0)
            if avg_response_time > 1000:  # 1 second
                status = "degraded"
                issues.append(f"High average response time: {avg_response_time}ms")
            
            # Check error rate
            error_rate = metrics.get('error_rate', 0)
            if error_rate > 0.05:  # 5%
                status = "degraded"
                issues.append(f"High error rate: {error_rate * 100:.1f}%")
            
            return {
                "status": status,
                "avg_response_time_ms": avg_response_time,
                "error_rate": error_rate,
                "throughput_rps": metrics.get('requests_per_second', 0),
                "issues": issues
            }
        except Exception as e:
            return {
                "status": "degraded",
                "error": str(e),
                "note": "Performance monitoring not available"
            }
    
    def get_readiness(self) -> Dict[str, Any]:
        """Check if the application is ready to serve requests."""
        readiness_status = {
            "ready": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {}
        }
        
        # Check critical dependencies
        readiness_status["checks"]["database"] = self._check_database()
        readiness_status["checks"]["filesystem"] = self._check_filesystem()
        
        # Determine overall readiness
        for check_name, check_result in readiness_status["checks"].items():
            if check_result.get("status") in ["unhealthy", "critical"]:
                readiness_status["ready"] = False
                break
        
        return readiness_status
    
    def get_liveness(self) -> Dict[str, Any]:
        """Check if the application is alive and responsive."""
        return {
            "alive": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "pid": os.getpid()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Prometheus-style metrics."""
        try:
            system_metrics = self._check_system_resources()
            
            metrics = {
                "# HELP system_cpu_usage_percent Current CPU usage percentage",
                "# TYPE system_cpu_usage_percent gauge",
                f"system_cpu_usage_percent {system_metrics.get('cpu_percent', 0)}",
                "",
                "# HELP system_memory_usage_percent Current memory usage percentage", 
                "# TYPE system_memory_usage_percent gauge",
                f"system_memory_usage_percent {system_metrics.get('memory_percent', 0)}",
                "",
                "# HELP system_disk_usage_percent Current disk usage percentage",
                "# TYPE system_disk_usage_percent gauge", 
                f"system_disk_usage_percent {system_metrics.get('disk_percent', 0)}",
                "",
                "# HELP application_uptime_seconds Application uptime in seconds",
                "# TYPE application_uptime_seconds counter",
                f"application_uptime_seconds {time.time() - self.start_time}",
                ""
            ]
            
            return "\n".join(metrics)
        except Exception as e:
            return f"# Error generating metrics: {str(e)}"


# Global health checker instance
health_checker = HealthChecker()


def check() -> Dict[str, Any]:
    """Main health check function for use in Docker healthcheck."""
    return health_checker.check_system_health()


def create_health_app() -> Flask:
    """Create Flask app with health endpoints."""
    app = Flask(__name__)
    
    @app.route('/health')
    def health():
        """Comprehensive health check endpoint."""
        health_status = health_checker.check_system_health()
        status_code = 200 if health_status["status"] == "healthy" else 503
        return jsonify(health_status), status_code
    
    @app.route('/health/ready')
    def readiness():
        """Readiness probe endpoint."""
        readiness_status = health_checker.get_readiness()
        status_code = 200 if readiness_status["ready"] else 503
        return jsonify(readiness_status), status_code
    
    @app.route('/health/live')
    def liveness():
        """Liveness probe endpoint."""
        liveness_status = health_checker.get_liveness()
        return jsonify(liveness_status), 200
    
    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint."""
        metrics_data = health_checker.get_metrics()
        return Response(metrics_data, mimetype='text/plain')
    
    @app.route('/status')
    def status():
        """Detailed status information."""
        return jsonify({
            "service": "Claude Code Manager",
            "version": "0.1.0",
            "environment": os.getenv("FLASK_ENV", "production"),
            "health": health_checker.check_system_health(),
            "readiness": health_checker.get_readiness(),
            "liveness": health_checker.get_liveness()
        })
    
    return app


if __name__ == "__main__":
    # Command line health check
    import sys
    
    result = check()
    print(json.dumps(result, indent=2))
    
    if result["status"] != "healthy":
        sys.exit(1)