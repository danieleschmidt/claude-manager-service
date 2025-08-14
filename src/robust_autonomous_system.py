#!/usr/bin/env python3
"""
Robust Autonomous System - Generation 2
Enhanced reliability, error handling, monitoring, and health checks
"""

import asyncio
import json
import logging
import time
import os
import threading
import traceback
import queue
import signal
import sys
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import subprocess

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('robust_system.log')
    ]
)
logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System state enumeration"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    SHUTDOWN = "shutdown"

class HealthStatus(Enum):
    """Health check status"""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable
    interval: int = 30  # seconds
    timeout: int = 10   # seconds
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True
    last_check: float = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0

@dataclass
class SystemMetrics:
    """System metrics data"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    error_rate: float
    response_time_avg: float
    uptime: float

@dataclass
class ErrorInfo:
    """Error information tracking"""
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    count: int = 1
    first_seen: datetime = None
    last_seen: datetime = None
    
    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = self.timestamp
        self.last_seen = self.timestamp

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class RetryManager:
    """Advanced retry mechanism with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_multiplier: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                if attempt >= self.max_retries:
                    logger.error(f"Function failed after {self.max_retries + 1} attempts: {e}")
                    raise e
                
                delay = min(
                    self.base_delay * (self.backoff_multiplier ** attempt),
                    self.max_delay
                )
                
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)

class RobustAutonomousSystem:
    """
    Robust autonomous system with comprehensive error handling and monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.state = SystemState.INITIALIZING
        self.health_checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager()
        self.metrics_history: List[SystemMetrics] = []
        self.error_tracking: Dict[str, ErrorInfo] = {}
        self.task_queue = queue.Queue()
        self.running = False
        self.start_time = time.time()
        self.shutdown_event = threading.Event()
        
        # Performance monitoring
        self.task_completion_times = []
        self.error_counts = {"total": 0, "by_type": {}}
        
        # Initialize health checks
        self._initialize_health_checks()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("Robust Autonomous System initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default system configuration"""
        return {
            "health_check_interval": 30,
            "metrics_retention_hours": 24,
            "error_tracking_enabled": True,
            "circuit_breaker_enabled": True,
            "auto_recovery_enabled": True,
            "max_task_queue_size": 1000,
            "performance_monitoring": True,
            "alert_thresholds": {
                "error_rate": 0.05,  # 5%
                "cpu_usage": 0.8,    # 80%
                "memory_usage": 0.85, # 85%
                "response_time": 5.0   # 5 seconds
            }
        }
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown_event.set()
    
    def _initialize_health_checks(self):
        """Initialize system health checks"""
        self.health_checks = {
            "system_resources": HealthCheck(
                name="System Resources",
                check_function=self._check_system_resources,
                interval=30,
                timeout=10
            ),
            "task_queue": HealthCheck(
                name="Task Queue Health",
                check_function=self._check_task_queue_health,
                interval=60,
                timeout=5
            ),
            "error_rate": HealthCheck(
                name="Error Rate Monitor",
                check_function=self._check_error_rate,
                interval=120,
                timeout=5
            ),
            "file_system": HealthCheck(
                name="File System Access",
                check_function=self._check_file_system,
                interval=300,
                timeout=15
            )
        }
    
    async def start_system(self):
        """Start the robust autonomous system"""
        try:
            logger.info("Starting Robust Autonomous System")
            self.running = True
            self.state = SystemState.HEALTHY
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self._health_monitor_loop()),
                asyncio.create_task(self._metrics_collector_loop()),
                asyncio.create_task(self._task_processor_loop()),
                asyncio.create_task(self._error_analyzer_loop())
            ]
            
            # Wait for shutdown signal
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)
            
            logger.info("Shutdown signal received, stopping system")
            self.state = SystemState.SHUTDOWN
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            self.state = SystemState.UNHEALTHY
            raise
        finally:
            self.running = False
            logger.info("Robust Autonomous System stopped")
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        try:
            while self.running and not self.shutdown_event.is_set():
                await self._run_health_checks()
                await asyncio.sleep(self.config["health_check_interval"])
                
        except asyncio.CancelledError:
            logger.info("Health monitor loop cancelled")
        except Exception as e:
            logger.error(f"Health monitor loop error: {e}")
            await self._handle_critical_error("health_monitor", e)
    
    async def _run_health_checks(self):
        """Execute all health checks"""
        overall_status = HealthStatus.OK
        failed_checks = []
        
        for check_name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
                
            try:
                # Check if it's time to run this check
                if time.time() - health_check.last_check < health_check.interval:
                    continue
                
                # Execute health check with timeout
                status = await asyncio.wait_for(
                    self._execute_health_check(health_check),
                    timeout=health_check.timeout
                )
                
                health_check.last_check = time.time()
                
                if status == HealthStatus.OK:
                    health_check.consecutive_successes += 1
                    health_check.consecutive_failures = 0
                else:
                    health_check.consecutive_failures += 1
                    health_check.consecutive_successes = 0
                    failed_checks.append(check_name)
                    
                    if status == HealthStatus.CRITICAL:
                        overall_status = HealthStatus.CRITICAL
                    elif status == HealthStatus.WARNING and overall_status == HealthStatus.OK:
                        overall_status = HealthStatus.WARNING
                
                # Handle recovery
                if (health_check.consecutive_successes >= health_check.recovery_threshold and
                    self.state == SystemState.DEGRADED):
                    self.state = SystemState.RECOVERING
                
                # Handle failures
                if health_check.consecutive_failures >= health_check.failure_threshold:
                    if overall_status == HealthStatus.CRITICAL:
                        self.state = SystemState.UNHEALTHY
                    else:
                        self.state = SystemState.DEGRADED
                        
            except asyncio.TimeoutError:
                logger.warning(f"Health check '{check_name}' timed out")
                health_check.consecutive_failures += 1
                failed_checks.append(check_name)
                
            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                health_check.consecutive_failures += 1
                failed_checks.append(check_name)
        
        # Update system state based on overall health
        if overall_status == HealthStatus.OK and failed_checks:
            self.state = SystemState.HEALTHY if self.state == SystemState.RECOVERING else self.state
        
        if failed_checks:
            logger.warning(f"Failed health checks: {failed_checks}")
    
    async def _execute_health_check(self, health_check: HealthCheck) -> HealthStatus:
        """Execute individual health check"""
        try:
            if asyncio.iscoroutinefunction(health_check.check_function):
                result = await health_check.check_function()
            else:
                result = health_check.check_function()
            
            return result if isinstance(result, HealthStatus) else HealthStatus.OK
            
        except Exception as e:
            logger.error(f"Health check '{health_check.name}' execution failed: {e}")
            return HealthStatus.CRITICAL
    
    def _check_system_resources(self) -> HealthStatus:
        """Check system resource utilization"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            thresholds = self.config["alert_thresholds"]
            
            if (cpu_percent > thresholds["cpu_usage"] * 100 or
                memory_percent > thresholds["memory_usage"] * 100 or
                disk_percent > 90):
                return HealthStatus.CRITICAL
            
            if (cpu_percent > thresholds["cpu_usage"] * 80 or
                memory_percent > thresholds["memory_usage"] * 80 or
                disk_percent > 80):
                return HealthStatus.WARNING
            
            return HealthStatus.OK
            
        except ImportError:
            # Fallback if psutil not available
            return HealthStatus.UNKNOWN
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return HealthStatus.CRITICAL
    
    def _check_task_queue_health(self) -> HealthStatus:
        """Check task queue health"""
        try:
            queue_size = self.task_queue.qsize()
            max_size = self.config.get("max_task_queue_size", 1000)
            
            if queue_size > max_size * 0.9:
                return HealthStatus.CRITICAL
            elif queue_size > max_size * 0.7:
                return HealthStatus.WARNING
            
            return HealthStatus.OK
            
        except Exception as e:
            logger.error(f"Task queue health check failed: {e}")
            return HealthStatus.CRITICAL
    
    def _check_error_rate(self) -> HealthStatus:
        """Check system error rate"""
        try:
            if not self.metrics_history:
                return HealthStatus.OK
            
            recent_metrics = self.metrics_history[-10:]  # Last 10 metrics
            avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
            
            threshold = self.config["alert_thresholds"]["error_rate"]
            
            if avg_error_rate > threshold * 2:
                return HealthStatus.CRITICAL
            elif avg_error_rate > threshold:
                return HealthStatus.WARNING
            
            return HealthStatus.OK
            
        except Exception as e:
            logger.error(f"Error rate check failed: {e}")
            return HealthStatus.CRITICAL
    
    def _check_file_system(self) -> HealthStatus:
        """Check file system access and permissions"""
        try:
            # Test file operations
            test_file = Path("health_check_test.tmp")
            
            # Write test
            with open(test_file, 'w') as f:
                f.write("health check")
            
            # Read test
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Cleanup
            test_file.unlink()
            
            if content != "health check":
                return HealthStatus.CRITICAL
            
            return HealthStatus.OK
            
        except Exception as e:
            logger.error(f"File system check failed: {e}")
            return HealthStatus.CRITICAL
    
    async def _metrics_collector_loop(self):
        """Background metrics collection loop"""
        try:
            while self.running and not self.shutdown_event.is_set():
                await self._collect_metrics()
                await asyncio.sleep(60)  # Collect metrics every minute
                
        except asyncio.CancelledError:
            logger.info("Metrics collector loop cancelled")
        except Exception as e:
            logger.error(f"Metrics collector loop error: {e}")
    
    async def _collect_metrics(self):
        """Collect system metrics"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Calculate error rate
            total_tasks = len(self.task_completion_times) + self.error_counts["total"]
            error_rate = self.error_counts["total"] / max(total_tasks, 1)
            
            # Calculate average response time
            avg_response_time = (
                sum(self.task_completion_times) / len(self.task_completion_times)
                if self.task_completion_times else 0
            )
            
            # System resource metrics
            try:
                import psutil
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent / 100
                disk_usage = psutil.disk_usage('/').percent / 100
            except ImportError:
                cpu_usage = memory_usage = disk_usage = 0
            
            metrics = SystemMetrics(
                timestamp=current_time,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                active_tasks=self.task_queue.qsize(),
                completed_tasks=len(self.task_completion_times),
                failed_tasks=self.error_counts["total"],
                error_rate=error_rate,
                response_time_avg=avg_response_time,
                uptime=time.time() - self.start_time
            )
            
            self.metrics_history.append(metrics)
            
            # Cleanup old metrics (keep only last 24 hours)
            retention_hours = self.config.get("metrics_retention_hours", 24)
            cutoff_time = current_time.timestamp() - (retention_hours * 3600)
            
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp.timestamp() > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    async def _task_processor_loop(self):
        """Background task processing loop"""
        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Check if system is healthy enough to process tasks
                    if self.state == SystemState.UNHEALTHY:
                        await asyncio.sleep(5)
                        continue
                    
                    # Get task from queue (non-blocking)
                    try:
                        task_data = self.task_queue.get_nowait()
                        await self._process_task_safely(task_data)
                        self.task_queue.task_done()
                    except queue.Empty:
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Task processing error: {e}")
                    await self._record_error("task_processing", e)
                    
        except asyncio.CancelledError:
            logger.info("Task processor loop cancelled")
        except Exception as e:
            logger.error(f"Task processor loop error: {e}")
    
    async def _process_task_safely(self, task_data: Dict[str, Any]):
        """Process task with error handling and monitoring"""
        start_time = time.time()
        task_id = task_data.get("id", "unknown")
        
        try:
            logger.info(f"Processing task {task_id}")
            
            # Use retry manager for task execution
            result = await self.retry_manager.execute_with_retry(
                self._execute_task, task_data
            )
            
            # Record successful completion
            completion_time = time.time() - start_time
            self.task_completion_times.append(completion_time)
            
            # Limit stored completion times to last 1000
            if len(self.task_completion_times) > 1000:
                self.task_completion_times = self.task_completion_times[-1000:]
            
            logger.info(f"Task {task_id} completed in {completion_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            await self._record_error("task_execution", e, {"task_id": task_id})
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Any:
        """Execute individual task (placeholder)"""
        # This is a placeholder - implement actual task execution logic
        task_type = task_data.get("type", "unknown")
        
        if task_type == "discovery":
            # Simulate task discovery
            await asyncio.sleep(0.1)
            return {"status": "completed", "discovered": True}
        elif task_type == "analysis":
            # Simulate analysis
            await asyncio.sleep(0.2)
            return {"status": "completed", "analyzed": True}
        else:
            # Default processing
            await asyncio.sleep(0.05)
            return {"status": "completed", "processed": True}
    
    async def _error_analyzer_loop(self):
        """Background error analysis loop"""
        try:
            while self.running and not self.shutdown_event.is_set():
                await self._analyze_error_patterns()
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
        except asyncio.CancelledError:
            logger.info("Error analyzer loop cancelled")
        except Exception as e:
            logger.error(f"Error analyzer loop error: {e}")
    
    async def _analyze_error_patterns(self):
        """Analyze error patterns and suggest improvements"""
        try:
            if not self.error_tracking:
                return
            
            # Find most common errors
            sorted_errors = sorted(
                self.error_tracking.values(),
                key=lambda e: e.count,
                reverse=True
            )
            
            for error_info in sorted_errors[:5]:  # Top 5 errors
                if error_info.count > 10:  # Frequent error threshold
                    logger.warning(f"Frequent error detected: {error_info.error_type} "
                                 f"({error_info.count} occurrences)")
                    
                    # Suggest auto-recovery if possible
                    if self.config.get("auto_recovery_enabled", False):
                        await self._attempt_auto_recovery(error_info)
            
        except Exception as e:
            logger.error(f"Error pattern analysis failed: {e}")
    
    async def _attempt_auto_recovery(self, error_info: ErrorInfo):
        """Attempt automatic recovery from common errors"""
        try:
            recovery_strategy = self._get_recovery_strategy(error_info.error_type)
            
            if recovery_strategy:
                logger.info(f"Attempting auto-recovery for {error_info.error_type}")
                await recovery_strategy()
                
        except Exception as e:
            logger.error(f"Auto-recovery failed: {e}")
    
    def _get_recovery_strategy(self, error_type: str) -> Optional[Callable]:
        """Get recovery strategy for error type"""
        recovery_strategies = {
            "connection_error": self._recover_connection_error,
            "file_system_error": self._recover_file_system_error,
            "memory_error": self._recover_memory_error
        }
        
        return recovery_strategies.get(error_type)
    
    async def _recover_connection_error(self):
        """Recover from connection errors"""
        # Implement connection recovery logic
        logger.info("Attempting connection recovery")
        await asyncio.sleep(1)  # Placeholder
    
    async def _recover_file_system_error(self):
        """Recover from file system errors"""
        # Implement file system recovery logic
        logger.info("Attempting file system recovery")
        await asyncio.sleep(1)  # Placeholder
    
    async def _recover_memory_error(self):
        """Recover from memory errors"""
        # Implement memory recovery logic
        logger.info("Attempting memory recovery - clearing caches")
        
        # Clear metrics history
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-50:]
        
        # Clear error tracking for old errors
        cutoff_time = datetime.now(timezone.utc).timestamp() - 3600  # 1 hour ago
        self.error_tracking = {
            k: v for k, v in self.error_tracking.items()
            if v.last_seen.timestamp() > cutoff_time
        }
    
    async def _record_error(self, error_type: str, error: Exception, 
                          context: Dict[str, Any] = None):
        """Record error for tracking and analysis"""
        try:
            error_key = f"{error_type}:{str(error)[:100]}"
            current_time = datetime.now(timezone.utc)
            
            if error_key in self.error_tracking:
                error_info = self.error_tracking[error_key]
                error_info.count += 1
                error_info.last_seen = current_time
            else:
                error_info = ErrorInfo(
                    timestamp=current_time,
                    error_type=error_type,
                    error_message=str(error),
                    stack_trace=traceback.format_exc(),
                    context=context or {}
                )
                self.error_tracking[error_key] = error_info
            
            # Update error counts
            self.error_counts["total"] += 1
            self.error_counts["by_type"][error_type] = (
                self.error_counts["by_type"].get(error_type, 0) + 1
            )
            
        except Exception as e:
            logger.error(f"Error recording failed: {e}")
    
    async def _handle_critical_error(self, component: str, error: Exception):
        """Handle critical system errors"""
        logger.critical(f"Critical error in {component}: {error}")
        
        # Record the critical error
        await self._record_error(f"critical_{component}", error)
        
        # Set system to unhealthy state
        self.state = SystemState.UNHEALTHY
        
        # Attempt recovery if enabled
        if self.config.get("auto_recovery_enabled", False):
            logger.info("Attempting system recovery from critical error")
            await asyncio.sleep(5)  # Give system time to stabilize
            
            # Try to restore healthy state
            self.state = SystemState.RECOVERING
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "state": self.state.value,
            "uptime": time.time() - self.start_time,
            "health_checks": {
                name: {
                    "enabled": check.enabled,
                    "consecutive_failures": check.consecutive_failures,
                    "consecutive_successes": check.consecutive_successes,
                    "last_check": check.last_check
                }
                for name, check in self.health_checks.items()
            },
            "task_queue_size": self.task_queue.qsize(),
            "error_counts": self.error_counts.copy(),
            "metrics_count": len(self.metrics_history),
            "circuit_breakers": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count
                }
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 metrics
        
        return {
            "total_metrics": len(self.metrics_history),
            "recent_avg_cpu": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "recent_avg_memory": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "recent_avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            "recent_avg_response_time": sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics),
            "uptime": time.time() - self.start_time
        }

async def main():
    """Main execution function for robust system demonstration"""
    print("🛡️ Starting Robust Autonomous System - Generation 2")
    print("Enhanced with reliability, monitoring, and error handling")
    print("-" * 60)
    
    # Initialize robust system
    robust_system = RobustAutonomousSystem()
    
    try:
        # Add some demo tasks to process
        demo_tasks = [
            {"id": "task_1", "type": "discovery", "data": "sample_data"},
            {"id": "task_2", "type": "analysis", "data": "sample_analysis"},
            {"id": "task_3", "type": "processing", "data": "sample_processing"}
        ]
        
        for task in demo_tasks:
            robust_system.task_queue.put(task)
        
        print(f"Added {len(demo_tasks)} demo tasks to queue")
        
        # Start system (this will run until interrupted)
        await robust_system.start_system()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        print("\nSystem Status at Shutdown:")
        status = robust_system.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        metrics = robust_system.get_metrics_summary()
        print(f"\nMetrics Summary:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())