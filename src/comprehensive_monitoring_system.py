#!/usr/bin/env python3
"""
Comprehensive Monitoring System - Generation 2
Advanced monitoring, alerting, and observability for autonomous systems
"""

import asyncio
import json
import logging
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path
import statistics
import collections

# Configure monitoring-specific logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"

@dataclass
class MetricValue:
    """Individual metric value"""
    timestamp: datetime
    value: Union[float, int, str]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metric:
    """Metric definition and storage"""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    values: List[MetricValue] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    max_values: int = 1000  # Maximum values to store

    def add_value(self, value: Union[float, int, str], tags: Dict[str, str] = None):
        """Add a value to the metric"""
        metric_value = MetricValue(
            timestamp=datetime.now(timezone.utc),
            value=value,
            tags=tags or {}
        )
        
        self.values.append(metric_value)
        
        # Trim old values if necessary
        if len(self.values) > self.max_values:
            self.values = self.values[-self.max_values:]

@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 100", "== 'error'"
    threshold: Union[float, int, str]
    severity: AlertSeverity
    description: str
    evaluation_window: int = 300  # seconds
    evaluation_interval: int = 60  # seconds
    cooldown_period: int = 300  # seconds to wait before re-alerting
    enabled: bool = True
    last_evaluation: float = 0
    last_alert_time: float = 0

@dataclass
class Alert:
    """Active alert"""
    id: str
    rule: AlertRule
    value: Union[float, int, str]
    severity: AlertSeverity
    status: AlertStatus
    message: str
    timestamp: datetime
    resolved_timestamp: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Collects and stores metrics from various sources"""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.collectors: List[Callable] = []
        self.lock = threading.Lock()
        
    def register_metric(self, name: str, metric_type: MetricType, 
                       description: str, unit: str = "", tags: Dict[str, str] = None):
        """Register a new metric"""
        with self.lock:
            self.metrics[name] = Metric(
                name=name,
                type=metric_type,
                description=description,
                unit=unit,
                tags=tags or {}
            )
    
    def record_metric(self, name: str, value: Union[float, int, str], 
                     tags: Dict[str, str] = None):
        """Record a metric value"""
        with self.lock:
            if name in self.metrics:
                self.metrics[name].add_value(value, tags)
            else:
                logger.warning(f"Metric '{name}' not registered")
    
    def increment_counter(self, name: str, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        with self.lock:
            if name in self.metrics and self.metrics[name].type == MetricType.COUNTER:
                current_value = 0
                if self.metrics[name].values:
                    last_value = self.metrics[name].values[-1].value
                    if isinstance(last_value, (int, float)):
                        current_value = last_value
                
                self.metrics[name].add_value(current_value + 1, tags)
            else:
                logger.warning(f"Counter metric '{name}' not found")
    
    def set_gauge(self, name: str, value: Union[float, int], tags: Dict[str, str] = None):
        """Set a gauge metric value"""
        self.record_metric(name, value, tags)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer metric"""
        self.record_metric(name, duration, tags)
    
    def get_metric_values(self, name: str, since: Optional[datetime] = None) -> List[MetricValue]:
        """Get metric values, optionally filtered by time"""
        with self.lock:
            if name not in self.metrics:
                return []
            
            values = self.metrics[name].values
            
            if since:
                values = [v for v in values if v.timestamp >= since]
            
            return values
    
    def get_latest_value(self, name: str) -> Optional[MetricValue]:
        """Get the latest value for a metric"""
        values = self.get_metric_values(name)
        return values[-1] if values else None
    
    def register_collector(self, collector_func: Callable):
        """Register a metrics collector function"""
        self.collectors.append(collector_func)
    
    async def collect_all_metrics(self):
        """Run all registered collectors"""
        for collector in self.collectors:
            try:
                if asyncio.iscoroutinefunction(collector):
                    await collector()
                else:
                    collector()
            except Exception as e:
                logger.error(f"Metrics collector failed: {e}")

class AlertManager:
    """Manages alerts and alert rules"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        self.lock = threading.Lock()
        
    def register_alert_rule(self, rule: AlertRule):
        """Register an alert rule"""
        with self.lock:
            self.alert_rules[rule.name] = rule
    
    def register_notification_handler(self, handler: Callable):
        """Register a notification handler for alerts"""
        self.notification_handlers.append(handler)
    
    async def evaluate_all_rules(self):
        """Evaluate all alert rules"""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check if it's time to evaluate this rule
            if current_time - rule.last_evaluation < rule.evaluation_interval:
                continue
            
            await self._evaluate_rule(rule)
            rule.last_evaluation = current_time
    
    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a specific alert rule"""
        try:
            # Get metric values for evaluation window
            since = datetime.now(timezone.utc) - timedelta(seconds=rule.evaluation_window)
            values = self.metrics_collector.get_metric_values(rule.metric_name, since)
            
            if not values:
                return
            
            # Extract numeric values for evaluation
            numeric_values = []
            for value in values:
                if isinstance(value.value, (int, float)):
                    numeric_values.append(value.value)
            
            if not numeric_values:
                return
            
            # Evaluate condition
            current_value = numeric_values[-1]  # Latest value
            alert_triggered = self._evaluate_condition(rule.condition, rule.threshold, current_value)
            
            alert_id = f"{rule.name}_{rule.metric_name}"
            
            if alert_triggered:
                await self._trigger_alert(alert_id, rule, current_value)
            else:
                await self._resolve_alert(alert_id)
                
        except Exception as e:
            logger.error(f"Alert rule evaluation failed for {rule.name}: {e}")
    
    def _evaluate_condition(self, condition: str, threshold: Union[float, int, str], 
                           value: Union[float, int, str]) -> bool:
        """Evaluate alert condition"""
        try:
            if condition.startswith('>'):
                return float(value) > float(threshold)
            elif condition.startswith('<'):
                return float(value) < float(threshold)
            elif condition.startswith('>='):
                return float(value) >= float(threshold)
            elif condition.startswith('<='):
                return float(value) <= float(threshold)
            elif condition.startswith('=='):
                return str(value) == str(threshold)
            elif condition.startswith('!='):
                return str(value) != str(threshold)
            else:
                logger.warning(f"Unknown condition: {condition}")
                return False
        except ValueError:
            logger.error(f"Invalid condition evaluation: {condition} {threshold} {value}")
            return False
    
    async def _trigger_alert(self, alert_id: str, rule: AlertRule, value: Union[float, int, str]):
        """Trigger an alert"""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - rule.last_alert_time < rule.cooldown_period:
            return
        
        with self.lock:
            if alert_id not in self.active_alerts:
                alert = Alert(
                    id=alert_id,
                    rule=rule,
                    value=value,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    message=f"{rule.description} - Current value: {value}, Threshold: {rule.threshold}",
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                rule.last_alert_time = current_time
                
                logger.warning(f"Alert triggered: {alert.message}")
                
                # Send notifications
                await self._send_notifications(alert)
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_timestamp = datetime.now(timezone.utc)
                
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert.message}")
                
                # Send resolution notification
                await self._send_notifications(alert)
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                
                logger.info(f"Alert acknowledged by {acknowledged_by}: {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        with self.lock:
            active_count = len(self.active_alerts)
            severity_counts = {}
            
            for alert in self.active_alerts.values():
                severity = alert.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                "active_alerts": active_count,
                "severity_breakdown": severity_counts,
                "total_historical_alerts": len(self.alert_history)
            }

class PerformanceMonitor:
    """Monitors system and application performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._register_performance_metrics()
        
        # Performance tracking
        self.operation_times = collections.defaultdict(list)
        self.error_counts = collections.defaultdict(int)
        
    def _register_performance_metrics(self):
        """Register standard performance metrics"""
        metrics = [
            ("cpu_usage_percent", MetricType.GAUGE, "CPU usage percentage", "%"),
            ("memory_usage_percent", MetricType.GAUGE, "Memory usage percentage", "%"),
            ("disk_usage_percent", MetricType.GAUGE, "Disk usage percentage", "%"),
            ("operation_duration", MetricType.TIMER, "Operation duration", "seconds"),
            ("request_count", MetricType.COUNTER, "Request count", "requests"),
            ("error_count", MetricType.COUNTER, "Error count", "errors"),
            ("response_time", MetricType.HISTOGRAM, "Response time", "seconds"),
            ("throughput", MetricType.GAUGE, "Throughput", "ops/sec"),
            ("queue_size", MetricType.GAUGE, "Queue size", "items"),
            ("connection_count", MetricType.GAUGE, "Active connections", "connections")
        ]
        
        for name, metric_type, description, unit in metrics:
            self.metrics_collector.register_metric(name, metric_type, description, unit)
    
    def record_operation_time(self, operation: str, duration: float):
        """Record operation execution time"""
        self.operation_times[operation].append(duration)
        self.metrics_collector.record_timer("operation_duration", duration, {"operation": operation})
        
        # Keep only recent measurements
        if len(self.operation_times[operation]) > 100:
            self.operation_times[operation] = self.operation_times[operation][-100:]
    
    def record_error(self, error_type: str):
        """Record an error occurrence"""
        self.error_counts[error_type] += 1
        self.metrics_collector.increment_counter("error_count", {"error_type": error_type})
    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        times = self.operation_times.get(operation, [])
        
        if not times:
            return {"count": 0}
        
        return {
            "count": len(times),
            "avg": statistics.mean(times),
            "min": min(times),
            "max": max(times),
            "median": statistics.median(times),
            "p95": self._percentile(times, 95),
            "p99": self._percentile(times, 99)
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

class SystemMonitor:
    """Collects system-level metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        
    async def collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # Try to use psutil if available
            try:
                import psutil
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics_collector.set_gauge("cpu_usage_percent", cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.metrics_collector.set_gauge("memory_usage_percent", memory.percent)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.metrics_collector.set_gauge("disk_usage_percent", disk_percent)
                
                # Process count
                process_count = len(psutil.pids())
                self.metrics_collector.set_gauge("process_count", process_count)
                
            except ImportError:
                # Fallback metrics without psutil
                self.metrics_collector.set_gauge("cpu_usage_percent", 0)
                self.metrics_collector.set_gauge("memory_usage_percent", 0)
                self.metrics_collector.set_gauge("disk_usage_percent", 0)
                
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")

class ComprehensiveMonitoringSystem:
    """
    Main monitoring system that orchestrates all monitoring components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        self.system_monitor = SystemMonitor(self.metrics_collector)
        
        # State
        self.running = False
        self.start_time = time.time()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup notifications
        self._setup_notifications()
        
        logger.info("Comprehensive Monitoring System initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration"""
        return {
            "collection_interval": 30,
            "alert_evaluation_interval": 60,
            "metrics_retention_hours": 24,
            "enable_system_monitoring": True,
            "enable_performance_monitoring": True,
            "enable_alerting": True,
            "alert_thresholds": {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0,
                "error_rate": 5.0,
                "response_time": 5.0
            }
        }
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        if not self.config.get("enable_alerting", True):
            return
        
        thresholds = self.config.get("alert_thresholds", {})
        
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                condition=">",
                threshold=thresholds.get("cpu_usage", 80.0),
                severity=AlertSeverity.WARNING,
                description="High CPU usage detected"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_usage_percent",
                condition=">",
                threshold=thresholds.get("memory_usage", 85.0),
                severity=AlertSeverity.CRITICAL,
                description="High memory usage detected"
            ),
            AlertRule(
                name="high_disk_usage",
                metric_name="disk_usage_percent",
                condition=">",
                threshold=thresholds.get("disk_usage", 90.0),
                severity=AlertSeverity.CRITICAL,
                description="High disk usage detected"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_count",
                condition=">",
                threshold=thresholds.get("error_rate", 5.0),
                severity=AlertSeverity.WARNING,
                description="High error rate detected",
                evaluation_window=300
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.register_alert_rule(rule)
    
    def _setup_notifications(self):
        """Setup notification handlers"""
        self.alert_manager.register_notification_handler(self._log_notification)
    
    async def _log_notification(self, alert: Alert):
        """Log-based notification handler"""
        status_emoji = {
            AlertStatus.ACTIVE: "🚨",
            AlertStatus.RESOLVED: "✅",
            AlertStatus.ACKNOWLEDGED: "👍"
        }
        
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "❌",
            AlertSeverity.EMERGENCY: "🆘"
        }
        
        emoji = status_emoji.get(alert.status, "❓")
        severity_icon = severity_emoji.get(alert.severity, "❓")
        
        log_message = f"{emoji} {severity_icon} Alert {alert.status.value}: {alert.message}"
        
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            logger.critical(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        self.running = True
        logger.info("Starting Comprehensive Monitoring System")
        
        # Start monitoring tasks
        tasks = []
        
        if self.config.get("enable_system_monitoring", True):
            tasks.append(asyncio.create_task(self._system_monitoring_loop()))
        
        if self.config.get("enable_alerting", True):
            tasks.append(asyncio.create_task(self._alert_evaluation_loop()))
        
        tasks.append(asyncio.create_task(self._metrics_collection_loop()))
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Monitoring tasks cancelled")
        except Exception as e:
            logger.error(f"Monitoring system error: {e}")
        finally:
            self.running = False
    
    async def _system_monitoring_loop(self):
        """System monitoring loop"""
        while self.running:
            try:
                await self.system_monitor.collect_system_metrics()
                await asyncio.sleep(self.config.get("collection_interval", 30))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _alert_evaluation_loop(self):
        """Alert evaluation loop"""
        while self.running:
            try:
                await self.alert_manager.evaluate_all_rules()
                await asyncio.sleep(self.config.get("alert_evaluation_interval", 60))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collection_loop(self):
        """Metrics collection loop"""
        while self.running:
            try:
                await self.metrics_collector.collect_all_metrics()
                await asyncio.sleep(self.config.get("collection_interval", 30))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        logger.info("Stopping monitoring system")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "system_status": {
                "uptime": time.time() - self.start_time,
                "running": self.running
            },
            "metrics_summary": self._get_metrics_summary(),
            "alerts": {
                "active": self.alert_manager.get_active_alerts(),
                "summary": self.alert_manager.get_alert_summary()
            },
            "performance": self._get_performance_summary()
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {}
        
        key_metrics = ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent"]
        
        for metric_name in key_metrics:
            latest_value = self.metrics_collector.get_latest_value(metric_name)
            if latest_value:
                summary[metric_name] = latest_value.value
            else:
                summary[metric_name] = None
        
        return summary
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "operation_stats": {
                op: self.performance_monitor.get_operation_stats(op)
                for op in self.performance_monitor.operation_times.keys()
            },
            "error_counts": dict(self.performance_monitor.error_counts)
        }

# Example usage and testing
async def test_monitoring_system():
    """Test the monitoring system"""
    monitoring_system = ComprehensiveMonitoringSystem()
    
    print("🔍 Testing Comprehensive Monitoring System")
    print("-" * 50)
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(monitoring_system.start_monitoring())
    
    # Simulate some operations and errors
    await asyncio.sleep(2)
    
    # Record some performance metrics
    monitoring_system.performance_monitor.record_operation_time("test_operation", 0.15)
    monitoring_system.performance_monitor.record_operation_time("test_operation", 0.23)
    monitoring_system.performance_monitor.record_operation_time("slow_operation", 1.5)
    
    # Record some errors
    monitoring_system.performance_monitor.record_error("validation_error")
    monitoring_system.performance_monitor.record_error("network_error")
    
    # Wait a bit for metrics collection
    await asyncio.sleep(3)
    
    # Get dashboard data
    dashboard_data = monitoring_system.get_dashboard_data()
    
    print("Dashboard Data:")
    print(f"  System Uptime: {dashboard_data['system_status']['uptime']:.1f} seconds")
    print(f"  System Running: {dashboard_data['system_status']['running']}")
    
    print(f"  Active Alerts: {len(dashboard_data['alerts']['active'])}")
    
    performance = dashboard_data['performance']
    if performance['operation_stats']:
        print("  Performance Stats:")
        for op, stats in performance['operation_stats'].items():
            print(f"    {op}: avg={stats.get('avg', 0):.3f}s, count={stats['count']}")
    
    if performance['error_counts']:
        print("  Error Counts:")
        for error_type, count in performance['error_counts'].items():
            print(f"    {error_type}: {count}")
    
    # Stop monitoring
    monitoring_system.stop_monitoring()
    monitoring_task.cancel()
    
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    
    print("Monitoring system test completed")

if __name__ == "__main__":
    asyncio.run(test_monitoring_system())