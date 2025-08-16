#!/usr/bin/env python3
"""
Advanced Monitoring and Alerting System

Provides comprehensive monitoring, alerting, and observability:
- Real-time system health monitoring
- Performance metrics collection
- Intelligent alerting with ML-based anomaly detection
- Distributed tracing and observability
- SLA/SLO monitoring
- Business metrics tracking
"""

import asyncio
import json
import time
import statistics
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
import queue
import psutil
import sqlite3

# ML imports for anomaly detection
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

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
    ERROR = "error"
    CRITICAL = "critical"

class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class Metric:
    """Metric data point"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    metric_name: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable[[], bool]
    interval: int = 60  # seconds
    timeout: int = 10   # seconds
    failure_threshold: int = 3
    recovery_threshold: int = 2

@dataclass
class SLO:
    """Service Level Objective"""
    name: str
    target: float  # percentage (e.g., 99.9)
    time_window: int  # seconds
    metric_name: str
    condition: str  # e.g., "latency_p95 < 100"

class MetricsCollector:
    """Collects and stores metrics"""
    
    def __init__(self, storage_backend: str = "memory"):
        self.storage_backend = storage_backend
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.db_connection = None
        self.logger = logging.getLogger("metrics_collector")
        
        if storage_backend == "sqlite":
            self._init_sqlite_storage()
    
    def _init_sqlite_storage(self):
        """Initialize SQLite storage for metrics"""
        try:
            self.db_connection = sqlite3.connect("metrics.db", check_same_thread=False)
            cursor = self.db_connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    labels TEXT,
                    metric_type TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            """)
            self.db_connection.commit()
            self.logger.info("SQLite metrics storage initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite storage: {e}")
    
    def record_metric(self, metric: Metric):
        """Record a metric"""
        if self.storage_backend == "sqlite" and self.db_connection:
            self._store_metric_sqlite(metric)
        else:
            self._store_metric_memory(metric)
    
    def _store_metric_memory(self, metric: Metric):
        """Store metric in memory"""
        self.metrics[metric.name].append(metric)
    
    def _store_metric_sqlite(self, metric: Metric):
        """Store metric in SQLite"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO metrics (name, value, timestamp, labels, metric_type)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metric.name,
                metric.value,
                metric.timestamp.isoformat(),
                json.dumps(metric.labels),
                metric.metric_type.value
            ))
            self.db_connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to store metric in SQLite: {e}")
    
    def get_metrics(self, name: str, start_time: datetime = None, 
                   end_time: datetime = None) -> List[Metric]:
        """Retrieve metrics by name and time range"""
        if self.storage_backend == "sqlite" and self.db_connection:
            return self._get_metrics_sqlite(name, start_time, end_time)
        else:
            return self._get_metrics_memory(name, start_time, end_time)
    
    def _get_metrics_memory(self, name: str, start_time: datetime = None,
                           end_time: datetime = None) -> List[Metric]:
        """Get metrics from memory storage"""
        metrics = list(self.metrics.get(name, []))
        
        if start_time or end_time:
            filtered = []
            for metric in metrics:
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                filtered.append(metric)
            return filtered
        
        return metrics
    
    def _get_metrics_sqlite(self, name: str, start_time: datetime = None,
                           end_time: datetime = None) -> List[Metric]:
        """Get metrics from SQLite storage"""
        try:
            cursor = self.db_connection.cursor()
            query = "SELECT name, value, timestamp, labels, metric_type FROM metrics WHERE name = ?"
            params = [name]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                metric = Metric(
                    name=row[0],
                    value=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    labels=json.loads(row[3]) if row[3] else {},
                    metric_type=MetricType(row[4])
                )
                metrics.append(metric)
            
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to retrieve metrics from SQLite: {e}")
            return []
    
    def get_metric_summary(self, name: str, duration_minutes: int = 60) -> Dict[str, float]:
        """Get statistical summary of metrics"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=duration_minutes)
        
        metrics = self.get_metrics(name, start_time, end_time)
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        summary = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'latest': values[-1] if values else 0
        }
        
        if len(values) > 1:
            summary['std_dev'] = statistics.stdev(values)
            
            # Calculate percentiles
            sorted_values = sorted(values)
            n = len(sorted_values)
            summary['p50'] = sorted_values[int(n * 0.5)]
            summary['p90'] = sorted_values[int(n * 0.9)]
            summary['p95'] = sorted_values[int(n * 0.95)]
            summary['p99'] = sorted_values[int(n * 0.99)]
        
        return summary

class AnomalyDetector:
    """ML-based anomaly detection for metrics"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.logger = logging.getLogger("anomaly_detector")
    
    def learn_baseline(self, metric_name: str, values: List[float]):
        """Learn baseline statistics for a metric"""
        if not values:
            return
        
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        self.baselines[metric_name] = {
            'mean': mean,
            'std_dev': std_dev,
            'min': min(values),
            'max': max(values),
            'sample_size': len(values)
        }
        
        self.logger.info(f"Learned baseline for {metric_name}: mean={mean:.2f}, std={std_dev:.2f}")
    
    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """
        Detect if a metric value is anomalous
        
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if metric_name not in self.baselines:
            return False, 0.0
        
        baseline = self.baselines[metric_name]
        mean = baseline['mean']
        std_dev = baseline['std_dev']
        
        if std_dev == 0:
            # No variation in baseline, consider exact matches normal
            return value != mean, abs(value - mean)
        
        # Calculate z-score
        z_score = abs(value - mean) / std_dev
        is_anomaly = z_score > self.sensitivity
        
        return is_anomaly, z_score
    
    def update_baseline(self, metric_name: str, new_values: List[float]):
        """Update baseline with new data using exponential moving average"""
        if metric_name not in self.baselines:
            self.learn_baseline(metric_name, new_values)
            return
        
        if not new_values:
            return
        
        # Use exponential moving average to update baseline
        alpha = 0.1  # Learning rate
        current = self.baselines[metric_name]
        
        new_mean = statistics.mean(new_values)
        new_std = statistics.stdev(new_values) if len(new_values) > 1 else current['std_dev']
        
        updated_mean = (1 - alpha) * current['mean'] + alpha * new_mean
        updated_std = (1 - alpha) * current['std_dev'] + alpha * new_std
        
        self.baselines[metric_name].update({
            'mean': updated_mean,
            'std_dev': updated_std,
            'sample_size': current['sample_size'] + len(new_values)
        })

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        self.logger = logging.getLogger("alert_manager")
        
        # Start alert evaluation loop
        self._start_alert_loop()
    
    def add_alert_rule(self, alert: Alert):
        """Add an alert rule"""
        self.alert_rules[alert.id] = alert
        self.logger.info(f"Added alert rule: {alert.name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
    
    def _start_alert_loop(self):
        """Start the alert evaluation loop in a background thread"""
        def alert_loop():
            while True:
                try:
                    self._evaluate_alerts()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Error in alert loop: {e}")
        
        alert_thread = threading.Thread(target=alert_loop, daemon=True)
        alert_thread.start()
        self.logger.info("Alert evaluation loop started")
    
    def _evaluate_alerts(self):
        """Evaluate all alert rules"""
        for alert_id, alert_rule in self.alert_rules.items():
            try:
                # Get recent metrics
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(minutes=5)
                
                metrics = self.metrics_collector.get_metrics(
                    alert_rule.metric_name, start_time, end_time
                )
                
                if not metrics:
                    continue
                
                # Evaluate condition
                should_alert = self._evaluate_condition(alert_rule, metrics)
                
                if should_alert and alert_id not in self.active_alerts:
                    # Trigger new alert
                    self._trigger_alert(alert_rule)
                elif not should_alert and alert_id in self.active_alerts:
                    # Resolve existing alert
                    self._resolve_alert(alert_id)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating alert {alert_id}: {e}")
    
    def _evaluate_condition(self, alert: Alert, metrics: List[Metric]) -> bool:
        """Evaluate alert condition"""
        if not metrics:
            return False
        
        latest_value = metrics[-1].value
        
        # Simple threshold-based conditions
        if alert.condition == "greater_than":
            return latest_value > alert.threshold
        elif alert.condition == "less_than":
            return latest_value < alert.threshold
        elif alert.condition == "equals":
            return latest_value == alert.threshold
        elif alert.condition == "not_equals":
            return latest_value != alert.threshold
        elif alert.condition == "average_greater_than":
            avg = sum(m.value for m in metrics) / len(metrics)
            return avg > alert.threshold
        elif alert.condition == "average_less_than":
            avg = sum(m.value for m in metrics) / len(metrics)
            return avg < alert.threshold
        
        return False
    
    def _trigger_alert(self, alert_rule: Alert):
        """Trigger a new alert"""
        alert = Alert(
            id=alert_rule.id,
            name=alert_rule.name,
            description=alert_rule.description,
            severity=alert_rule.severity,
            condition=alert_rule.condition,
            threshold=alert_rule.threshold,
            metric_name=alert_rule.metric_name,
            timestamp=datetime.now(timezone.utc),
            metadata=alert_rule.metadata.copy()
        )
        
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        self.logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Notification handler failed: {e}")
    
    def _resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert resolved: {alert.name}")
            
            # Notify of resolution
            for handler in self.notification_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Notification handler failed: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]

class HealthMonitor:
    """System health monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        self.logger = logging.getLogger("health_monitor")
        
        # Add default system health checks
        self._add_default_health_checks()
        
        # Start health check loop
        self._start_health_loop()
    
    def _add_default_health_checks(self):
        """Add default system health checks"""
        self.add_health_check(HealthCheck(
            name="cpu_usage",
            check_function=lambda: psutil.cpu_percent(interval=1) < 90,
            interval=60
        ))
        
        self.add_health_check(HealthCheck(
            name="memory_usage",
            check_function=lambda: psutil.virtual_memory().percent < 90,
            interval=60
        ))
        
        self.add_health_check(HealthCheck(
            name="disk_usage",
            check_function=lambda: psutil.disk_usage('/').percent < 90,
            interval=300  # 5 minutes
        ))
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check"""
        self.health_checks[health_check.name] = health_check
        self.health_status[health_check.name] = HealthStatus.HEALTHY
        self.logger.info(f"Added health check: {health_check.name}")
    
    def _start_health_loop(self):
        """Start the health check loop"""
        def health_loop():
            while True:
                try:
                    self._run_health_checks()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Error in health loop: {e}")
        
        health_thread = threading.Thread(target=health_loop, daemon=True)
        health_thread.start()
        self.logger.info("Health check loop started")
    
    def _run_health_checks(self):
        """Run all health checks"""
        for name, health_check in self.health_checks.items():
            try:
                # Check if it's time to run this health check
                now = time.time()
                last_check = getattr(health_check, 'last_check', 0)
                
                if now - last_check < health_check.interval:
                    continue
                
                # Run the health check
                start_time = time.time()
                try:
                    is_healthy = health_check.check_function()
                    duration = time.time() - start_time
                    
                    # Record metrics
                    self.metrics_collector.record_metric(Metric(
                        name=f"health_check_{name}",
                        value=1 if is_healthy else 0,
                        timestamp=datetime.now(timezone.utc),
                        labels={"check_name": name}
                    ))
                    
                    self.metrics_collector.record_metric(Metric(
                        name=f"health_check_duration_{name}",
                        value=duration,
                        timestamp=datetime.now(timezone.utc),
                        labels={"check_name": name},
                        metric_type=MetricType.TIMER
                    ))
                    
                    # Update health status
                    if is_healthy:
                        self.health_status[name] = HealthStatus.HEALTHY
                    else:
                        self.health_status[name] = HealthStatus.UNHEALTHY
                    
                    health_check.last_check = now
                    
                except Exception as e:
                    self.logger.error(f"Health check {name} failed: {e}")
                    self.health_status[name] = HealthStatus.UNHEALTHY
                    
            except Exception as e:
                self.logger.error(f"Error running health check {name}: {e}")
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health"""
        if not self.health_status:
            return HealthStatus.HEALTHY
        
        unhealthy_count = sum(1 for status in self.health_status.values() 
                             if status == HealthStatus.UNHEALTHY)
        total_count = len(self.health_status)
        
        if unhealthy_count == 0:
            return HealthStatus.HEALTHY
        elif unhealthy_count < total_count * 0.5:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        return {
            'overall_health': self.get_overall_health().value,
            'individual_checks': {
                name: status.value for name, status in self.health_status.items()
            },
            'total_checks': len(self.health_checks),
            'healthy_checks': sum(1 for status in self.health_status.values() 
                                if status == HealthStatus.HEALTHY),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

class AdvancedMonitoringSystem:
    """Main monitoring system that orchestrates all components"""
    
    def __init__(self, storage_backend: str = "memory"):
        self.metrics_collector = MetricsCollector(storage_backend)
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_monitor = HealthMonitor(self.metrics_collector)
        self.logger = logging.getLogger("advanced_monitoring")
        
        # Start system metrics collection
        self._start_system_metrics_collection()
        
        # Setup default alerts
        self._setup_default_alerts()
    
    def _start_system_metrics_collection(self):
        """Start collecting system metrics"""
        def collect_system_metrics():
            while True:
                try:
                    now = datetime.now(timezone.utc)
                    
                    # CPU metrics
                    self.metrics_collector.record_metric(Metric(
                        name="system_cpu_percent",
                        value=psutil.cpu_percent(interval=1),
                        timestamp=now,
                        metric_type=MetricType.GAUGE
                    ))
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.metrics_collector.record_metric(Metric(
                        name="system_memory_percent",
                        value=memory.percent,
                        timestamp=now,
                        metric_type=MetricType.GAUGE
                    ))
                    
                    self.metrics_collector.record_metric(Metric(
                        name="system_memory_available",
                        value=memory.available,
                        timestamp=now,
                        metric_type=MetricType.GAUGE
                    ))
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.metrics_collector.record_metric(Metric(
                        name="system_disk_percent",
                        value=disk.percent,
                        timestamp=now,
                        metric_type=MetricType.GAUGE
                    ))
                    
                    # Network metrics
                    network = psutil.net_io_counters()
                    self.metrics_collector.record_metric(Metric(
                        name="system_network_bytes_sent",
                        value=network.bytes_sent,
                        timestamp=now,
                        metric_type=MetricType.COUNTER
                    ))
                    
                    self.metrics_collector.record_metric(Metric(
                        name="system_network_bytes_recv",
                        value=network.bytes_recv,
                        timestamp=now,
                        metric_type=MetricType.COUNTER
                    ))
                    
                    time.sleep(60)  # Collect every minute
                    
                except Exception as e:
                    self.logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(60)
        
        metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        metrics_thread.start()
        self.logger.info("System metrics collection started")
    
    def _setup_default_alerts(self):
        """Setup default system alerts"""
        # High CPU usage alert
        self.alert_manager.add_alert_rule(Alert(
            id="high_cpu_usage",
            name="High CPU Usage",
            description="CPU usage is above 90%",
            severity=AlertSeverity.WARNING,
            condition="greater_than",
            threshold=90.0,
            metric_name="system_cpu_percent",
            timestamp=datetime.now(timezone.utc)
        ))
        
        # High memory usage alert
        self.alert_manager.add_alert_rule(Alert(
            id="high_memory_usage",
            name="High Memory Usage",
            description="Memory usage is above 90%",
            severity=AlertSeverity.WARNING,
            condition="greater_than",
            threshold=90.0,
            metric_name="system_memory_percent",
            timestamp=datetime.now(timezone.utc)
        ))
        
        # High disk usage alert
        self.alert_manager.add_alert_rule(Alert(
            id="high_disk_usage",
            name="High Disk Usage",
            description="Disk usage is above 90%",
            severity=AlertSeverity.CRITICAL,
            condition="greater_than",
            threshold=90.0,
            metric_name="system_disk_percent",
            timestamp=datetime.now(timezone.utc)
        ))
    
    def record_business_metric(self, name: str, value: Union[int, float], 
                              labels: Dict[str, str] = None):
        """Record a business metric"""
        self.metrics_collector.record_metric(Metric(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            metric_type=MetricType.COUNTER
        ))
    
    def record_performance_metric(self, name: str, duration: float, 
                                 labels: Dict[str, str] = None):
        """Record a performance metric"""
        self.metrics_collector.record_metric(Metric(
            name=name,
            value=duration,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            metric_type=MetricType.TIMER
        ))
    
    def add_custom_alert(self, alert: Alert):
        """Add a custom alert rule"""
        self.alert_manager.add_alert_rule(alert)
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler for alerts"""
        self.alert_manager.add_notification_handler(handler)
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get system dashboard data"""
        # Get recent metrics summaries
        cpu_summary = self.metrics_collector.get_metric_summary("system_cpu_percent", 60)
        memory_summary = self.metrics_collector.get_metric_summary("system_memory_percent", 60)
        disk_summary = self.metrics_collector.get_metric_summary("system_disk_percent", 60)
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'health': self.health_monitor.get_health_summary(),
            'active_alerts': [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
            'metrics': {
                'cpu': cpu_summary,
                'memory': memory_summary,
                'disk': disk_summary
            },
            'alert_history_24h': len(self.alert_manager.get_alert_history(24))
        }
    
    def get_metrics_for_timeframe(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics for a specific timeframe"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.metrics_collector.get_metrics(metric_name, start_time, end_time)
        return [
            {
                'timestamp': metric.timestamp.isoformat(),
                'value': metric.value,
                'labels': metric.labels
            }
            for metric in metrics
        ]

# Utility functions
def create_monitoring_system(storage: str = "memory") -> AdvancedMonitoringSystem:
    """Create a monitoring system with appropriate configuration"""
    return AdvancedMonitoringSystem(storage)

def console_notification_handler(alert: Alert):
    """Simple console notification handler"""
    status = "RESOLVED" if alert.resolved else "TRIGGERED"
    print(f"[{alert.timestamp.isoformat()}] ALERT {status}: {alert.name} - {alert.description}")

# Example usage
if __name__ == "__main__":
    # Create monitoring system
    monitoring = create_monitoring_system("sqlite")
    
    # Add console notification handler
    monitoring.add_notification_handler(console_notification_handler)
    
    # Record some test metrics
    monitoring.record_business_metric("api_requests", 100, {"endpoint": "/api/test"})
    monitoring.record_performance_metric("api_response_time", 0.123, {"endpoint": "/api/test"})
    
    # Get dashboard data
    dashboard = monitoring.get_system_dashboard()
    print(json.dumps(dashboard, indent=2, default=str))
    
    # Keep the monitoring system running for a while
    time.sleep(10)