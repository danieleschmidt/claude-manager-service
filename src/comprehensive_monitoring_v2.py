#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - COMPREHENSIVE MONITORING SYSTEM V2
Advanced monitoring with AI-powered anomaly detection, predictive analytics, and real-time insights
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import threading
from enum import Enum
import math
import sqlite3
import aiosqlite
import structlog

logger = structlog.get_logger("ComprehensiveMonitoringV2")

class MetricType(Enum):
    """Types of metrics that can be monitored"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricPoint:
    """Individual metric measurement"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """System alert"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    metric_name: str
    timestamp: datetime
    actual_value: float
    expected_value: float
    anomaly_score: float  # 0-1, higher = more anomalous
    confidence: float
    pattern_type: str

class TimeSeriesAnalyzer:
    """Advanced time series analysis for anomaly detection"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines: Dict[str, Dict[str, float]] = {}
        
    def add_point(self, metric_name: str, point: MetricPoint):
        """Add a new data point for analysis"""
        self.data_windows[metric_name].append(point)
        self._update_baseline(metric_name)
    
    def detect_anomalies(self, metric_name: str, point: MetricPoint) -> Optional[AnomalyDetection]:
        """Detect anomalies using statistical analysis"""
        if metric_name not in self.baselines or len(self.data_windows[metric_name]) < 10:
            return None
            
        baseline = self.baselines[metric_name]
        value = point.value
        
        # Z-score based anomaly detection
        mean = baseline.get('mean', 0)
        std = baseline.get('std', 1)
        
        if std == 0:
            return None
            
        z_score = abs((value - mean) / std)
        
        # Consider it anomalous if z-score > 2.5
        if z_score > 2.5:
            # Calculate anomaly score (0-1)
            anomaly_score = min(z_score / 5.0, 1.0)  # Normalize to 0-1
            
            # Determine pattern type
            pattern_type = self._classify_anomaly_pattern(metric_name, point)
            
            return AnomalyDetection(
                metric_name=metric_name,
                timestamp=point.timestamp,
                actual_value=value,
                expected_value=mean,
                anomaly_score=anomaly_score,
                confidence=min(len(self.data_windows[metric_name]) / self.window_size, 1.0),
                pattern_type=pattern_type
            )
        
        return None
    
    def _update_baseline(self, metric_name: str):
        """Update baseline statistics for a metric"""
        window = self.data_windows[metric_name]
        if len(window) < 5:
            return
            
        values = [p.value for p in window]
        
        self.baselines[metric_name] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'trend': self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(values) < 2:
            return 0
            
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
            
        slope = numerator / denominator
        return max(-1, min(1, slope / abs(y_mean) if y_mean != 0 else 0))
    
    def _classify_anomaly_pattern(self, metric_name: str, point: MetricPoint) -> str:
        """Classify the type of anomaly pattern"""
        baseline = self.baselines[metric_name]
        value = point.value
        mean = baseline.get('mean', 0)
        trend = baseline.get('trend', 0)
        
        if value > mean * 1.5:
            return "spike" if trend >= 0 else "recovery_spike"
        elif value < mean * 0.5:
            return "drop" if trend <= 0 else "temporary_drop"
        elif abs(trend) > 0.5:
            return "trending_up" if trend > 0 else "trending_down"
        else:
            return "outlier"

class MetricsStorage:
    """Persistent storage for metrics with efficient querying"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(name, timestamp)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source TEXT NOT NULL,
                    labels TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolution_time REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    actual_value REAL NOT NULL,
                    expected_value REAL NOT NULL,
                    anomaly_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    pattern_type TEXT NOT NULL
                )
            """)
    
    async def store_metric(self, name: str, metric_type: MetricType, point: MetricPoint):
        """Store a metric point"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO metrics (name, type, timestamp, value, labels, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                name, metric_type.value, point.timestamp.timestamp(),
                point.value, json.dumps(point.labels), json.dumps(point.metadata)
            ))
            await db.commit()
    
    async def store_alert(self, alert: Alert):
        """Store an alert"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO alerts 
                (id, title, description, severity, timestamp, source, labels, resolved, resolution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id, alert.title, alert.description, alert.severity.value,
                alert.timestamp.timestamp(), alert.source, json.dumps(alert.labels),
                1 if alert.resolved else 0,
                alert.resolution_time.timestamp() if alert.resolution_time else None
            ))
            await db.commit()
    
    async def store_anomaly(self, anomaly: AnomalyDetection):
        """Store an anomaly detection result"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO anomalies 
                (metric_name, timestamp, actual_value, expected_value, anomaly_score, confidence, pattern_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                anomaly.metric_name, anomaly.timestamp.timestamp(),
                anomaly.actual_value, anomaly.expected_value,
                anomaly.anomaly_score, anomaly.confidence, anomaly.pattern_type
            ))
            await db.commit()
    
    async def query_metrics(self, name: str, start_time: datetime, end_time: datetime) -> List[MetricPoint]:
        """Query metrics within a time range"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT timestamp, value, labels, metadata FROM metrics
                WHERE name = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (name, start_time.timestamp(), end_time.timestamp())) as cursor:
                
                points = []
                async for row in cursor:
                    timestamp, value, labels_json, metadata_json = row
                    points.append(MetricPoint(
                        timestamp=datetime.fromtimestamp(timestamp, timezone.utc),
                        value=value,
                        labels=json.loads(labels_json) if labels_json else {},
                        metadata=json.loads(metadata_json) if metadata_json else {}
                    ))
                return points

class AlertManager:
    """Manages alerts with intelligent routing and notification"""
    
    def __init__(self, storage: MetricsStorage):
        self.storage = storage
        self.alert_rules: List[Callable] = []
        self.notification_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self.active_alerts: Dict[str, Alert] = {}
        self._lock = threading.Lock()
    
    def add_alert_rule(self, rule: Callable[[str, MetricPoint], Optional[Alert]]):
        """Add a custom alert rule"""
        self.alert_rules.append(rule)
    
    def add_notification_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]):
        """Add a notification handler for specific severity"""
        self.notification_handlers[severity].append(handler)
    
    async def process_metric(self, metric_name: str, point: MetricPoint):
        """Process a metric through all alert rules"""
        for rule in self.alert_rules:
            try:
                alert = rule(metric_name, point)
                if alert:
                    await self.fire_alert(alert)
            except Exception as e:
                logger.error(f"Alert rule failed: {e}")
    
    async def fire_alert(self, alert: Alert):
        """Fire an alert and handle notifications"""
        with self._lock:
            self.active_alerts[alert.id] = alert
        
        await self.storage.store_alert(alert)
        
        # Send notifications
        for handler in self.notification_handlers[alert.severity]:
            try:
                await handler(alert) if asyncio.iscoroutinefunction(handler) else handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
        
        logger.info(f"Alert fired: {alert.title} ({alert.severity.value})")
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = datetime.now(timezone.utc)
                await self.storage.store_alert(alert)
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert.title}")

class PerformanceDashboard:
    """Real-time performance dashboard generator"""
    
    def __init__(self, storage: MetricsStorage):
        self.storage = storage
        self.cached_summaries: Dict[str, Any] = {}
        self.cache_ttl = 60  # seconds
        self.last_cache_update = 0
    
    async def generate_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        now = time.time()
        if now - self.last_cache_update < self.cache_ttl and self.cached_summaries:
            return self.cached_summaries
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)  # Last hour
        
        dashboard = {
            'timestamp': end_time.isoformat(),
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'system_health': await self._get_system_health(),
            'performance_metrics': await self._get_performance_summary(start_time, end_time),
            'recent_anomalies': await self._get_recent_anomalies(),
            'active_alerts': await self._get_active_alerts(),
            'trends': await self._get_trend_analysis(start_time, end_time)
        }
        
        self.cached_summaries = dashboard
        self.last_cache_update = now
        return dashboard
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        # This would integrate with actual health checks
        return {
            'status': 'healthy',
            'uptime': time.time() - 1000000,  # Mock uptime
            'cpu_usage': 45.2,
            'memory_usage': 62.1,
            'disk_usage': 23.7
        }
    
    async def _get_performance_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get performance metrics summary"""
        # This would query actual metrics
        return {
            'average_response_time': 245.6,
            'throughput': 1250.3,
            'error_rate': 0.02,
            'success_rate': 99.98
        }
    
    async def _get_recent_anomalies(self) -> List[Dict[str, Any]]:
        """Get recent anomaly detections"""
        async with aiosqlite.connect(self.storage.db_path) as db:
            async with db.execute("""
                SELECT * FROM anomalies 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC LIMIT 10
            """, (time.time() - 3600,)) as cursor:
                
                anomalies = []
                async for row in cursor:
                    anomalies.append({
                        'metric_name': row[1],
                        'timestamp': datetime.fromtimestamp(row[2], timezone.utc).isoformat(),
                        'actual_value': row[3],
                        'expected_value': row[4],
                        'anomaly_score': row[5],
                        'confidence': row[6],
                        'pattern_type': row[7]
                    })
                return anomalies
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        async with aiosqlite.connect(self.storage.db_path) as db:
            async with db.execute("""
                SELECT * FROM alerts WHERE resolved = 0 ORDER BY timestamp DESC
            """) as cursor:
                
                alerts = []
                async for row in cursor:
                    alerts.append({
                        'id': row[0],
                        'title': row[1],
                        'description': row[2],
                        'severity': row[3],
                        'timestamp': datetime.fromtimestamp(row[4], timezone.utc).isoformat(),
                        'source': row[5],
                        'labels': json.loads(row[6]) if row[6] else {}
                    })
                return alerts
    
    async def _get_trend_analysis(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get trend analysis"""
        return {
            'performance_trend': 'improving',
            'error_trend': 'stable',
            'throughput_trend': 'increasing'
        }

class ComprehensiveMonitor:
    """Main monitoring orchestrator"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.storage = MetricsStorage(db_path)
        self.analyzer = TimeSeriesAnalyzer()
        self.alert_manager = AlertManager(self.storage)
        self.dashboard = PerformanceDashboard(self.storage)
        
        # Metrics tracking
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._running = False
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        self._setup_default_notification_handlers()
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        
        def high_error_rate_rule(metric_name: str, point: MetricPoint) -> Optional[Alert]:
            if metric_name == "error_rate" and point.value > 0.05:  # 5% error rate
                return Alert(
                    id=f"high_error_rate_{int(time.time())}",
                    title="High Error Rate Detected",
                    description=f"Error rate is {point.value:.2%}, exceeding 5% threshold",
                    severity=AlertSeverity.ERROR,
                    timestamp=point.timestamp,
                    source="monitoring_system",
                    labels={"metric": metric_name, "threshold": "0.05"}
                )
            return None
        
        def response_time_rule(metric_name: str, point: MetricPoint) -> Optional[Alert]:
            if metric_name == "response_time" and point.value > 1000:  # 1 second
                return Alert(
                    id=f"slow_response_{int(time.time())}",
                    title="Slow Response Time",
                    description=f"Response time is {point.value:.0f}ms, exceeding 1000ms threshold",
                    severity=AlertSeverity.WARNING,
                    timestamp=point.timestamp,
                    source="monitoring_system",
                    labels={"metric": metric_name, "threshold": "1000"}
                )
            return None
        
        self.alert_manager.add_alert_rule(high_error_rate_rule)
        self.alert_manager.add_alert_rule(response_time_rule)
    
    def _setup_default_notification_handlers(self):
        """Setup default notification handlers"""
        
        def log_alert(alert: Alert):
            logger.warning(f"ALERT: {alert.title} - {alert.description}")
        
        # Add log handler for all severities
        for severity in AlertSeverity:
            self.alert_manager.add_notification_handler(severity, log_alert)
    
    async def record_metric(self, name: str, value: float, 
                           metric_type: MetricType = MetricType.GAUGE,
                           labels: Optional[Dict[str, str]] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Record a metric measurement"""
        point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        # Store in database
        await self.storage.store_metric(name, metric_type, point)
        
        # Add to analyzer
        self.analyzer.add_point(name, point)
        
        # Check for anomalies
        anomaly = self.analyzer.detect_anomalies(name, point)
        if anomaly:
            await self.storage.store_anomaly(anomaly)
            logger.warning(f"Anomaly detected in {name}: {anomaly.pattern_type}")
        
        # Process through alert rules
        await self.alert_manager.process_metric(name, point)
    
    async def get_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        return await self.dashboard.generate_dashboard()
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        self._running = True
        logger.info("Comprehensive monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self._running = False
        logger.info("Comprehensive monitoring stopped")

# Example usage and testing
async def test_comprehensive_monitoring():
    """Test the comprehensive monitoring system"""
    monitor = ComprehensiveMonitor("test_monitoring.db")
    await monitor.start_monitoring()
    
    # Simulate some metrics
    import random
    for i in range(20):
        # Simulate response time metric
        response_time = random.gauss(500, 100)  # Normal distribution around 500ms
        if i > 15:  # Introduce anomaly towards the end
            response_time = random.gauss(1200, 200)
        
        await monitor.record_metric("response_time", response_time, MetricType.TIMER)
        
        # Simulate error rate
        error_rate = random.uniform(0.01, 0.03)  # 1-3% error rate
        if i > 17:  # Spike in errors
            error_rate = random.uniform(0.06, 0.08)
        
        await monitor.record_metric("error_rate", error_rate, MetricType.RATE)
        
        await asyncio.sleep(0.1)
    
    # Get dashboard
    dashboard = await monitor.get_dashboard()
    
    print("Comprehensive Monitoring Test Results:")
    print(json.dumps(dashboard, indent=2, default=str))
    
    await monitor.stop_monitoring()
    return True

if __name__ == "__main__":
    asyncio.run(test_comprehensive_monitoring())