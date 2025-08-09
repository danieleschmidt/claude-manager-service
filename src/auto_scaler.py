"""
Auto-Scaling and Load Balancing System for Claude Manager Service Generation 3

This module provides enterprise-grade auto-scaling capabilities including:
- Horizontal scaling triggers based on system metrics
- Load balancing across multiple service instances
- Predictive scaling using historical data
- Circuit breaker patterns for fault tolerance
- Health-based traffic routing
- Container orchestration integration
- Cost-aware scaling decisions
- Custom scaling policies and rules
"""

import asyncio
import json
import math
import time
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, NamedTuple
import logging
import os
import psutil

from .logger import get_logger
from .performance_monitor import monitor_performance, get_monitor
from .error_handler import ScalingError, with_enhanced_error_handling


logger = get_logger(__name__)


class ScalingDirection(Enum):
    """Scaling directions"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """Scaling trigger types"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_DEPTH = "queue_depth"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


class HealthStatus(Enum):
    """Instance health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricValue:
    """Represents a metric value with timestamp"""
    value: float
    timestamp: float
    source: Optional[str] = None


@dataclass
class ScalingRule:
    """Scaling rule configuration"""
    name: str
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    evaluation_periods: int
    cooldown_seconds: int
    scaling_increment: int = 1
    enabled: bool = True
    custom_evaluator: Optional[Callable[[List[float]], bool]] = None


@dataclass
class ServiceInstance:
    """Represents a service instance"""
    instance_id: str
    endpoint: str
    health_status: HealthStatus
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_count: int = 0
    error_count: int = 0
    response_time: float = 0.0
    last_health_check: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    weight: float = 1.0  # Load balancing weight
    connections: int = 0  # Current connections


@dataclass
class ScalingEvent:
    """Represents a scaling event"""
    timestamp: datetime
    direction: ScalingDirection
    trigger: ScalingTrigger
    old_capacity: int
    new_capacity: int
    reason: str
    metrics: Dict[str, float]


class LoadBalancer:
    """Load balancer for distributing requests across instances"""
    
    def __init__(self, algorithm: str = "round_robin"):
        self.algorithm = algorithm
        self.instances: Dict[str, ServiceInstance] = {}
        self.current_index = 0
        self.request_counts = defaultdict(int)
        
        logger.info(f"LoadBalancer initialized with {algorithm} algorithm")
    
    def add_instance(self, instance: ServiceInstance):
        """Add instance to load balancer"""
        self.instances[instance.instance_id] = instance
        logger.info(f"Added instance {instance.instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str):
        """Remove instance from load balancer"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            logger.info(f"Removed instance {instance_id} from load balancer")
    
    def get_next_instance(self) -> Optional[ServiceInstance]:
        """Get next instance based on load balancing algorithm"""
        healthy_instances = [
            instance for instance in self.instances.values()
            if instance.health_status == HealthStatus.HEALTHY
        ]
        
        if not healthy_instances:
            return None
        
        if self.algorithm == "round_robin":
            return self._round_robin(healthy_instances)
        elif self.algorithm == "least_connections":
            return self._least_connections(healthy_instances)
        elif self.algorithm == "weighted_round_robin":
            return self._weighted_round_robin(healthy_instances)
        elif self.algorithm == "least_response_time":
            return self._least_response_time(healthy_instances)
        else:
            return healthy_instances[0]  # Fallback to first healthy instance
    
    def _round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin load balancing"""
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance
    
    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections load balancing"""
        return min(instances, key=lambda x: x.connections)
    
    def _weighted_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin load balancing"""
        total_weight = sum(instance.weight for instance in instances)
        target = (self.current_index % total_weight) + 1
        
        current_weight = 0
        for instance in instances:
            current_weight += instance.weight
            if current_weight >= target:
                self.current_index += 1
                return instance
        
        return instances[0]  # Fallback
    
    def _least_response_time(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least response time load balancing"""
        return min(instances, key=lambda x: x.response_time)
    
    def update_instance_metrics(self, instance_id: str, metrics: Dict[str, float]):
        """Update instance metrics"""
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            instance.cpu_usage = metrics.get('cpu_usage', instance.cpu_usage)
            instance.memory_usage = metrics.get('memory_usage', instance.memory_usage)
            instance.response_time = metrics.get('response_time', instance.response_time)
            instance.connections = metrics.get('connections', instance.connections)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        total_instances = len(self.instances)
        healthy_instances = sum(1 for i in self.instances.values() if i.health_status == HealthStatus.HEALTHY)
        
        return {
            'algorithm': self.algorithm,
            'total_instances': total_instances,
            'healthy_instances': healthy_instances,
            'unhealthy_instances': total_instances - healthy_instances,
            'total_requests': sum(self.request_counts.values()),
            'instances': [asdict(instance) for instance in self.instances.values()]
        }


class MetricsCollector:
    """Collects and processes scaling metrics"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: Dict[ScalingTrigger, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.custom_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        
        logger.info("MetricsCollector initialized")
    
    def add_metric(self, trigger: ScalingTrigger, value: float, timestamp: Optional[float] = None):
        """Add metric value"""
        if timestamp is None:
            timestamp = time.time()
        
        metric_value = MetricValue(value, timestamp)
        self.metrics_history[trigger].append(metric_value)
    
    def add_custom_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Add custom metric value"""
        if timestamp is None:
            timestamp = time.time()
        
        metric_value = MetricValue(value, timestamp, name)
        self.custom_metrics[name].append(metric_value)
    
    def get_metric_values(self, trigger: ScalingTrigger, period_seconds: int = 300) -> List[float]:
        """Get metric values for specified period"""
        cutoff_time = time.time() - period_seconds
        values = []
        
        for metric in self.metrics_history[trigger]:
            if metric.timestamp >= cutoff_time:
                values.append(metric.value)
        
        return values
    
    def get_metric_average(self, trigger: ScalingTrigger, period_seconds: int = 300) -> Optional[float]:
        """Get average metric value for period"""
        values = self.get_metric_values(trigger, period_seconds)
        return statistics.mean(values) if values else None
    
    def get_metric_percentile(self, trigger: ScalingTrigger, percentile: float, period_seconds: int = 300) -> Optional[float]:
        """Get percentile value for metric"""
        values = self.get_metric_values(trigger, period_seconds)
        if not values:
            return None
        
        values.sort()
        k = (len(values) - 1) * (percentile / 100)
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return values[int(k)]
        else:
            return values[int(f)] * (c - k) + values[int(c)] * (k - f)
    
    def collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.add_metric(ScalingTrigger.CPU_USAGE, cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.add_metric(ScalingTrigger.MEMORY_USAGE, memory_percent)
            
            logger.debug(f"Collected system metrics: CPU={cpu_percent}%, Memory={memory_percent}%")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class PredictiveScaler:
    """Predictive scaling using historical data"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.prediction_window = 1800  # 30 minutes
        self.min_history_points = 10
        
        logger.info("PredictiveScaler initialized")
    
    def predict_metric_value(self, trigger: ScalingTrigger, forecast_seconds: int = 300) -> Optional[float]:
        """Predict metric value using linear regression"""
        values = self.metrics_collector.get_metric_values(trigger, self.prediction_window)
        
        if len(values) < self.min_history_points:
            return None
        
        # Simple linear regression
        n = len(values)
        x_values = list(range(n))
        
        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        # Calculate slope and intercept
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return y_mean  # No trend, return average
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Predict future value
        future_x = n + (forecast_seconds / (self.prediction_window / n))
        predicted_value = slope * future_x + intercept
        
        # Bound the prediction to reasonable limits
        max_value = max(values) * 1.5
        min_value = min(values) * 0.5
        predicted_value = max(min_value, min(max_value, predicted_value))
        
        logger.debug(f"Predicted {trigger.value}: {predicted_value:.2f}")
        return predicted_value
    
    def should_preemptive_scale(self, rules: List[ScalingRule]) -> Optional[ScalingDirection]:
        """Determine if preemptive scaling is needed"""
        for rule in rules:
            if not rule.enabled:
                continue
            
            predicted_value = self.predict_metric_value(rule.trigger)
            if predicted_value is None:
                continue
            
            # Check if predicted value would trigger scaling
            if predicted_value > rule.threshold_up:
                logger.info(f"Preemptive scale up recommended for {rule.trigger.value}: predicted {predicted_value:.2f}")
                return ScalingDirection.UP
            elif predicted_value < rule.threshold_down:
                logger.info(f"Preemptive scale down recommended for {rule.trigger.value}: predicted {predicted_value:.2f}")
                return ScalingDirection.DOWN
        
        return None


class AutoScaler:
    """Main auto-scaling controller"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.current_capacity = 1
        self.min_capacity = 1
        self.max_capacity = 10
        self.target_capacity = 1
        
        # Components
        self.metrics_collector = MetricsCollector()
        self.load_balancer = LoadBalancer()
        self.predictive_scaler = PredictiveScaler(self.metrics_collector)
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = []
        self.last_scaling_events: Dict[str, datetime] = {}
        self.scaling_history: deque = deque(maxlen=100)
        
        # Control flags
        self.is_running = False
        self.enable_predictive_scaling = True
        self.enable_cost_optimization = True
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.scaling_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Default scaling rules
        self._setup_default_rules()
        
        logger.info(f"AutoScaler initialized for service '{service_name}'")
    
    def _setup_default_rules(self):
        """Setup default scaling rules"""
        # CPU usage rule
        cpu_rule = ScalingRule(
            name="cpu_usage",
            trigger=ScalingTrigger.CPU_USAGE,
            threshold_up=70.0,
            threshold_down=30.0,
            evaluation_periods=3,
            cooldown_seconds=300
        )
        
        # Memory usage rule
        memory_rule = ScalingRule(
            name="memory_usage",
            trigger=ScalingTrigger.MEMORY_USAGE,
            threshold_up=80.0,
            threshold_down=40.0,
            evaluation_periods=3,
            cooldown_seconds=300
        )
        
        # Request rate rule (placeholder - would need actual request metrics)
        request_rule = ScalingRule(
            name="request_rate",
            trigger=ScalingTrigger.REQUEST_RATE,
            threshold_up=1000.0,  # requests per minute
            threshold_down=200.0,
            evaluation_periods=2,
            cooldown_seconds=180
        )
        
        self.scaling_rules = [cpu_rule, memory_rule, request_rule]
    
    async def start(self):
        """Start the auto-scaler"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"AutoScaler started for service '{self.service_name}'")
    
    async def stop(self):
        """Stop the auto-scaler"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        tasks = [self.monitoring_task, self.scaling_task, self.health_check_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info(f"AutoScaler stopped for service '{self.service_name}'")
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add custom scaling rule"""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove scaling rule"""
        self.scaling_rules = [rule for rule in self.scaling_rules if rule.name != rule_name]
        logger.info(f"Removed scaling rule: {rule_name}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        logger.debug("Started monitoring loop")
        
        while self.is_running:
            try:
                # Collect system metrics
                self.metrics_collector.collect_system_metrics()
                
                # Collect custom metrics from performance monitor
                perf_monitor = get_monitor()
                if hasattr(perf_monitor, 'get_performance_report'):
                    report = perf_monitor.get_performance_report(hours=1)
                    if 'overall_stats' in report:
                        stats = report['overall_stats']
                        if 'p95_duration' in stats:
                            self.metrics_collector.add_metric(
                                ScalingTrigger.RESPONSE_TIME,
                                stats['p95_duration'] * 1000  # Convert to ms
                            )
                        if 'success_rate' in stats:
                            error_rate = (1.0 - stats['success_rate']) * 100
                            self.metrics_collector.add_metric(
                                ScalingTrigger.ERROR_RATE,
                                error_rate
                            )
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _scaling_loop(self):
        """Background scaling evaluation loop"""
        logger.debug("Started scaling loop")
        
        while self.is_running:
            try:
                # Evaluate scaling rules
                scaling_decision = await self._evaluate_scaling_rules()
                
                if scaling_decision != ScalingDirection.STABLE:
                    await self._execute_scaling_decision(scaling_decision)
                
                # Check predictive scaling
                if self.enable_predictive_scaling:
                    predictive_decision = self.predictive_scaler.should_preemptive_scale(self.scaling_rules)
                    if predictive_decision and predictive_decision != ScalingDirection.STABLE:
                        logger.info(f"Predictive scaling suggests: {predictive_decision.value}")
                        # Could implement predictive scaling here
                
                await asyncio.sleep(60)  # Evaluate every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_loop(self):
        """Background health check loop"""
        logger.debug("Started health check loop")
        
        while self.is_running:
            try:
                # Check health of all instances
                for instance in self.load_balancer.instances.values():
                    await self._check_instance_health(instance)
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_scaling_rules(self) -> ScalingDirection:
        """Evaluate all scaling rules to determine scaling decision"""
        scale_up_votes = 0
        scale_down_votes = 0
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            last_event_time = self.last_scaling_events.get(rule.name)
            if last_event_time:
                time_since_last = (datetime.now() - last_event_time).total_seconds()
                if time_since_last < rule.cooldown_seconds:
                    continue
            
            # Get metric values for evaluation period
            period_seconds = rule.evaluation_periods * 60  # Convert periods to seconds
            
            if rule.custom_evaluator:
                values = self.metrics_collector.get_metric_values(rule.trigger, period_seconds)
                if rule.custom_evaluator(values):
                    scale_up_votes += 1
            else:
                avg_value = self.metrics_collector.get_metric_average(rule.trigger, period_seconds)
                if avg_value is None:
                    continue
                
                if avg_value > rule.threshold_up:
                    scale_up_votes += 1
                    logger.debug(f"Rule {rule.name}: {avg_value:.2f} > {rule.threshold_up} (scale up)")
                elif avg_value < rule.threshold_down:
                    scale_down_votes += 1
                    logger.debug(f"Rule {rule.name}: {avg_value:.2f} < {rule.threshold_down} (scale down)")
        
        # Make scaling decision
        if scale_up_votes > 0 and self.current_capacity < self.max_capacity:
            return ScalingDirection.UP
        elif scale_down_votes > scale_up_votes and self.current_capacity > self.min_capacity:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    async def _execute_scaling_decision(self, direction: ScalingDirection):
        """Execute scaling decision"""
        if direction == ScalingDirection.UP:
            new_capacity = min(self.current_capacity + 1, self.max_capacity)
        else:
            new_capacity = max(self.current_capacity - 1, self.min_capacity)
        
        if new_capacity == self.current_capacity:
            return
        
        # Create scaling event
        current_metrics = {}
        for trigger in [ScalingTrigger.CPU_USAGE, ScalingTrigger.MEMORY_USAGE]:
            value = self.metrics_collector.get_metric_average(trigger, 300)
            if value is not None:
                current_metrics[trigger.value] = value
        
        event = ScalingEvent(
            timestamp=datetime.now(),
            direction=direction,
            trigger=ScalingTrigger.CPU_USAGE,  # Primary trigger for logging
            old_capacity=self.current_capacity,
            new_capacity=new_capacity,
            reason=f"Auto-scaling based on metrics evaluation",
            metrics=current_metrics
        )
        
        # Execute scaling (this would integrate with container orchestration)
        success = await self._scale_instances(new_capacity)
        
        if success:
            self.current_capacity = new_capacity
            self.scaling_history.append(event)
            
            # Update last scaling event times
            for rule in self.scaling_rules:
                self.last_scaling_events[rule.name] = datetime.now()
            
            logger.info(f"Scaled {direction.value}: {event.old_capacity} -> {event.new_capacity}")
        else:
            logger.error(f"Failed to scale {direction.value} to {new_capacity} instances")
    
    async def _scale_instances(self, target_capacity: int) -> bool:
        """Scale service instances (integration point for orchestration)"""
        try:
            current_instances = len(self.load_balancer.instances)
            
            if target_capacity > current_instances:
                # Scale up - add instances
                for i in range(target_capacity - current_instances):
                    instance_id = f"{self.service_name}-{int(time.time())}-{i}"
                    endpoint = f"http://service-{instance_id}:8000"  # Example endpoint
                    
                    instance = ServiceInstance(
                        instance_id=instance_id,
                        endpoint=endpoint,
                        health_status=HealthStatus.UNKNOWN
                    )
                    
                    # This would typically call container orchestrator API
                    # await self._create_container_instance(instance)
                    
                    self.load_balancer.add_instance(instance)
                    logger.info(f"Added instance {instance_id}")
            
            elif target_capacity < current_instances:
                # Scale down - remove instances
                instances_to_remove = list(self.load_balancer.instances.keys())[:current_instances - target_capacity]
                
                for instance_id in instances_to_remove:
                    # This would typically call container orchestrator API
                    # await self._remove_container_instance(instance_id)
                    
                    self.load_balancer.remove_instance(instance_id)
                    logger.info(f"Removed instance {instance_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error scaling instances: {e}")
            return False
    
    async def _check_instance_health(self, instance: ServiceInstance):
        """Check health of a service instance"""
        try:
            # This would make actual health check request
            # For now, simulate health check
            
            # Update health check timestamp
            instance.last_health_check = datetime.now()
            
            # Simulate health check logic
            if instance.cpu_usage > 95 or instance.memory_usage > 95:
                instance.health_status = HealthStatus.UNHEALTHY
            elif instance.cpu_usage > 85 or instance.memory_usage > 85:
                instance.health_status = HealthStatus.DEGRADED
            else:
                instance.health_status = HealthStatus.HEALTHY
            
        except Exception as e:
            logger.error(f"Health check failed for instance {instance.instance_id}: {e}")
            instance.health_status = HealthStatus.UNHEALTHY
    
    async def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics"""
        recent_events = list(self.scaling_history)[-10:]  # Last 10 events
        
        return {
            'service_name': self.service_name,
            'current_capacity': self.current_capacity,
            'target_capacity': self.target_capacity,
            'min_capacity': self.min_capacity,
            'max_capacity': self.max_capacity,
            'is_running': self.is_running,
            'scaling_rules_count': len([r for r in self.scaling_rules if r.enabled]),
            'recent_scaling_events': [asdict(event) for event in recent_events],
            'load_balancer_stats': self.load_balancer.get_stats(),
            'current_metrics': {
                'cpu_usage': self.metrics_collector.get_metric_average(ScalingTrigger.CPU_USAGE, 300),
                'memory_usage': self.metrics_collector.get_metric_average(ScalingTrigger.MEMORY_USAGE, 300),
                'response_time': self.metrics_collector.get_metric_average(ScalingTrigger.RESPONSE_TIME, 300),
                'error_rate': self.metrics_collector.get_metric_average(ScalingTrigger.ERROR_RATE, 300)
            }
        }


# Global auto-scaler instance
_auto_scaler: Optional[AutoScaler] = None
_scaler_lock = asyncio.Lock()


async def get_auto_scaler(service_name: str = "claude-manager") -> AutoScaler:
    """Get global auto-scaler instance"""
    global _auto_scaler
    
    if _auto_scaler is None:
        async with _scaler_lock:
            if _auto_scaler is None:
                _auto_scaler = AutoScaler(service_name)
                await _auto_scaler.start()
    
    return _auto_scaler


# Integration functions
@with_enhanced_error_handling("auto_scaling")
async def trigger_scaling_evaluation():
    """Manually trigger scaling evaluation"""
    scaler = await get_auto_scaler()
    decision = await scaler._evaluate_scaling_rules()
    
    if decision != ScalingDirection.STABLE:
        await scaler._execute_scaling_decision(decision)
    
    return decision


async def add_custom_scaling_rule(name: str, 
                                 trigger: ScalingTrigger,
                                 threshold_up: float,
                                 threshold_down: float,
                                 evaluation_periods: int = 3):
    """Add custom scaling rule"""
    scaler = await get_auto_scaler()
    
    rule = ScalingRule(
        name=name,
        trigger=trigger,
        threshold_up=threshold_up,
        threshold_down=threshold_down,
        evaluation_periods=evaluation_periods,
        cooldown_seconds=300
    )
    
    scaler.add_scaling_rule(rule)
    logger.info(f"Added custom scaling rule: {name}")


async def get_load_balancer_instance(request_data: Optional[Dict] = None) -> Optional[ServiceInstance]:
    """Get next instance from load balancer for request routing"""
    scaler = await get_auto_scaler()
    return scaler.load_balancer.get_next_instance()


async def report_instance_metrics(instance_id: str, metrics: Dict[str, float]):
    """Report metrics for a specific instance"""
    scaler = await get_auto_scaler()
    scaler.load_balancer.update_instance_metrics(instance_id, metrics)